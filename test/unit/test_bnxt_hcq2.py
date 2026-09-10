import contextlib, itertools, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tinygrad import Device, dtypes
from tinygrad.device import Buffer, BufferStorage
from tinygrad.engine.realize import lower_and_compile, run_linear
from tinygrad.runtime.ops_amd import AMDComputeQueue
from tinygrad.runtime.ops_bnxt import BNXTAllocator
from tinygrad.runtime.support.bnxt import send_wqe, recv_wqe, msn_entry
from tinygrad.runtime.support.hcq2 import HCQInfo, hcq_link, lower_call, rt_addr
from tinygrad.runtime.support.memory import AddrSpace, VirtMapping
from tinygrad.runtime.support.system import PCIIfaceBase, PCIAllocationMeta
from tinygrad.uop.ops import UOp, Ops, KernelInfo

class TestBNXTAllocator(unittest.TestCase):
  def setUp(self):
    self.iface = PCIIfaceBase.__new__(PCIIfaceBase)
    self.iface.pci_dev = SimpleNamespace(peer_group="node")
    self.iface.p2p_paddrs = lambda pages: ([(p + 0x100000000, s) for p, s in pages], False)
    self.nic = SimpleNamespace(peer_group="node", iface=SimpleNamespace(dev_impl=Mock()))
    self.nic.iface.dev_impl.register_mem.return_value = 0x1234
    self.nic.allocator = BNXTAllocator(self.nic)
    gpu = SimpleNamespace(iface=self.iface, allocator=Mock(_offset=lambda b, size, off: b + off))
    self.lookup = patch.object(type(Device), "__getitem__", lambda _, d: {"AMD": gpu, "BNXT": self.nic}[d])
    self.lookup.start()
    self.addCleanup(self.lookup.stop)

  def buffer(self, pages, va=0x200000, aspace=AddrSpace.PHYS):
    size = sum(s for _, s in pages)
    buf = Buffer("AMD", size, dtypes.uint8, opaque=BufferStorage(va, PCIAllocationMeta(VirtMapping(va, size, pages, aspace), False)))
    self.addCleanup(buf.deallocate)
    return buf

  def test_vram_huge_pages_and_views(self):
    buf = self.buffer([(0x400000, 0x400000)])
    view = buf.view(16, dtypes.uint8, 128).ensure_allocated()
    self.addCleanup(view.deallocate)
    self.assertEqual((view.get_buf("AMD"), view.get_buf("BNXT"), buf.get_buf("BNXT")), (0x200080, 0x1234, 0x1234))
    self.nic.iface.dev_impl.register_mem.assert_called_once_with([0x100400000, 0x100600000], 0x400000, 21, va=0x200000)
    self.nic.allocator._unmap(buf.get_storage("BNXT"))
    self.nic.iface.dev_impl.unregister_mem.assert_called_once_with(0x1234)

  def test_fragmented_pages(self):
    buf = self.buffer([(0x401000, 0x1000), (0x800000, 0x2000)])
    self.assertEqual(buf.get_buf("BNXT"), 0x1234)
    self.nic.iface.dev_impl.register_mem.assert_called_once_with([0x100401000, 0x100800000, 0x100801000], 0x3000, 12, va=0x200000)

  def test_sysmem_does_not_add_bar(self):
    self.buffer([(0x401000, 0x2000)], aspace=AddrSpace.SYS).get_buf("BNXT")
    self.nic.iface.dev_impl.register_mem.assert_called_once_with([0x401000, 0x402000], 0x2000, 12, va=0x200000)

  def test_reject_other_node(self):
    self.nic.peer_group = "other"
    with self.assertRaisesRegex(RuntimeError, "memory on its node"): self.buffer([(0x400000, 0x1000)]).get_buf("BNXT")
    self.nic.iface.dev_impl.register_mem.assert_not_called()

class TestBNXTEncode(unittest.TestCase):
  def test_packed_address_words(self):
    for inputs in (False, True):
      with self.subTest(inputs=inputs):
        count = 1100
        data = [Buffer("CPU", count, dtypes.uint8, preallocate=True) for _ in range(2)]
        src = UOp.param(0, dtypes.uint8, count, device="CPU") if inputs else UOp.from_buffer(data[0])
        out = UOp.placeholder((count,), dtypes.uint64, device="CPU", tag="result")
        body = UOp.sink(*[out.index(i).store(rt_addr(src[i:i+1])) for i in range(count)], arg=KernelInfo("packed_addrs"), tag=1)
        lowered = lower_call(body.call(aux=HCQInfo(("CPU",))))
        self.assertLessEqual(lowered.without_after.arg.aux.nargs, 3)
        linked = hcq_link(lower_and_compile(UOp(Ops.LINEAR, src=(lowered,))), allow_cache=False)
        result = next(b.buffer for p, b in zip(lowered.without_after.src[1:], linked.src[0].without_after.src[1:]) if p.tag == "result")
        for buf in data if inputs else data[:1]:
          run_linear(linked, jit=True, input_uops=[UOp.from_buffer(buf)])
          self.assertEqual(result.host.view(fmt="Q")[:], [buf._buf + i for i in range(count)])

  def test_post_replay_wrap(self):
    for recv, inputs, chunks in itertools.product((False, True), (False, True), (1, 2, 32)):
      with self.subTest(recv=recv, inputs=inputs, chunks=chunks):
        args = {name: UOp.placeholder((size,), dtype, device="CPU", volatile=True, tag=name) for name, size, dtype in
                [(n, 8192 if n == "sq" else 4096, dtypes.uint8) for n in ("sq", "rq", "scq", "rcq", "db")] +
                [(n, 1, dtypes.uint64) for n in ("sq_prod", "sq_psn", "rq_prod", "scq_cons", "rcq_cons")]}
        nic = SimpleNamespace(device="CPU", arg=lambda pair, name: args[name], iface=SimpleNamespace(dev_impl=SimpleNamespace(db_off=0)),
                              qp=lambda *a: SimpleNamespace(qpn=5, scq_id=6, rcq_id=7))
        q = AMDComputeQueue.__new__(AMDComputeQueue)
        q.dev, q.devs, q.ctx = SimpleNamespace(device="CPU", nic=nic), ("CPU",), SimpleNamespace(host="CPU")
        q.pm4 = SimpleNamespace(data_sel__mec_release_mem__send_64_bit_data=2, int_sel__mec_release_mem__none=0)
        q.nic_posts, q.nic_counts = [], {}
        q.pred_exec, q.release_mem, q.wait_reg_mem, q.acquire_mem = lambda **kw: contextlib.nullcontext(), Mock(), Mock(), Mock()
        src, dst, alt_src, alt_dst = [Buffer("CPU", 8192, dtypes.uint8, preallocate=True) for _ in range(4)]
        a, b = [UOp.param(i, dtypes.uint8, 8192, device="CPU") for i in range(2)] if inputs else [UOp.from_buffer(x) for x in (src, dst)]
        call = a.copy_to_device("CPU").replace(arg="recv" if recv else None).call(b, a)
        with patch("tinygrad.runtime.ops_amd.is_rdma", return_value=True), patch("tinygrad.runtime.ops_amd.RDMA_CHUNK", 8192 // chunks):
          q.copy(call)
        checks = UOp.placeholder((4,), dtypes.uint64, device="CPU", tag="cmdbuf")
        values = [x.args[1] for x in q.release_mem.call_args_list[-2:]] + [q.wait_reg_mem.call_args.kwargs["mem"], q.wait_reg_mem.call_args.args[0]]
        out = q.nic_post(checks.after(*[checks.index(i).store(v) for i, v in enumerate(values)]))
        self.assertEqual(q.wait_reg_mem.call_args.kwargs["mask"], 0xff01)
        lowered = lower_call(UOp.sink(out.index(0).load(), arg=KernelInfo("nic_post_test"), tag=1).call(aux=HCQInfo(("CPU",))))
        linked = hcq_link(lower_and_compile(UOp(Ops.LINEAR, src=(lowered,))), allow_cache=False)
        buffers = {p.tag: b.buffer for p, b in zip(lowered.without_after.src[1:], linked.src[0].without_after.src[1:])}
        ring, prod, cons = ("rq", "rq_prod", "rcq_cons") if recv else ("sq", "sq_prod", "scq_cons")
        for name in (prod, cons): buffers[name].host.view(fmt="Q")[0] = 0
        if not recv: buffers["sq_psn"].host.view(fmt="Q")[0] = 0xfffffe
        advance = max(1, 2 // chunks)
        for i in range(130):
          current = (alt_src, alt_dst) if inputs and i % 2 else (src, dst)
          run_linear(linked, jit=True, input_uops=[UOp.from_buffer(x) for x in current])
          end = (i + 1) * chunks
          self.assertEqual(buffers[prod].host.view(fmt="Q")[0], end)
          self.assertEqual(buffers[cons].host.view(fmt="Q")[0], end)
          doorbell, cq_doorbell, cq_addr, toggle = buffers["cmdbuf"].host.view(fmt="Q")[:]
          self.assertEqual(doorbell & 0xffffffff, end % 32 | ((end // 32) & 1) << 24)
          self.assertEqual(cq_doorbell & 0xffffffff, end % 128 | ((end // 128) & 1) << 24)
          if i == 0: cq_base = cq_addr - 24 - (chunks - 1) * 32
          self.assertEqual((cq_addr, toggle), (cq_base + ((end - 1) % 128) * 32 + 24, 1 ^ (((end - 1) // 128) & 1)))
          data = current[1 if recv else 0]
          for chunk in range(chunks):
            slot = (i * chunks + chunk) % 32
            wqe = (recv_wqe if recv else send_wqe)(data._buf + chunk * (8192 // chunks), data._buf & 0xffffffff, 8192 // chunks)
            self.assertEqual(bytes(buffers[ring].host[slot*128:slot*128+48]), wqe)
            if not recv:
              start = (0xfffffe + (i * chunks + chunk) * advance) & 0xffffff
              self.assertEqual(buffers[ring].host.view(fmt="Q")[512 + slot], msn_entry(slot, start, 8192 // chunks)[0])
          if not recv: self.assertEqual(buffers["sq_psn"].host.view(fmt="Q")[0], 0xfffffe + (i + 1) * chunks * advance)


if __name__ == "__main__": unittest.main()
