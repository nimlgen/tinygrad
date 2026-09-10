import unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, KernelInfo, ProgramInfo, graph_rewrite
from tinygrad.runtime.support import hcq2
from tinygrad.engine import realize


def copy(src, dst): return src.copy_to_device(dst.device).call(dst, src)
def buf(slot, device, size=16): return UOp.param(slot, dtypes.uint8, size, device)
def kernel(b, write=True): return UOp(Ops.PROGRAM, arg=ProgramInfo(outs=(0,) if write else (), ins=() if write else (0,))).call(b)
def queued(batch): return [c for c in batch.src[0].toposort() if c.op is Ops.CALL and c.src[0].op is Ops.COPY]

class TestRDMASchedule(unittest.TestCase):
  def setUp(self):
    self.enterContext(patch.object(hcq2, "getenv", return_value=1))
    self.devs = {d: SimpleNamespace(peer_group=g, remote_peer=None, has_copy_queue=True, pm_batch=None)
                 for d, g in (("AMD:1", "a"), ("AMD:2", "b"), ("AMD:3", "a"))}
    get_device = type(Device).__getitem__
    self.enterContext(patch.object(type(Device), "__getitem__", lambda obj, d: self.devs[d] if d in self.devs else get_device(obj, d)))

  def prepare(self, calls):
    return graph_rewrite(UOp(Ops.LINEAR, src=tuple(calls)), hcq2.pm_unwrap_multi+hcq2.pm_split_rdma+realize.pm_flatten_linear)

  def test_split(self):
    src, dst = buf(0, "AMD:1"), buf(1, "AMD:2")
    send, recv = self.prepare([copy(src, dst)]).src
    self.assertEqual((send.src[0].arg, recv.src[0].arg), ("send", "recv"))
    self.assertEqual((hcq2.get_enqueue_devs(send), hcq2.get_enqueue_devs(recv)), ("AMD:1", "AMD:2"))
    self.assertIsNone(hcq2.stage_copy((), send, dst, src))
    self.assertIsNone(hcq2.split_rdma(send))
    self.assertIsNone(hcq2.split_rdma(copy(src, buf(2, "AMD:3"))))
    with patch.object(hcq2, "getenv", return_value=0): self.assertIsNone(hcq2.split_rdma(copy(src, dst)))

  def test_dependencies_stay_on_each_node(self):
    src, dst = buf(0, "AMD:1"), buf(1, "AMD:2")
    send, recv = self.prepare([copy(src, dst)]).src
    calls = [(kernel(src), ("AMD:1",), "COPY:0"), (kernel(dst, False), ("AMD:2",), "COPY:0"),
             (send, ("AMD:1",), "COMPUTE:0"), (recv, ("AMD:2",), "COMPUTE:0"),
             (kernel(src), ("AMD:1",), "COPY:0"), (kernel(dst, False), ("AMD:2",), "COPY:0")]
    ctx = hcq2.BatchCtx(calls, False, rdma=True)
    waits = [hcq2._wait_ins(ctx, c, ds[0], q, i) for i, (c, ds, q) in enumerate(calls)]
    self.assertEqual([[w.src[1].val for w in ws] for ws in waits], [[], [], [1], [2], [3], [4]])
    for (_, ds, _), ws in zip(calls, waits):
      self.assertTrue(all(self.devs[w.src[0].device[0]].peer_group == self.devs[ds[0]].peer_group for w in ws))

  def test_ring_capacity(self):
    for sizes in ([16] * 65, [65 * hcq2.RDMA_CHUNK]):
      with self.subTest(sizes=sizes):
        linear = self.prepare([copy(buf(2*i, "AMD:1", n), buf(2*i+1, "AMD:2", n)) for i, n in enumerate(sizes)])
        batches = hcq2.sched_batches(linear, False).src
        self.assertEqual([b.arg.aux.device for b in batches], [("AMD:1",), ("AMD:2",)] * 3)
        self.assertTrue(all(b.arg.aux.rdma and not b.arg.aux.host_deps for b in batches))
        counts = [sum(hcq2.ceildiv(c.src[2].nbytes(), hcq2.RDMA_CHUNK) for c in queued(b)) for b in batches]
        self.assertEqual(counts, [32, 32, 32, 32, 1, 1])
        for b in batches:
          self.assertTrue(all(u.src[0].arg[1] == "COMPUTE:0" for u in b.src[0].toposort()
                              if u.op is Ops.CUSTOM_FUNCTION and str(u.arg).startswith("submit_")))

  def test_ring_capacity_device_aliases(self):
    self.devs["AMD"] = self.devs["AMD:0"] = self.devs["AMD:1"]
    linear = self.prepare([copy(buf(2*i, "AMD" if i % 2 else "AMD:0"), buf(2*i+1, "AMD:2")) for i in range(33)])
    self.assertEqual([len(queued(b)) for b in hcq2.sched_batches(linear, False).src], [32, 32, 1, 1])

  def test_ring_reuse_fences_all_devices(self):
    devs = ("AMD:1", "AMD:3")
    slots = [UOp.placeholder((2,), dtypes.uint64, device=(d,), tag="slots") for d in devs]
    fence = hcq2.hcq_fence(SimpleNamespace(devs=devs, rdma=True), UOp.custom_function("hcq_fence", *slots))
    waits = [u for u in fence.toposort() if u.op is Ops.CMPLT]
    self.assertEqual(len(waits), 2)
    self.assertEqual([w.src[1].src[0].src[0].without_after for w in waits], [hcq2.timeline((d,)) for d in devs])
    self.assertIn(waits[0], waits[1].src[1].toposort())
    fence = graph_rewrite(fence, hcq2.pm_patches, ctx=SimpleNamespace(lt_patches=[]))
    params = {b: UOp.param(i, b.dtype, b.shape, device="CPU", volatile=True) for i, b in
              enumerate(u for u in fence.toposort() if u.op is Ops.PARAM)}
    call = UOp.sink(fence.substitute(params), arg=KernelInfo("rdma_fence")).call(*params.values())
    realize.lower_and_compile(UOp(Ops.LINEAR, src=(call,)))

  def test_wait_posts_both_nodes_first(self):
    events = []
    calls = [UOp(Ops.PROGRAM, arg=ProgramInfo(name=d)).call(aux=hcq2.HCQInfo((d,), rdma=True)) for d in ("AMD:1", "AMD:2")]
    def execute(call, ctx):
      self.assertFalse(ctx.wait)
      events.append(("submit", call.arg.aux.device))
      return [None]
    def finish(ctx, call, ets):
      self.assertTrue(ctx.wait)
      events.append(("wait", call.arg.aux.device))
      return ets
    with patch.object(realize.pm_exec, "rewrite", side_effect=execute), patch.object(realize, "finish_hcq", side_effect=finish), \
         patch.object(realize, "track_stats"):
      realize.run_linear(UOp(Ops.LINEAR, src=tuple(calls)), jit=True, wait=True)
    self.assertEqual(events, [(op, (d,)) for op in ("submit", "wait") for d in ("AMD:1", "AMD:2")])

if __name__ == "__main__": unittest.main()
