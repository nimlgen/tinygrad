import unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Device, dtypes
from tinygrad.uop.ops import UOp, Ops
from tinygrad.runtime.support import hcq2

class TestSDMAPeerGroups(unittest.TestCase):
  def assignments(self, groups, queues=8, shift=0, nic=False):
    membership = {d: str(i) for i,group in enumerate(groups) for d in group}
    class Devices:
      canonicalize = staticmethod(Device.canonicalize)
      def __getitem__(self, d): return SimpleNamespace(peer_group=membership[d])
    buffers = {d: UOp.new_buffer(d, 256, dtypes.uint8) for d in membership}
    calls = [buffers[dst].store_call(buffers[src]) for group in groups for src in group for dst in group if src != dst]
    if nic:
      wire = UOp.new_buffer('RDMA', 256, dtypes.uint8)
      calls += [wire.store_call(buffers['AMD']), buffers['AMD'].store_call(wire)]
    recorded = []
    def finalize(ctx, skip_wait):
      recorded.extend(ctx.batch)
      return ctx.batch[0][0]
    knobs = {'HCQ_NUM_SDMA': queues, 'HCQ_QUEUE_SHIFT': shift}
    with patch.object(hcq2, 'Device', Devices()), patch.object(hcq2, 'HCQ_DEVS', hcq2.HCQ_DEVS | {'AMD'}), \
         patch.object(hcq2, 'getenv', lambda k,default=0: knobs.get(k,default)), \
         patch.object(hcq2, 'BatchCtx', lambda batch,profile: SimpleNamespace(batch=batch)), \
         patch.object(hcq2, '_finalize_batch', finalize):
      hcq2.sched_batches(UOp(Ops.LINEAR, src=tuple(calls)), False)
    self.assertEqual(len(recorded), len(calls))
    return {(c.src[2].device, c.src[1].device): q for c,_,q in recorded}

  def test_two_nodes_use_distinct_queues_for_each_local_peer(self):
    groups = [[Device.canonicalize(f'AMD:{i}') for i in range(start,start+8)] for start in (0,8)]
    for shift in (0,1,7):
      with self.subTest(shift=shift):
        assigned = self.assignments(groups, shift=shift)
        for group in groups:
          for src in group:
            self.assertEqual(len({assigned[src,dst] for dst in group if src != dst}), 7)
          for dst in group:
            self.assertEqual(len({assigned[src,dst] for src in group if src != dst}), 7)

  def test_single_node_and_single_queue(self):
    group = [Device.canonicalize(f'AMD:{i}') for i in range(8)]
    assigned = self.assignments([group])
    self.assertEqual(assigned['AMD','AMD:1'], 'COPY:0')
    self.assertEqual(assigned['AMD:1','AMD'], 'COPY:6')
    self.assertEqual(set(self.assignments([group], queues=1).values()), {'COPY:0'})

  def test_nic_send_and_receive_keep_dedicated_queues(self):
    assigned = self.assignments([['AMD','AMD:1']], nic=True)
    self.assertEqual(assigned['AMD','RDMA'], 'COPY:8')
    self.assertEqual(assigned['RDMA','AMD'], 'COPY:9')

if __name__ == '__main__': unittest.main()
