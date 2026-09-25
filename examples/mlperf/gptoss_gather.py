"""Persistent expert shards gathered lazily by their next forward consumers."""
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv

class DeferredExpertGather:
  # Transient coordination must not enter the model/optimizer checkpoint state.
  __slots__ = ('entries', 'dirty')

  def __init__(self, optimizer, model):
    from examples.mlperf import optim as implementation
    assert implementation.FUSED_ADAM_MXFP8 and implementation.MXFP8 and not implementation.PRESTORE_WT
    self.entries, self.dirty = {}, False
    for weights in zip(model.w_gate_up, model.w_down):
      for parameter in weights:
        matches = [o for o in optimizer.optimizers if any(p is parameter for p in o.params)]
        assert len(matches) == 1 and matches[0].zero
        opt = matches[0]
        compact = bool(getenv('GPTOSS_COMPACT_Q_GATHER', 0))
        q = parameter
        if compact:
          rows = 5760 if hasattr(parameter, '_fc1_packed_si') else 2880
          q = q[:, :rows, :2880].contiguous().bitcast(dtypes.uint32)
        targets = [parameter, parameter._inv_scale]
        values = [q, parameter._inv_scale]
        if hasattr(parameter, '_fc1_packed_si'):
          targets.append(parameter._fc1_packed_si)
          values.append(parameter._fc1_packed_si)
        mailboxes = [opt._zero_shard(v).realize() for v in values]
        entry = (targets, mailboxes, compact, opt)
        self.entries[id(parameter)] = entry
        assert getattr(opt, '_deferred_experts', self) is self
        opt._deferred_experts = self

  def stage(self, parameter, values):
    targets, mailboxes, _, _ = self.entries[id(parameter)]
    assert len(values) == len(targets)
    for mailbox, value in zip(mailboxes, values):
      assert (mailbox.shape, mailbox.dtype, mailbox.uop.axis) == (value.shape, value.dtype, value.uop.axis)
      mailbox.assign(value)

  def scheduled_outputs(self, outputs, optimizer):
    entries = [entry for entry in self.entries.values() if entry[3] is optimizer]
    target_ids = {id(t) for targets, _, _, _ in entries for t in targets}
    return [t for t in outputs if id(t) not in target_ids] + [m for _, mailboxes, _, _ in entries for m in mailboxes]

  def prefetch(self):
    from examples.mlperf import optim as implementation
    # Assign lazy gathers together so the forward graph can start every ready
    # transfer. Each layer depends only on the parameters it actually consumes.
    for targets, mailboxes, compact, opt in self.entries.values():
      gather = implementation._gptoss_gather_owned if compact and getenv('GPTOSS_OWNED_EXPERT_GATHER', 0) else opt._zero_gather
      for i, (target, mailbox) in enumerate(zip(targets, mailboxes)):
        value = gather(mailbox)
        if i == 0 and compact:
          value = value.pad(((0, 0), (0, target.shape[1]-value.shape[1]), (0, target.shape[2]//4-value.shape[2]))).bitcast(target.dtype)
        target.assign(value.reshape(target.shape))

  def mark_updated(self): self.dirty = True

  def drain(self):
    if not self.dirty: return
    self.prefetch()
    Tensor.realize(*[t for targets, _, _, _ in self.entries.values() for t in targets])
    self.dirty = False
