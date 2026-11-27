"""Single-rank benchmark: connect FP32 accumulators directly to optimizer replay.

Requires the benchmark's batched-Muon detector snapshot. No model, optimizer
arithmetic, optimizer state, gradient precision or checkpoint format changes.
"""
import torch
import json


def gradient_storage(grads):
    """Cold diagnostic: count provably adjacent leaves, never infer aliasing
    from nearby addresses in different allocations. No values or host sync."""
    buckets={}
    singles=0
    for g in grads:
        if not g.is_contiguous() or not g.numel():
            singles+=1
            continue
        st=g.untyped_storage()
        key=(g.device,g.dtype,st.data_ptr())
        buckets.setdefault(key,[]).append((g.storage_offset(),g.numel(),g.element_size()))
    # A different ordering in one future multi-tensor launch is legal;
    # report the potential, not that today's foreach already uses it.
    runs=singles;joined_bytes=0;joined_leaves=0
    for parts in buckets.values():
        count=size=0;end=None
        for off,n,item in sorted(parts):
            if off!=end:
                if count>1:joined_bytes+=size;joined_leaves+=count
                count=size=0;runs+=1
            count+=1;size+=n*item;end=off+n
        if count>1:joined_bytes+=size;joined_leaves+=count
    return dict(leaves=len(grads),contiguous_spans=runs,
                leaves_in_shared_spans=joined_leaves,bytes_in_shared_spans=joined_bytes)


def bind(update, accumulator):
    if update.graph is not None or accumulator.reduce_at_finish:
        raise ValueError("Expected an unprepared single-rank optimizer")
    if tuple(map(id, accumulator.parameters)) != tuple(map(id, update.parameters)):
        raise ValueError("Accumulator and optimizer must share ordered parameters")
    if any(g.dtype != torch.float32 or (g.shape,g.device)!=(p.shape,p.device)
           for p,g in zip(update.masters,accumulator.buffers,strict=True)):
        raise ValueError("Expected FP32 accumulation buffers matching the optimizer masters")
    update._rtx_accumulator = accumulator


def bound_class(base):
    class BoundUpdate(base):
        @torch.no_grad()
        def step(self, loss=None, *, grads=None):
            acc=self._rtx_accumulator
            if grads is not acc.buffers or loss is None or loss.numel()!=1:
                raise ValueError("Bound update requires the accumulator's tuple and scalar loss")
            self.lr.fill_(self.schedule.lr_at(self.update_index))
            for key,lr in self.group_lrs.items():
                lr.fill_(self.group_schedules[key].lr_at(self.update_index))
            if self.graph is None:
                # Let the actual trainer allocate/restore all optimizer states
                # and compile its arithmetic. Capture never mutates parameters
                # until replay; there is no extra optimizer step at binding.
                self._prepare()
                self._bound_loss=torch.zeros_like(loss.detach())
                def diagnostic():
                    m,finite=self.diagnostics(self.master_grads,self.lr,self._bound_loss)
                    torch._assert_async(finite,"Nonfinite gradients: optimizer update aborted")
                    if self.clip is not None:self.clip(self.master_grads,m[0])
                    return m
                diagnostic()  # compile on zero scratch; real first gradient remains intact
                stream=torch.cuda.current_stream()
                capture=torch.cuda.Stream()
                capture.wait_stream(stream)
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g,stream=capture):
                    self._bound_metrics=diagnostic()
                    self.optimizer.opt_adamw.inner.step()
                    self.muon_step()
                    self.copy_tensors(self.copy_dst,self.copy_src)
                stream.wait_stream(capture)
                # Transfer ownership exactly once, AFTER warmup and before the
                # first real replay. Subsequent microbatches write here directly.
                self.copy_tensors(self.master_grads,grads)
                acc.buffers=self.master_grads
                self.graph=g
                self.input_grads=()
            self._bound_loss.copy_(loss.detach())
            self.graph.replay()
            self.update_index+=1
            self.update.fill_(self.update_index)
            # Keep the original caller-owned, nonaliasing metrics semantics.
            return self._bound_metrics.clone()
    return BoundUpdate


def install(trainer):
    # Explicit pairing at startup, never a lookup or data-pointer check in
    # the training loop. The detector creates these consecutively in main().
    cls=bound_class(trainer.GraphedOptimizerUpdate)
    original_acc=trainer.GradientAccumulator
    pending=[]
    def update(*args,**kwargs):
        if pending:raise RuntimeError("Second optimizer before binding the first")
        obj=cls(*args,**kwargs);pending.append(obj)
        return obj
    def accumulator(*args,**kwargs):
        obj=original_acc(*args,**kwargs)
        bind(pending.pop(),obj)
        # Read host-side tensor metadata twice on both sides of the
        # accumulation boundary, after setup and before update 11. Restore
        # the original method before the throughput measurement window.
        original=obj.accumulate
        calls=0
        def observe(loss,*,first):
            nonlocal calls
            calls+=1
            if calls in (13,14,17,18):
                info=gradient_storage([p.grad for p in obj.parameters])
                print("ACCUMULATION_STORAGE",json.dumps(dict(micro=calls,first=first,**info)),flush=True)
            if calls==18:del obj.accumulate
            return original(loss,first=first)
        obj.accumulate=observe
        return obj
    trainer.GraphedOptimizerUpdate=update
    trainer.GradientAccumulator=accumulator
