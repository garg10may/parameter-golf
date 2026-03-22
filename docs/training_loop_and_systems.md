# Training Loop and Systems Choices

This document explains:

- why the training loop is shaped by challenge rules
- how gradient accumulation is used
- why warmup gets undone
- why wallclock-aware schedules matter
- how DDP and Flash Attention fit in

## 1. This Is Not a Normal "Train Until Done" Script

The challenge has a wallclock rule.

So the training loop is designed around:

- fixed iteration budgets
- wallclock caps
- multi-GPU scaling
- reproducible timing

That is why the code looks more engineered than a tutorial loop.

## 2. The Effective Batch Trick

The script sets:

```text
grad_accum_steps = 8 // world_size
```

and requires `world_size` to divide `8`.

This means:

- 8 GPUs -> no accumulation
- 4 GPUs -> accumulate 2 microsteps
- 1 GPU -> accumulate 8 microsteps

The goal is to preserve the same **effective global batch** as the 8-GPU baseline.

That makes cross-machine comparisons more meaningful.

## 3. Why Scale the Loss During Accumulation?

When you do gradient accumulation, you do not want gradients to become `N` times too large.

So the script multiplies loss by:

```text
1 / grad_accum_steps
```

before backprop.

That way accumulated gradients match the average gradient over the large batch.

## 4. Why DDP Sync Only on the Last Microstep?

In distributed training, syncing every microstep would be wasteful.

So the script sets:

```text
model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
```

That means:

- local accumulation happens first
- cross-rank gradient sync happens only when needed

This saves communication overhead.

## 5. Why `torch.compile` Is Used

The script compiles:

- the Muon helper
- the model itself

Why?

- better runtime efficiency
- lower overhead after the compile path is warmed up

In a wallclock-limited challenge, runtime engineering matters.

## 6. Why There Is a Warmup That Gets Reset

This is one of the coolest systems tricks in the file.

Warmup steps:

- run real forward/backward/optimizer code
- trigger compilation and kernel setup
- then restore the original model and optimizer state

So the purpose is not "early learning."

The purpose is:

- pay one-time runtime costs before the timed run really begins

That keeps the measured training closer to steady-state performance.

## 7. Wallclock-Aware Warmdown

The LR schedule is not only iteration-based.

If a wallclock cap exists, the script estimates:

- average step time
- remaining wallclock budget

and then scales the LR down as training approaches the time limit.

This is clever because in this challenge:

- the stop condition is not just step count
- it may be "we are out of allowed minutes"

So the LR schedule needs to respect time, not just step number.

## 8. Why Early Stop Is Synchronized Across Ranks

In DDP, all ranks need to agree on when training stops.

If one rank hits the wallclock cap and others keep going, training breaks.

So the script uses an `all_reduce` on a stop flag.

That way:

- if any rank reaches the cap
- all ranks stop together

## 9. Why Validation Is Expensive but Necessary

Validation runs on the full fixed validation split.

That is not cheap.

But it matters because:

- the challenge score is based on validation
- you need to know how the model is actually doing on the real benchmark

The script supports periodic validation through `VAL_LOSS_EVERY`, and always does final validation.

## 10. Why Flash Attention Backend Is Enabled

The script explicitly enables the flash scaled-dot-product attention backend and disables slower alternatives.

Why?

- it is faster on supported CUDA hardware
- it lowers attention overhead

That is a systems optimization, not an architecture change.

## 11. Why CUDA/BF16 Autocast Is Used

The script runs forward passes under:

```text
torch.autocast(device_type="cuda", dtype=torch.bfloat16)
```

That gives:

- lower memory use
- higher throughput
- generally good numerical behavior on modern GPUs

Again, perfect for a speed-limited challenge.

## 12. Why Logging Includes Timing

The logs print:

- step number
- train or validation loss
- `val_bpb`
- training time in milliseconds
- average step time

That is not just nice-to-have.

It is essential because:

- timing is part of the challenge
- the wallclock cap affects real behavior

## 13. How To Reason About Training-Loop Changes

If you change the training loop, ask:

- is the effective batch still comparable?
- did compile overhead move into the timed region?
- did communication cost rise?
- did training become less fair or less reproducible?

In other words, do not think only like a model researcher here.

Also think like a benchmark engineer.

## 14. Quick Summary

- the training loop is shaped by the 8xH100 / wallclock rules
- gradient accumulation keeps effective batch consistent across GPU counts
- warmup exists to pay compile/setup costs and is then undone
- wallclock-aware LR warmdown and synchronized early stopping are challenge-specific systems choices
