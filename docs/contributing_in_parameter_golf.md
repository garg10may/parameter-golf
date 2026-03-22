# How To Contribute Ideas in Parameter Golf

This document is about moving from understanding to contribution.

The goal is:

- after reading the handbook, you should be able to look at a design choice and ask the right questions before changing it

## 1. Think Like the Challenge, Not Like a Generic Repo

A normal ML project often optimizes for:

- best validation loss
- maintainable software
- easy deployment later

Parameter Golf optimizes for a more unusual bundle:

- strong final `val_bpb`
- compressed artifact under the byte cap
- fast enough training for the challenge track
- compact, self-contained code

So every idea should be judged through that lens.

## 2. The Main Axes of Design in This Repo

When you consider a change, ask which axis it touches.

### Architecture

Examples:

- change RoPE
- change head counts
- change MLP width
- change skip routing

### Optimizer

Examples:

- adjust Muon usage
- change learning rates
- change momentum schedule

### Training systems

Examples:

- compile behavior
- accumulation pattern
- validation cadence

### Tokenizer / data

Examples:

- change vocab size
- rebuild tokenizer
- retokenize shards

### Export / compression

Examples:

- new quantization scheme
- different tensor keep-float policy

## 3. The Four Questions You Should Always Ask

Whenever you propose a change, ask:

1. Does it improve the final roundtrip `val_bpb`?
2. Does it keep the artifact within budget?
3. Does it keep training stable enough?
4. Is the code complexity worth the gain?

If you forget even one of those, you can make a change that looks smart locally but loses the actual competition.

## 4. A Good Beginner Workflow

1. Read [`train_gpt.py`](../train_gpt.py) with [the overview](./train_gpt_explained.md) beside it.
2. Pick one subsystem only.
3. Form one hypothesis.
4. Predict how it affects:
   loss, `val_bpb`, bytes, and stability.
5. Implement the smallest test of that idea.
6. Compare against the baseline honestly.

That is much better than changing five things at once.

## 5. Good Early Contribution Ideas

If you are just getting started, these are good first experiments:

- tweak `ROPE_BASE`
- try different `NUM_KV_HEADS`
- try `MLP_MULT=3`
- remove or simplify one residual trick
- adjust `TIED_EMBED_LR`
- adjust which tensors stay float during quantization

These are good because:

- they are local
- you can reason about them
- they teach you how the objective behaves

## 6. Higher-Risk Ideas

These can be strong, but are harder:

- new tokenizer family
- new quantization format
- replacing `ReLU^2` with a gated MLP
- major changes to optimizer grouping
- alternative positional encoding

Do these after you are comfortable with the baseline.

## 7. How To Read Results Correctly

Do not stop at:

- train loss looks better

Also check:

- periodic validation
- final roundtrip `val_bpb`
- compressed model size
- code size impact

In this repo, a "better model" that fails the artifact budget is not really better for the challenge.

## 8. How To Think About Simplicity

Sometimes a clever idea is not worth it because:

- it adds too much code
- it makes debugging harder
- it increases artifact bytes
- it only gives a tiny gain

This repo's baseline is intentionally compact.

So one sign of maturity here is not just inventing tricks.

It is learning which tricks are actually worth carrying.

## 9. What To Learn Next After the Handbook

After you finish the docs in this folder, the best next steps are:

- trace one real forward pass with tensor shapes
- trace one training step with gradient accumulation
- compare the root script with a record run snapshot in `records/`
- make one small change and inspect the logs

That is how understanding turns into useful contribution.

## 10. Final Mindset

The baseline in this repo is not sacred.

It is a launch pad.

Your job as a contributor is to learn:

- which parts are modern defaults
- which parts are challenge-specific hacks
- which parts are promising levers for better tradeoffs

Once you can tell those apart, you are ready to contribute meaningful ideas.
