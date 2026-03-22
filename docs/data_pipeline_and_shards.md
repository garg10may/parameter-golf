# Data Pipeline and Shards

This document explains:

- where the training data comes from
- what the shard files contain
- how the training loader works
- what the two `data/` scripts are for

## 1. The High-Level Data Story

This repo does not expect you to build the dataset from scratch for normal baseline use.

The normal path is:

1. download published dataset shards and tokenizer
2. point `train_gpt.py` at them
3. train

The optional advanced path is:

1. download the published document list
2. rebuild or change the tokenizer
3. re-export shards locally

That split is important.

## 2. The Easy Path: Cached Published Data

[`data/cached_challenge_fineweb.py`](../data/cached_challenge_fineweb.py) downloads:

- train shard files
- validation shard files
- tokenizer artifacts

from the published Hugging Face dataset repo.

This is the easiest way to start.

## 3. The Advanced Path: Retokenize From Published Docs

[`data/download_hf_docs_and_tokenize.py`](../data/download_hf_docs_and_tokenize.py) is for:

- downloading the selected document list
- rebuilding tokenizers
- re-tokenizing the docs into shard files

Use this if you want to experiment with tokenizer design.

## 4. What Is in a Shard?

Each shard file is a binary file containing:

- a fixed header
- token IDs

The header includes:

- a magic number
- a version
- token count

Then the tokens are stored as:

```text
uint16
```

Why `uint16`?

- vocab sizes here fit in 16 bits
- it saves disk space

## 5. Why the Training Loader Is So Simple

The loader design is intentionally very plain:

- read shards in sorted order
- stream through them sequentially
- wrap around forever

That gives:

- determinism
- very little Python overhead
- easy reasoning about what each batch means

This is not a worker-heavy random-sampling dataloader.

## 6. What `TokenStream` Does

`TokenStream` is the base utility.

It:

- holds the current shard
- keeps a position inside it
- advances to the next shard when needed
- wraps around at the end

Then `take(n)` returns the next `n` tokens from the global stream.

## 7. What `DistributedTokenLoader` Does

This is the DDP-aware layer on top.

For each batch:

1. compute how many tokens each rank should get
2. take one large contiguous chunk from the shared token stream
3. slice out this rank's span
4. build `x` and `y` by shifting by one token

So rank 0, rank 1, and so on each get disjoint spans from the same global stream.

That is simple and clean.

## 8. Why the Loader Takes `+1` Extra Token

To build:

```text
x = tokens[:-1]
y = tokens[1:]
```

you need one extra token at the end.

That is why the loader takes:

```text
local_tokens + 1
```

for each rank span.

## 9. Why Validation Is Loaded Differently

Validation wants:

- a fixed deterministic benchmark

So the script loads the validation token stream into memory, trims it to a clean multiple of sequence length, and then slices it across ranks deterministically.

Training wants streaming simplicity.

Validation wants stable benchmark coverage.

## 10. Why Data and Tokenizer Are So Tied Together

In this challenge, tokenizer choice affects:

- number of tokens
- embedding size
- `val_bpb`
- final parameter budget

So the data pipeline is not just "feed text into model."

It is tightly connected to model design.

## 11. How To Reason About Data Changes

If you change tokenizer or dataset export, ask:

- is the shard format still compatible?
- is `VOCAB_SIZE` still correct?
- is `val_bpb` byte counting still correct?
- are you still using the intended training/validation split?

That matters a lot for fair comparison.

## 12. Quick Summary

- `cached_challenge_fineweb.py` is the normal downloader path
- `download_hf_docs_and_tokenize.py` is the advanced rebuild path
- training shards are sequential `uint16` token streams with a small header
- the training loader streams contiguous spans and slices them per rank
