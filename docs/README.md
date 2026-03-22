# Parameter Golf Learning Handbook

This folder is meant to take you from:

- "I know a basic GPT-2 style transformer"

to:

- "I can read `train_gpt.py`, understand why it looks different, and reason about changes I might make in this repo."

You should be able to work through these notes without needing to constantly jump to external sources.

## Best Reading Order

1. [Overview: `train_gpt.py` Explained](./train_gpt_explained.md)
   Start here for the whole-pipeline picture.
2. [Embeddings and RoPE](./embeddings_and_rope.md)
   Learn how positional information used to be handled and what RoPE changes.
3. [Normalization and RMSNorm](./normalization_and_rmsnorm.md)
   Understand LayerNorm vs RMSNorm and why modern LLMs often prefer RMSNorm.
4. [Attention, MHA, MQA, and GQA](./attention_and_gqa.md)
   Understand grouped-query attention and why this repo uses fewer KV heads.
5. [MLP, Activations, and Width](./mlp_activation_and_width.md)
   Learn why the FFN here is smaller than the GPT-2 one and why it uses `ReLU^2`.
6. [Residual Paths and Skip Connections](./residual_paths_and_skips.md)
   This explains the most unusual architecture choices in this baseline.
7. [Optimizers, Adam, and Muon](./optimizers_and_muon.md)
   Understand the optimizer split and what Muon is doing.
8. [Challenge Metric: `val_bpb`](./challenge_metric_val_bpb.md)
   Learn why this repo cannot just report validation loss and call it a day.
9. [Quantization and Artifact Size](./quantization_and_artifact.md)
   Understand why export logic is part of the modeling story in this challenge.
10. [Training Loop and Systems Choices](./training_loop_and_systems.md)
    Learn why the training loop is shaped by wallclock rules, compilation, and DDP.
11. [Data Pipeline and Shards](./data_pipeline_and_shards.md)
    Understand how data gets into the training script.
12. [How To Contribute Ideas in This Repo](./contributing_in_parameter_golf.md)
    A practical guide to reasoning about changes once the basics feel clear.

## Fast Topic Map

If you want to jump directly to one question:

- "What replaced positional embeddings?" -> [Embeddings and RoPE](./embeddings_and_rope.md)
- "Why RMSNorm instead of LayerNorm?" -> [Normalization and RMSNorm](./normalization_and_rmsnorm.md)
- "Why 8 heads but only 4 KV heads?" -> [Attention, MHA, MQA, and GQA](./attention_and_gqa.md)
- "Why is the FFN only 2x wide?" -> [MLP, Activations, and Width](./mlp_activation_and_width.md)
- "What is this `x0`, `resid_mix`, and `skip_weights` business?" -> [Residual Paths and Skip Connections](./residual_paths_and_skips.md)
- "What is Muon and why not Adam everywhere?" -> [Optimizers, Adam, and Muon](./optimizers_and_muon.md)
- "What even is `val_bpb`?" -> [Challenge Metric: `val_bpb`](./challenge_metric_val_bpb.md)
- "Why does the script quantize and then evaluate again?" -> [Quantization and Artifact Size](./quantization_and_artifact.md)
- "Why is the training loop so engineered?" -> [Training Loop and Systems Choices](./training_loop_and_systems.md)
- "Where do the `.bin` shards come from?" -> [Data Pipeline and Shards](./data_pipeline_and_shards.md)

## What This Handbook Assumes You Already Know

- what a token is
- the basic idea of token embeddings
- the basic idea of self-attention
- residual connections
- feed-forward layers
- next-token prediction

If you know those basics, this handbook will focus on:

- what changed from a beginner GPT-2 mental model
- why it changed
- what tradeoffs come with each choice
- how those choices affect this specific repo

## Current Code Center

The main script is still [`train_gpt.py`](../train_gpt.py).

These docs are not replacing the code. They are a study map for the code.
