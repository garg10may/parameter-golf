# Attention, MHA, MQA, and GQA

This document explains:

- the standard multi-head attention picture
- what grouped-query attention changes
- why this repo uses `num_heads=8` and `num_kv_heads=4`
- what the `q_gain` and Q/K RMSNorm are doing

## 1. Standard Multi-Head Attention First

In the usual picture, for each token representation `x`:

```text
q = x W_q
k = x W_k
v = x W_v
```

Then attention computes:

```text
Attention(q, k, v) = softmax(q k^T / sqrt(d_head)) v
```

With multi-head attention, you do that in several heads in parallel.

If there are `H` heads:

- each head gets its own slice
- each head has its own `q`, `k`, `v`
- outputs are concatenated

## 2. The Standard Cost Problem

Full multi-head attention gives each query head its own key head and value head.

That is expressive, but it costs:

- more parameters
- more memory traffic
- more KV storage

In small models and fast-training settings, that cost matters.

## 3. MHA vs MQA vs GQA

These three are worth comparing carefully.

### 3.1 MHA: Multi-Head Attention

If `num_heads = 8`, then:

- 8 query heads
- 8 key heads
- 8 value heads

This is the classic setup.

### 3.2 MQA: Multi-Query Attention

If `num_heads = 8`, MQA uses:

- 8 query heads
- 1 shared key head
- 1 shared value head

This saves a lot, but can be too aggressive.

### 3.3 GQA: Grouped-Query Attention

This is the middle ground.

Example:

- 8 query heads
- 4 key heads
- 4 value heads

Two query heads share each KV head pair.

That is what this repo does by default.

## 4. Why GQA Is Attractive

GQA tries to get the best of both sides:

- more expressive than MQA
- cheaper than full MHA

That makes it especially appealing in:

- fast inference setups
- long-context setups
- parameter-budget-constrained models

This repo is in the third category.

## 5. What This Repo Does Exactly

The attention module is initialized with:

- `num_heads`
- `num_kv_heads`

and then:

- `c_q` projects to full query width
- `c_k` projects to smaller KV width
- `c_v` projects to smaller KV width

If:

```text
model_dim = 512
num_heads = 8
```

then:

```text
head_dim = 512 / 8 = 64
```

If:

```text
num_kv_heads = 4
```

then KV projection width is:

```text
4 * 64 = 256
```

So compared with full MHA, K and V each cost half as many output channels.

## 6. Shape Walkthrough

Suppose:

- batch size = `B`
- sequence length = `T`
- model dim = `D`
- number of query heads = `H`
- number of KV heads = `H_kv`
- head dim = `d = D / H`

Then:

```text
q: [B, T, D] -> [B, H, T, d]
k: [B, T, H_kv * d] -> [B, H_kv, T, d]
v: [B, T, H_kv * d] -> [B, H_kv, T, d]
```

Then the attention kernel handles the grouping.

In this repo that happens through:

```python
F.scaled_dot_product_attention(..., enable_gqa=True)
```

when the numbers of Q and KV heads differ.

## 7. Why This Matters for Parameter Golf

This challenge rewards small compressed artifacts.

GQA helps because:

- K projection gets smaller
- V projection gets smaller
- activation movement for KV is smaller

That is a very good trade when the model is only width `512` and trying to be competitive under a tiny byte budget.

## 8. RoPE and GQA Can Coexist Naturally

In this repo:

- queries and keys are both RoPE-rotated
- values are not
- GQA just changes how many KV heads exist

These are separate design choices that work fine together.

That combination is also common in modern LLM design.

## 9. Why Q and K Are RMS-Normalized Here

After projection, this repo does:

```python
q = F.rms_norm(q, ...)
k = F.rms_norm(k, ...)
```

before RoPE and attention.

Why do that?

- attention scores come from dot products of `q` and `k`
- if their magnitudes drift too much, attention can become too sharp or too noisy
- normalizing them makes the scale more controlled

That gives the script a more stable attention path.

## 10. What Is `q_gain`?

The repo adds a learned parameter:

```text
q_gain in R^(num_heads)
```

and multiplies each query head by its own gain.

So after normalization and RoPE:

```text
q_h = q_h * q_gain[h]
```

This gives each head some control over how strong or sharp its query signal should be.

It is a lightweight attention control knob.

## 11. Why Bias-Free Projections?

The projections in this attention block use:

- `bias=False` for `c_q`
- `bias=False` for `c_k`
- `bias=False` for `c_v`
- `bias=False` for the output projection

That keeps the module smaller and simpler.

Again, in a challenge with strict artifact limits, small savings add up.

## 12. What Other Attention Variants Exist?

### 12.1 Full MHA

Most standard and easy to reason about.

### 12.2 MQA

Very efficient. Often used when KV efficiency matters a lot.

### 12.3 GQA

Strong compromise. Very popular in modern LLMs.

### 12.4 Sliding-window or local attention

Useful for long context and efficiency.

### 12.5 Hybrid local/global attention

Some newer models mix local and global attention patterns depending on the layer.

Official Gemma 3 architecture docs, for example, describe hybrid attention patterns together with RMSNorm, RoPE, and GQA-style grouping.

## 13. What Is Common Today?

A useful snapshot is:

- many modern open decoder-only LLMs use RoPE + RMSNorm + GQA
- full MHA still exists, especially in simpler or older implementations
- long-context systems often add local/global attention tricks on top

That is why this repo's attention looks "modern" rather than "classic GPT-2."

## 14. Simple Cost Comparison

Suppose `D = 512`, `H = 8`, `d = 64`.

### Full MHA

`W_k` and `W_v` each output:

```text
8 * 64 = 512
```

### GQA with 4 KV heads

`W_k` and `W_v` each output:

```text
4 * 64 = 256
```

So you halve the output width of K and V projections.

That is a meaningful savings.

## 15. Quick Summary

- standard MHA: every query head gets its own KV head
- MQA: many query heads share one KV head
- GQA: middle ground
- this repo uses GQA because it saves cost while keeping strong multi-head behavior

## 16. Optional References

- Vaswani et al., *Attention Is All You Need*
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*
- Gemma 3 architecture notes in NVIDIA Megatron Bridge docs
