# Normalization and RMSNorm

This document explains:

- what normalization is doing in a transformer
- LayerNorm vs RMSNorm
- why modern decoder-only models often use RMSNorm
- what this repo is doing

## 1. Why Normalize at All?

During training, hidden activations can drift in scale.

That creates problems:

- optimization becomes harder
- gradients can become unstable
- one layer can hand the next layer inputs with awkward scale

Normalization layers try to keep the network numerically better behaved.

In transformers, normalization is one of the main tools for stable deep training.

## 2. The Classic Mental Model: LayerNorm

LayerNorm normalizes each token's hidden vector by:

1. subtracting the mean
2. dividing by the standard deviation
3. optionally applying learned scale and shift

For a vector `x`:

```text
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
```

In many beginner explanations of GPT-2, LayerNorm is the default norm to learn first.

## 3. What RMSNorm Changes

RMSNorm stands for **Root Mean Square Normalization**.

Instead of centering by subtracting the mean, it only rescales by the RMS magnitude:

```text
RMS(x) = sqrt(mean(x^2))
RMSNorm(x) = x / (RMS(x) + eps)
```

Some variants also include a learned gain parameter. In this repo, `F.rms_norm(...)` is used directly, without an explicit learned affine weight in the custom `RMSNorm` wrapper.

So the key difference is:

- LayerNorm uses mean and variance
- RMSNorm uses only root-mean-square magnitude

## 4. Why That Can Help

RMSNorm is attractive because it is:

- simpler
- a bit cheaper
- often good enough or very good in modern LLMs

The intuition is:

- maybe you do not need to re-center the activation around zero every time
- maybe controlling the scale is the main thing you need

That is not always universally true, but it often works well.

## 5. Small Comparison

Suppose:

```text
x = [2, 4]
```

### LayerNorm-style logic

- mean = `3`
- centered vector = `[-1, 1]`
- variance-based scaling then follows

### RMSNorm-style logic

- do not subtract the mean
- RMS = `sqrt((2^2 + 4^2)/2) = sqrt(10)`
- output is roughly:

```text
[2/sqrt(10), 4/sqrt(10)]
```

So RMSNorm preserves the direction of the original vector more directly.

## 6. What This Repo Does

This repo uses RMSNorm in several places:

- after token embeddings at the start of `GPT.forward`
- before attention inside each block
- before the MLP inside each block
- after the full stack as `final_norm`
- also directly on `q` and `k` inside attention

So normalization is used heavily here, but it is RMS-style rather than LayerNorm-style.

## 7. Why This Repo Likely Chose RMSNorm

For a compact, modern decoder-only baseline, RMSNorm makes sense because:

- it keeps the implementation small
- it is common in recent LLM practice
- it pairs well with pre-norm transformer designs
- it avoids some extra computation relative to LayerNorm

The root script is trying to stay readable while still reflecting modern design choices.

## 8. What Is "Pre-Norm" vs "Post-Norm"?

This is an important related concept.

### Post-norm

The older pattern is roughly:

```text
x = x + Sublayer(x)
x = Norm(x)
```

### Pre-norm

A now very common pattern is:

```text
x = x + Sublayer(Norm(x))
```

This repo is using a **pre-norm** style:

- normalize before attention
- normalize before MLP

That is one reason the model can train more stably even when it uses aggressive engineering elsewhere.

## 9. LayerNorm vs RMSNorm in Practice

### LayerNorm strengths

- very standard
- easy to explain
- historically common in many transformer families

### RMSNorm strengths

- slightly simpler
- often cheaper
- widely adopted in decoder-only LLMs

### LayerNorm downside

- it does a bit more work
- the mean-centering step may be unnecessary in some LLM settings

### RMSNorm downside

- if you learned transformers through LayerNorm, it feels less intuitive at first
- some model families still prefer other normalization setups

## 10. What Other Normalization Ideas Exist?

### 10.1 LayerNorm

Still very common, especially in older or non-LLM transformer codebases.

### 10.2 RMSNorm

Very common in modern decoder-only LLMs.

### 10.3 ScaleNorm

Another scale-focused normalization idea. Less mainstream in production than LayerNorm or RMSNorm.

### 10.4 NoNorm / tiny-norm experiments

Some research explores reducing or removing normalization entirely, but that is not the safe baseline choice for most projects.

### 10.5 Residual scaling methods

Some papers change residual branch scaling rather than the normalization rule itself. This is related but not the same thing.

## 11. What Is Common Today?

A useful practical summary is:

- old GPT-style tutorials often teach LayerNorm first
- many modern open decoder-only LLMs use RMSNorm
- newer architectures often combine RMSNorm with RoPE and GQA

That same cluster of choices appears in official modern model architecture docs. For example, Gemma 3 architecture notes describe a design using RMSNorm together with RoPE, grouped-query attention, and a modern MLP.

## 12. Code Sketch

LayerNorm-style pseudocode:

```python
mu = x.mean(dim=-1, keepdim=True)
var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
out = (x - mu) / torch.sqrt(var + eps)
```

RMSNorm-style pseudocode:

```python
rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
out = x / rms
```

This repo uses PyTorch's built-in `F.rms_norm(...)` helper rather than open-coding the math.

## 13. Why You See No Learned Norm Weights in the Custom Class

The `RMSNorm` class in this repo is very small:

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
```

That is intentionally minimal.

This script is balancing:

- clarity
- compactness
- challenge byte budget

So it keeps some design choices very lean.

## 14. How To Reason About Changes Here

If you want to replace RMSNorm with LayerNorm, ask:

- does training get more stable or less stable?
- do params increase?
- does inference cost change?
- does the model still fit the artifact budget comfortably?

In a challenge like this, even "small" architecture changes can matter because the margin is tight.

## 15. Quick Summary

- LayerNorm normalizes by mean and variance
- RMSNorm normalizes by magnitude only
- this repo uses RMSNorm because it is simple, modern, and a good fit for compact decoder-only LLMs

## 16. Optional References

- Ba et al., *Layer Normalization*
- Zhang and Sennrich, *Root Mean Square Layer Normalization*
- Gemma 3 architecture notes in NVIDIA Megatron Bridge docs
