# MLP, Activations, and Width

This document explains:

- what the transformer MLP is doing
- why GPT-2-style MLPs are often wide
- why this repo uses a smaller MLP
- why it uses `ReLU^2`
- what other activation and FFN choices are common today

## 1. What the MLP Is For

In each transformer block, attention mixes information **across tokens**.

The MLP then processes each token's hidden state **independently** and adds more nonlinear compute capacity.

So a block usually has two major sublayers:

- attention
- MLP / feed-forward network

You can think of attention as "communication across positions" and the MLP as "per-position feature transformation."

## 2. Classic GPT-2 Style MLP

A typical GPT-2-like MLP looks like:

```text
d_model -> 4 * d_model -> d_model
```

with an activation like GELU in the middle:

```text
MLP(x) = W2(GELU(W1 x))
```

Why expand to `4 * d_model`?

Because the hidden layer is where the network creates richer intermediate features.

More width means:

- more expressivity
- more parameters
- more compute

## 3. What This Repo Does

This repo uses:

```text
d_model -> mlp_mult * d_model -> d_model
```

with:

```text
mlp_mult = 2
```

by default.

And the activation is:

```text
ReLU(x)^2
```

The code is effectively:

```python
x = torch.relu(self.fc(x))
return self.proj(x.square())
```

So the MLP is:

- narrower than the GPT-2 default
- simpler than many modern gated MLPs
- intentionally cheap

## 4. Why Narrow the MLP?

This challenge is not asking:

- "what architecture gets the best loss if parameter count is large?"

It is asking something closer to:

- "what gets strong performance under brutal size and time limits?"

The MLP is one of the biggest consumers of parameters.

So reducing width from `4x` to `2x` can save a lot.

That trade is especially reasonable when:

- the model is small
- embeddings and attention already consume meaningful budget
- final compressed size matters

## 5. What Does `ReLU^2` Mean?

Take a normal ReLU:

```text
ReLU(z) = max(0, z)
```

Then square it:

```text
ReLU^2(z) = max(0, z)^2
```

So:

- negative values still become `0`
- positive values become amplified quadratically

Small examples:

```text
z = -3 -> 0
z = -1 -> 0
z =  1 -> 1
z =  2 -> 4
z =  3 -> 9
```

So it is a sharper nonlinearity than plain ReLU.

## 6. Why Might `ReLU^2` Be Useful?

Possible reasons:

- very simple to implement
- cheap
- stronger activation on positive values
- works surprisingly well in some transformer baselines

In this repo, it is explicitly described as coming from the `modded-nanogpt` setup.

So part of the answer is simply:

- it is a known good baseline trick in this training lineage

## 7. Why Not GELU?

GELU is a very common default in older transformer codebases and GPT-like tutorials.

Reasons a repo might prefer `ReLU^2` instead:

- less complexity
- cheaper
- easier to keep the implementation compact

This repo is especially sensitive to simplicity because the baseline is also meant to be readable.

## 8. Why Not SwiGLU or GeGLU?

Today many strong models use **gated MLPs** such as:

- SwiGLU
- GeGLU

These often outperform plain GELU/ReLU MLPs, but they also:

- add more moving parts
- often change the width formulas
- can increase parameter and implementation complexity

Official Gemma 3 architecture notes, for example, list **GeGLU** together with RMSNorm, RoPE, and GQA.

So why does this repo not do that?

Because this baseline is aiming for:

- a simple starting point
- compact code
- strong enough performance under tight artifact limits

It is not trying to be the fanciest possible SOTA architecture.

## 9. How MLP Width Affects Parameters

Suppose:

- `d_model = 512`

### GPT-2 style 4x MLP

Hidden size:

```text
2048
```

Main weight shapes:

```text
512 x 2048
2048 x 512
```

### This repo's 2x MLP

Hidden size:

```text
1024
```

Main weight shapes:

```text
512 x 1024
1024 x 512
```

That is roughly half the inner-dimension weight cost.

This is a big reason the repo can stay under the size limit.

## 10. What Other FFN Choices Exist?

### 10.1 GELU FFN

Classic modern transformer default.

### 10.2 ReLU FFN

Simple and cheap, but often less favored than GELU or gated variants.

### 10.3 `ReLU^2`

Simple but stronger than plain ReLU.

### 10.4 SwiGLU / GeGLU / other GLU variants

Very popular in stronger recent models.

### 10.5 Mixture-of-Experts MLPs

Instead of one FFN per layer, some models route tokens through selected experts.

This can greatly increase capacity, but it is a different design regime from this baseline.

## 11. What Is Common Today?

A useful practical picture:

- GPT-2 tutorials often teach GELU with a 4x FFN
- many recent strong models use gated MLPs like SwiGLU or GeGLU
- this repo uses a smaller, simpler `ReLU^2` MLP because the challenge rewards compactness and speed

So the MLP here is not "the modern universal default." It is a deliberate budget-conscious baseline choice.

## 12. How To Reason About Changing This Part

If you change the MLP, ask:

- does performance improve enough to justify extra bytes?
- do the new parameters quantize well?
- does the code stay readable and compact?
- does training remain stable?

Possible experiments:

- switch `MLP_MULT` from `2` to `3` or `4`
- try GELU
- try SwiGLU or GeGLU
- try narrower or wider blocks with fewer layers

## 13. Quick Summary

- the MLP is the per-token nonlinear compute part of each block
- GPT-2 often uses `4x` width with GELU
- this repo uses `2x` width with `ReLU^2`
- why: save parameters, keep code simple, stay competitive under the challenge budget

## 14. Optional References

- Brown et al., *Language Models are Few-Shot Learners* for the classic GPT-family FFN picture
- Shazeer, *GLU Variants Improve Transformer*
- Gemma 3 architecture notes showing a modern GeGLU-based design
