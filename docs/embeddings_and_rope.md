# Embeddings and RoPE

This document explains:

- what token embeddings are
- how positional information was traditionally added
- what RoPE changes
- why this repo uses RoPE instead of learned positional embeddings
- what other positional schemes exist
- what is common in modern models

## 1. Start With the Basic Problem

A transformer sees a sequence of token IDs:

```text
[17, 42, 901, 8, ...]
```

By themselves, these IDs are just integers. The model needs vector representations.

So we use an embedding table:

```text
E in R^(vocab_size x d_model)
```

and each token ID `t_i` is turned into:

```text
x_i = E[t_i]
```

If you stopped there, the model would know:

- which token is present

but not:

- where that token is in the sequence

That is the positional encoding problem.

## 2. Traditional Approach: Absolute Positional Embeddings

In the GPT-2 picture, you usually have:

```text
x_i = token_embedding(t_i) + position_embedding(i)
```

where:

- `token_embedding(t_i)` says what token it is
- `position_embedding(i)` says where it is in the sequence

### 2.1 Learned absolute position embeddings

This is the familiar GPT-2 style:

```text
P in R^(max_seq_len x d_model)
x_i = E[t_i] + P[i]
```

Good parts:

- simple
- easy to understand
- works well

Less good parts:

- it adds another parameter table
- it is tied to a maximum sequence length
- extrapolating beyond the trained length is not naturally elegant

### 2.2 Sinusoidal absolute encodings

An older alternative from the original Transformer was sinusoidal positions.

These use deterministic sine/cosine functions instead of learned position vectors.

Good parts:

- no learned position table
- elegant math

Less good parts:

- in practice, many later decoder-only LLMs moved to RoPE or related schemes

#### 2.2.1 The exact formula

For model dimension `d_model`, position `pos`, and channel index `i`, the original Transformer uses:

```text
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

So:

- even dimensions use sine
- odd dimensions use cosine
- low-index dimensions change quickly with position
- high-index dimensions change slowly with position

That gives you one position vector:

```text
PE(pos) in R^(d_model)
```

and then the input to the transformer becomes:

```text
x_pos = token_embedding(token_at_pos) + PE(pos)
```

This is still an **absolute** positional method because position `17` gets the vector for position `17`, position `18` gets the vector for position `18`, and so on.

#### 2.2.2 Why sine and cosine?

There are two beautiful ideas here.

First:

- the encoding is deterministic
- so there is no learned position table

Second:

- different dimensions oscillate at different frequencies

That means the final position vector is like a bundle of waves:

- some dimensions change fast
- some change slowly

Together, those waves give the model a rich signature for each position.

#### 2.2.3 Why use many frequencies?

If you used only one sine wave, many positions would look too similar because the pattern repeats.

By mixing many frequencies, the model gets:

- fine local detail from fast-changing dimensions
- coarse long-range structure from slow-changing dimensions

That makes positions easier to distinguish.

#### 2.2.4 Tiny example

Suppose `d_model = 4`.

Then you have two sine/cosine frequency pairs:

```text
PE(pos, 0) = sin(pos / 10000^(0/4)) = sin(pos)
PE(pos, 1) = cos(pos / 10000^(0/4)) = cos(pos)
PE(pos, 2) = sin(pos / 10000^(2/4)) = sin(pos / 100)
PE(pos, 3) = cos(pos / 10000^(2/4)) = cos(pos / 100)
```

So for position `pos = 3`:

```text
PE(3) = [sin(3), cos(3), sin(0.03), cos(0.03)]
```

Notice what happened:

- the first pair changes quickly
- the second pair changes slowly

That is the multi-scale idea.

#### 2.2.5 Code sketch

Here is a simple PyTorch-style implementation:

```python
import math
import torch

def sinusoidal_positions(seq_len: int, d_model: int) -> torch.Tensor:
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe
```

Then you would use it like:

```python
x = tok_emb(input_ids) + pos_emb[:seq_len]
```

where `pos_emb` is the fixed sinusoidal matrix.

#### 2.2.6 Why does this help with relative position at all?

Even though sinusoidal encodings are **absolute** encodings, they have a useful mathematical property:

- shifting the position changes the sine/cosine values in a structured way

Because of trigonometric identities like:

```text
sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
```

the model can, in principle, learn functions of **relative offsets** from these absolute vectors.

This is one reason sinusoidal encodings were so elegant:

- they are absolute in how they are assigned
- but they still contain algebraic structure that helps with relative reasoning

#### 2.2.7 Why is it called "absolute" if relative structure is still possible?

Because the encoding assigned to a token depends directly on:

```text
its own absolute position index
```

You first compute `PE(5)`, `PE(6)`, `PE(7)`, and then add those vectors to the tokens at those exact positions.

That is different from methods that explicitly encode:

- distance between two tokens
- position difference inside the attention score

So sinusoidal encodings are still categorized as absolute positional encodings.

#### 2.2.8 Advantages of sinusoidal absolute encodings

- no learned position parameters
- simple closed-form math
- elegant multi-frequency structure
- easy to generate for any sequence length

#### 2.2.9 Limitations compared with later methods

- the position signal is added at the input rather than directly shaping `q` and `k`
- many later decoder-only LLMs found RoPE-style attention-integrated position handling more attractive
- long-context extrapolation often led researchers to explore newer tricks

So sinusoidal encodings are historically important and mathematically elegant, but they are not the dominant decoder-only choice in the kind of model family this repo is imitating.

## 3. What `train_gpt.py` Does Instead

This repo does **not** add a learned position vector at the input.

Instead:

1. it embeds the token IDs
2. it projects to `q`, `k`, `v`
3. it applies **rotary position embeddings** to `q` and `k`

So the position signal enters **inside attention** rather than as an extra vector added to the token embedding.

That is why the code looks like this in spirit:

```python
x = tok_emb(input_ids)
q = c_q(x)
k = c_k(x)
q = apply_rotary_emb(q, cos, sin)
k = apply_rotary_emb(k, cos, sin)
```

## 4. What Is RoPE?

RoPE stands for **Rotary Positional Embedding**.

The core idea is:

- take pairs of dimensions in `q` and `k`
- rotate them by an angle that depends on the token position

So instead of adding a position vector, we **rotate the representation** based on position.

## 5. Intuition Before Math

Imagine each pair of coordinates as a 2D point:

```text
(x1, x2)
```

RoPE rotates that point by an angle `theta(position)`.

Different dimension pairs rotate at different frequencies.

So:

- token at position 10 gets one set of rotations
- token at position 11 gets slightly different rotations
- token at position 200 gets much more rotated in some dimensions

When attention computes `q dot k`, those rotations make the score sensitive to **relative position**.

That is the important conceptual payoff:

> RoPE makes attention naturally position-aware, and the dot products end up encoding relative offsets in a useful way.

## 6. The Math Idea

For one pair of channels, a rotation looks like:

```text
[x1']   [ cos(theta)  -sin(theta) ] [x1]
[x2'] = [ sin(theta)   cos(theta) ] [x2]
```

RoPE applies this across many 2D pairs of the head dimension.

Different pairs use different frequencies:

```text
theta_i(pos) = pos / base^(2i / d)
```

where:

- `pos` is the token position
- `i` indexes the pair
- `d` is the head dimension
- `base` is a hyperparameter like `10000`

In this repo, the rotary base is controlled by:

```text
ROPE_BASE
```

and defaults to `10000.0`.

## 7. Why RoPE Is Attractive

### 7.1 No learned position table

That saves parameters.

In a challenge where code bytes and compressed model bytes matter, every large table deserves scrutiny.

### 7.2 Good fit for attention

RoPE modifies `q` and `k`, which directly control attention scores.

That often works very well in decoder-only LLMs.

### 7.3 Strong practical track record

RoPE-style position encoding is widely used in modern open decoder-only model families. For example, official architecture notes for Gemma 3 describe a design that combines RoPE with RMSNorm, grouped-query attention, and modern MLP choices. That is very similar to the overall design direction this repo borrows from.

## 8. Code Walkthrough in This Repo

The relevant pieces are:

- `Rotary`
- `apply_rotary_emb`
- `CausalSelfAttention`

Conceptually, the flow is:

```python
q = self.c_q(x).reshape(...)
k = self.c_k(x).reshape(...)

q = rms_norm(q)
k = rms_norm(k)

cos, sin = self.rotary(seqlen, x.device, q.dtype)
q = apply_rotary_emb(q, cos, sin)
k = apply_rotary_emb(k, cos, sin)
```

This means:

- only `q` and `k` are rotated
- `v` is not rotated

That is normal for RoPE.

## 9. Small Worked Example

Suppose one head has a tiny 4D vector:

```text
q = [1.0, 0.0, 2.0, 0.0]
```

Think of this as two 2D pairs:

```text
(1.0, 0.0) and (2.0, 0.0)
```

At a certain position, suppose the first pair rotates by 90 degrees:

```text
(1.0, 0.0) -> (0.0, 1.0)
```

and the second pair rotates by a different angle, say 30 degrees:

```text
(2.0, 0.0) -> (2 cos 30 deg, 2 sin 30 deg)
```

Now the final `q` depends on position even though we never added a separate position vector.

That is the spirit of RoPE.

## 10. Why This Repo Probably Chose RoPE

For this baseline, RoPE is a strong choice because it:

- removes a learned position table
- is common in modern decoder-only LLMs
- works well with causal attention
- keeps the implementation short enough for a compact challenge baseline

In this repo, those advantages line up nicely with the challenge goals.

## 11. What Other Positional Methods Exist?

### 11.1 Learned absolute embeddings

Classic GPT-2 style.

Use when:

- simplicity matters
- you do not mind an explicit position table

### 11.2 Sinusoidal encodings

Original Transformer style.

Use when:

- you want fixed deterministic positional features

### 11.3 Relative position bias

Instead of modifying the token representation itself, some methods modify the attention scores based on relative distance.

Common in:

- encoder-decoder and encoder-style transformer variants
- T5-family style designs

### 11.4 ALiBi

ALiBi adds linear attention biases based on token distance.

People like it because:

- it is simple
- it can extrapolate to longer lengths surprisingly well

### 11.5 RoPE variants and scaling tricks

A lot of long-context work today is not "replace RoPE completely" but rather:

- keep RoPE
- change its scaling
- use different bases
- use position interpolation or related long-context tricks

So RoPE is not one frozen idea. It has a whole ecosystem of extensions.

## 12. What Is Common Today?

A useful high-level picture is:

- older GPT-like models often used learned absolute positions
- many modern open decoder-only LLMs use **RoPE**
- some encoder or encoder-decoder families use relative bias schemes instead
- some long-context research explores ALiBi, RoPE scaling, or other extrapolation tricks

For your purposes in this repo, the key thing is:

> RoPE is a very standard modern choice, and it fits the parameter-budget setting better than a learned position table.

## 13. Important Tradeoffs

RoPE is not "strictly better in every way."

It has tradeoffs:

- you need the head dimension to be even
- long-context behavior can depend a lot on the chosen base/scaling
- understanding it is less immediate than understanding `x + pos_emb`

But in practice it is a very sensible choice here.

## 14. If You Want To Modify This Part of the Repo

Possible experiments include:

- changing `ROPE_BASE`
- trying learned absolute embeddings instead
- trying ALiBi
- trying long-context RoPE scaling tricks

If you do that, ask yourself:

- does it change parameter count?
- does it change training stability?
- does it change extrapolation to longer sequence lengths?
- does it help or hurt the final compressed artifact?

## 15. Quick Summary

- Traditional GPT-2: `token_emb + position_emb`
- This repo: token embedding first, then RoPE on `q` and `k`
- Why: fewer parameters, modern standard, good fit for causal attention

## 16. Optional References

- Vaswani et al., *Attention Is All You Need*
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*
- Gemma 3 architecture summary in NVIDIA Megatron Bridge docs
