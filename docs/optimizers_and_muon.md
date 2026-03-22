# Optimizers, Adam, and Muon

This document explains:

- what an optimizer is doing
- why Adam is the familiar default
- what Muon is trying to do differently
- why this repo uses different optimizers for different parameter groups

## 1. What an Optimizer Does

Training gives you gradients:

```text
grad = d loss / d param
```

The optimizer turns those gradients into parameter updates.

At a very high level:

```text
param = param - learning_rate * update
```

The whole question is:

- what should `update` be?

Different optimizers answer that differently.

## 2. The Familiar Baseline: SGD

Plain stochastic gradient descent says:

```text
update = grad
```

This is conceptually simple but often not ideal for large transformer training without additional tricks.

## 3. Momentum

Momentum keeps a running average of recent gradients.

Intuition:

- if many steps agree on a direction, keep moving that way
- do not react too violently to every tiny noisy gradient

This often makes optimization smoother and faster.

## 4. Adam / AdamW

Adam is the optimizer most people learn early for transformers.

Intuition:

- keep running statistics of the gradient
- adapt step sizes per parameter

Why people like Adam:

- strong default behavior
- widely used
- forgiving

Why people sometimes move beyond "Adam everywhere":

- not every parameter type behaves the same
- large matrices, embeddings, tiny scalar controls, and output heads may want different treatment

## 5. What This Repo Does

This repo does **not** use one optimizer for all parameters.

It splits parameters into:

- token embeddings -> Adam
- untied output head -> Adam
- matrix-shaped transformer block weights -> Muon
- small vectors/scalars/control tensors -> Adam

That is one of the most important training-engineering choices in the script.

## 6. Why Split Parameters by Type?

Because different parameters do different jobs.

### Embeddings

- often sensitive
- often benefit from their own learning rate

### Large 2D hidden weights

- dominate representational learning
- may benefit from a matrix-aware optimizer

### Tiny control tensors

- can be fragile
- often safer with Adam-like treatment

So the split is not arbitrary. It reflects the role of the parameters.

## 7. What Is Muon?

Muon is an optimizer aimed at **hidden weight matrices**.

The official Muon repository describes it this way:

- use Muon for hidden weights
- use AdamW for embeddings, classifier heads, gains, and biases instead

That is almost exactly the logic this repo follows.

## 8. The Core Muon Idea

Muon is trying to improve how updates behave for matrix-shaped parameters.

In this repo's implementation, the rough flow is:

1. take the gradient for a weight matrix
2. maintain a momentum buffer
3. optionally use Nesterov-style momentum
4. pass the matrix update through a Newton-Schulz based orthogonalization step
5. apply the transformed update

That orthogonalization step is the unusual part.

## 9. What Does "Orthogonalize the Update" Mean Intuitively?

A matrix gradient has structure.

It is not just "a bag of unrelated numbers."

Muon tries to shape the update so it behaves more like a well-conditioned matrix transformation instead of a raw noisy tensor.

You do not need the full linear algebra to get the high-level idea:

> Muon tries to make matrix updates more directionally clean and better scaled than raw gradient descent.

## 10. What Is the Newton-Schulz Step Doing?

The helper function:

```python
zeropower_via_newtonschulz5(...)
```

uses an iterative matrix procedure.

At a beginner level, the safest mental model is:

- it takes a matrix-shaped gradient
- normalizes and transforms it
- nudges it toward an orthogonalized update direction

You do not need to derive the polynomial coefficients to understand the repo structure.

## 11. Why Not Use Muon for Everything?

Because Muon is not intended for every parameter type.

The official Muon guidance explicitly says to keep embeddings, classifier heads, and gains/biases on AdamW-style optimization rather than Muon.

That is exactly why this repo keeps:

- embeddings on Adam
- small control tensors on Adam
- only big 2D block matrices on Muon

## 12. Why This Split Makes Sense in This Repo

Remember what the model contains:

- token embedding matrix
- attention projection matrices
- MLP matrices
- tiny learned scales like `attn_scale`, `mlp_scale`, `q_gain`
- skip routing weights

The matrix-heavy hidden transformations are where Muon is most plausible.

The small control parameters are not the place you want an aggressive matrix-focused optimizer.

## 13. Pseudocode Comparison

### Adam-style mental model

```python
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad * grad
update = m / (sqrt(v) + eps)
param -= lr * update
```

### Muon-style mental model in this repo

```python
buf = momentum * buf + grad
g = grad + momentum * buf   # Nesterov-like when enabled
g = orthogonalize_matrix_update(g)
param -= lr * g
```

That is not exact full math, but it captures the role split well.

## 14. What Is the Momentum Warmup Doing?

This repo does something subtle:

- Muon momentum starts lower
- then warms up toward the target momentum value

Why?

- high momentum immediately at the start can sometimes be too aggressive
- warming it up makes early training less jumpy

So Muon itself gets a mini schedule, not just a fixed value.

## 15. Is Muon the Mainstream Industry Default?

No.

The safe picture today is:

- AdamW-family optimizers are still the mainstream default for transformer training
- Muon is a newer, more specialized optimizer that has attracted attention in fast-training and small-model experimentation

So Muon is better thought of as:

- a promising specialized tool
- not the universal default

That is why it is good to understand, but also why you should not assume every model repo should use it.

## 16. Why Muon Fits This Challenge

This challenge encourages unusual but effective design choices under tight constraints.

Muon fits that spirit because:

- it may improve how large hidden matrices learn
- the implementation can stay relatively short
- it aligns with the modded-nanogpt ecosystem this repo borrows from

## 17. What Could Go Wrong With Muon?

Possible downsides:

- less standard than AdamW
- harder to explain
- can be easier to misuse if you apply it to the wrong parameter groups
- may interact differently with model scale and hyperparameters than familiar optimizers

So if you experiment here, you need to keep a close eye on:

- stability
- speed
- final validation score

## 18. How To Reason About Optimizer Changes in This Repo

Good questions:

- should embeddings really have their own LR?
- should output head get a separate optimizer group?
- should Muon be used for all 2D matrices or only some of them?
- should `beta1`, `beta2`, or momentum schedules change?

This repo is a good place to test optimizer splits because the architecture is compact and training cycles are short.

## 19. Quick Summary

- Adam is the familiar all-purpose default
- Muon is a matrix-focused optimizer
- this repo uses Muon only for hidden 2D transformer matrices and keeps embeddings/control tensors on Adam
- that follows the same spirit as the official Muon usage guidance

## 20. Optional References

- Kingma and Ba, *Adam: A Method for Stochastic Optimization*
- Loshchilov and Hutter, *Decoupled Weight Decay Regularization*
- Official Muon repository usage guidance
