# Quantization and Artifact Size

This document explains:

- why model export matters in this challenge
- what int8 quantization is doing here
- why some tensors are kept in float
- why the script re-loads the compressed model and evaluates it again

## 1. Why Export Is Part of the Real Task

In many ML projects, training and export are separate concerns.

You might say:

- "first I train the best model"
- "then later I worry about deployment"

That is **not** the mindset of this challenge.

Here, the final artifact size is part of the competition.

So export is part of modeling.

## 2. What Counts in the Artifact

The challenge README explains that the artifact includes:

- code bytes
- compressed model bytes

That means even if your model trains well, it is not a valid strong submission if it blows the artifact budget.

## 3. Why Saving the Raw Checkpoint Is Wasteful

Training happens in:

- bf16
- fp32

But saving all weights in high precision uses lots of bytes.

For a tight `16,000,000` byte cap, that is too expensive.

So the script compresses the model after training.

## 4. What This Repo's Quantization Does

The export logic:

1. walks the state dict
2. finds floating-point tensors
3. quantizes big float tensors to int8
4. saves scales for dequantization
5. keeps some tensors in float instead
6. serializes the result
7. zlib-compresses it

So the final file is:

```text
final_model.int8.ptz
```

## 5. Why Int8?

Int8 is attractive because:

- 8 bits per value is much smaller than fp16 or fp32
- matrix weights often tolerate moderate quantization reasonably well
- it is simple enough for a compact script

This is post-training quantization, not quantization-aware training.

## 6. Per-Row vs Per-Tensor Scaling

The script uses two main schemes.

### 6.1 Per-row scales for 2D tensors

For large matrices, it uses one scale per row.

Why?

- different rows often have different value ranges
- per-row scaling preserves more detail than one single scale for the whole matrix

### 6.2 Per-tensor scale for vectors/scalars

For smaller non-matrix tensors, a single scale is enough.

That keeps metadata smaller.

## 7. Why Some Tensors Stay in Float

The script deliberately does **not** quantize everything the same way.

Small or sensitive tensors may stay in float because:

- the byte savings are small anyway
- quantization noise could hurt disproportionately
- some control tensors are more fragile than large matrices

This includes tensors whose names match control patterns such as:

- `attn_scale`
- `mlp_scale`
- `resid_mix`
- `q_gain`
- `skip_weights`

That is a smart trade:

- large matrices get aggressive compression
- delicate control parameters get more protection

## 8. Why zlib After Quantization?

Int8 alone helps a lot.

zlib helps further because:

- serialized tensors often still contain structure and redundancy
- many small metadata fields compress well

So the full path is:

```text
float tensors -> quantized object -> torch.save bytes -> zlib compress
```

## 9. Why the Script Evaluates the Quantized Roundtrip Model

This is one of the most important practical ideas in the repo.

The script:

1. compresses the model
2. saves the compressed file
3. reads it back
4. dequantizes it
5. loads the dequantized weights into the model
6. evaluates again

Why?

Because the final scored submission is the compressed artifact, not the raw training weights.

So the script wants to measure:

- how much quality survives the export path

That is exactly the right thing to do for this challenge.

## 10. What Is Clipping Doing Before Quantization?

The quantization logic uses a high percentile clip rather than blindly scaling from the single largest outlier.

Why?

Because one extreme outlier can force the scale to be too large and waste int8 precision on the rest of the values.

Percentile clipping says:

- keep almost all values in range
- do not let a tiny number of outliers dominate the scale

That usually gives a better practical quantization trade.

## 11. Why This Is Not Just "Deployment Stuff"

In a normal project, you might treat quantization as late engineering.

Here it changes strategy earlier:

- architectures that quantize well may be preferable
- control tensors may deserve special handling
- tied embeddings and smaller vocab help both model size and export size

So export and architecture are connected.

## 12. What Other Quantization Options Exist?

### 12.1 fp16 / bf16 checkpoint

Easy, but too large for this challenge.

### 12.2 Int8 with scales

What this repo uses.

### 12.3 4-bit / NF4 / other low-bit schemes

Can save more bytes, but complexity goes up and quality may drop more.

### 12.4 Quantization-aware training

Potentially better compression behavior, but a lot more complexity than this baseline wants.

## 13. How To Reason About Changes Here

If you change export logic, ask:

- does the compressed file get smaller?
- how much does roundtrip `val_bpb` degrade?
- is the new format still simple and reproducible?
- does code size grow too much?

In this challenge, code complexity also has a byte cost.

## 14. Quick Summary

- the final artifact size is part of the challenge
- this repo uses int8 plus zlib to shrink the model
- big matrices are quantized aggressively
- small/control tensors are often kept in float
- the script evaluates the roundtrip model because that is the real submission target
