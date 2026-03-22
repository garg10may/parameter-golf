# Challenge Metric: `val_bpb`

This document explains:

- why this challenge cannot just compare token loss
- what `val_bpb` means
- how the script converts from token loss to bits-per-byte
- why tokenizer details matter

## 1. The Basic Problem

If everyone used the exact same tokenizer, then comparing:

- validation loss per token

would be straightforward.

But this challenge allows different tokenizers.

That changes everything.

Why?

Because "one token" is not the same amount of text across tokenizers.

## 2. Why Token Loss Alone Can Be Misleading

Suppose tokenizer A uses:

- many short tokens

and tokenizer B uses:

- fewer longer tokens

Then even if two models are equally good at compressing the actual text, their:

- loss per token

can differ just because the token units are different.

So raw token loss is not a fair cross-tokenizer metric.

## 3. The Challenge Solution

This challenge evaluates compression in:

```text
bits per byte
```

That is:

- how many bits the model needs, on average, to encode one byte of text

Lower is better.

## 4. Where `val_loss` Still Fits In

The model still computes normal next-token cross-entropy:

```text
val_loss
```

in **nats per token**.

So the script first gets the normal token loss.

Then it converts:

- nats per token

into:

- bits per token

using:

```text
bits_per_token = val_loss / ln(2)
```

because:

```text
1 nat = 1 / ln(2) bits
```

## 5. From Bits Per Token to Bits Per Byte

Now we also need:

```text
tokens_per_byte
```

Then:

```text
val_bpb = bits_per_token * tokens_per_byte
```

This is exactly the logic in the script.

## 6. Small Example

Suppose:

- `val_loss = 2.0794` nats per token

Then:

```text
bits_per_token = 2.0794 / ln(2) ≈ 3.0
```

Now suppose the tokenizer averages:

- `0.5` tokens per byte

Then:

```text
val_bpb = 3.0 * 0.5 = 1.5 bits per byte
```

That means the model needs about 1.5 bits to encode one byte of validation text on average.

## 7. Why SentencePiece Byte Counting Is Tricky

The tokenizer here is SentencePiece.

SentencePiece pieces are not just ordinary raw string chunks.

They may:

- include a leading-space marker like `▁`
- represent raw bytes
- represent control or unknown tokens

So the script cannot just say:

```text
bytes = len(piece)
```

and be done.

It builds lookup tables for:

- base byte count of each token
- whether a token has a leading-space marker
- whether a token is a boundary/control token

That is why the validation code has more machinery than a standard training script.

## 8. Why Leading Spaces Matter

In SentencePiece, a token like:

```text
▁hello
```

means something like:

```text
" hello"
```

So the true byte count may include a space depending on context.

The script keeps track of this carefully because:

- challenge scores depend on it
- a bug here could make one tokenizer look unfairly better

## 9. Why This Metric Is a Big Deal in This Repo

The challenge README makes it clear that the leaderboard is about compression-oriented evaluation, not just ordinary LM loss.

So `val_bpb` is not a side metric.

It is the real target metric.

That is why:

- the tokenizer gets validated
- byte LUTs are built
- the final quantized artifact is evaluated with `val_bpb`

## 10. Why the Final Quantized Model Is Re-Evaluated

Imagine:

- training weights give great `val_bpb`
- but quantization damages them

The final submission artifact is the compressed model, not the training checkpoint in memory.

So the script must answer:

- how good is the model **after** the final compression path?

That is why the last log line uses the quantized roundtrip model.

## 11. How To Reason About Tokenizer Changes

If you change the tokenizer, you must think about:

- vocabulary size
- average bytes per token
- embedding parameter count
- whether the `val_bpb` byte accounting is still correct

A tokenizer that lowers token loss but inflates tokens-per-byte can still lose on `val_bpb`.

So the right way to think is:

> The unit of competition is text compression, not token prediction in isolation.

## 12. What This Means for Model Design

Choices that affect tokenizer efficiency and parameter size are linked:

- larger vocab may reduce tokens-per-byte
- but larger vocab grows embeddings
- tied embeddings help control that cost
- post-training quantization changes the final artifact size

So `val_bpb` is connected to almost every major design choice in the repo.

## 13. Quick Summary

- `val_loss` is nats per token
- convert to bits per token using `ln(2)`
- multiply by `tokens / bytes`
- result is `val_bpb`
- this is the challenge's tokenizer-agnostic compression metric

## 14. Practical Takeaway

When you read results in this repo, always ask:

- "Did token loss improve?"
- "Did `val_bpb` improve?"
- "Did the final compressed roundtrip model also improve?"

That is the right Parameter Golf mindset.
