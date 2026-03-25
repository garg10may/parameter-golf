from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from train_gpt import (
    CONTROL_TENSOR_NAME_PATTERNS,
    INT8_CLIP_Q,
    INT8_KEEP_FLOAT_FP32_NAME_PATTERNS,
    INT8_KEEP_FLOAT_MAX_NUMEL,
    INT8_KEEP_FLOAT_STORE_DTYPE,
    INT8_PER_ROW_SCALE_DTYPE,
    CastedLinear,
    RMSNorm,
    Rotary,
    apply_rotary_emb,
    keep_float_tensor,
    tensor_nbytes,
)


def fake_quantize_per_row_ste(weight: Tensor, bits: int) -> Tensor:
    if weight.ndim != 2:
        raise ValueError(f"QAT only supports 2D weights, got ndim={weight.ndim}")
    max_q = (2 ** (bits - 1)) - 1
    w32 = weight.float()
    clip_abs = w32.abs().amax(dim=1).clamp_min(1.0 / max_q)
    scale = clip_abs / max_q
    q = torch.clamp(torch.round(torch.clamp(w32, -clip_abs[:, None], clip_abs[:, None]) / scale[:, None]), -max_q, max_q)
    dq = (q * scale[:, None]).to(dtype=weight.dtype)
    return weight + (dq - weight).detach()


def quantize_float_tensor_nbits(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    max_q = (2 ** (bits - 1)) - 1
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        ).clamp_min(1.0 / max_q)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = clip_abs / max_q
        q = torch.clamp(torch.round(clipped / scale[:, None]), -max_q, max_q).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / max_q if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -max_q, max_q).to(torch.int8).contiguous()
    return q, scale


class AdvancedCastedLinear(CastedLinear):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = False, qat_enabled: bool = False, qat_bits: int = 8):
        super().__init__(in_features, out_features, bias=bias)
        self.qat_enabled = qat_enabled
        self.qat_bits = qat_bits

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        if self.qat_enabled:
            weight = fake_quantize_per_row_ste(weight, bits=self.qat_bits)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight.to(x.dtype), bias)


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.full((dim,), 3.0, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        prev = torch.cat((x[:, :1, :], x[:, :-1, :]), dim=1)
        gate = torch.sigmoid(self.gate).to(dtype=x.dtype)[None, None, :]
        return gate * x + (1.0 - gate) * prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None

    def bigram_hash(self, token_ids: Tensor) -> Tensor:
        prev = torch.cat((torch.zeros_like(token_ids[:, :1]), token_ids[:, :-1]), dim=1)
        return ((prev * 31 + token_ids) % self.bigram_vocab_size).long()

    def forward(self, token_ids: Tensor) -> Tensor:
        hidden = self.embed(self.bigram_hash(token_ids))
        return self.proj(hidden) if self.proj is not None else hidden


class AdvancedCausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        *,
        qat_enabled: bool,
        qat_bits: int,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = AdvancedCastedLinear(dim, dim, bias=False, qat_enabled=qat_enabled, qat_bits=qat_bits)
        self.c_k = AdvancedCastedLinear(dim, kv_dim, bias=False, qat_enabled=qat_enabled, qat_bits=qat_bits)
        self.c_v = AdvancedCastedLinear(dim, kv_dim, bias=False, qat_enabled=qat_enabled, qat_bits=qat_bits)
        self.proj = AdvancedCastedLinear(dim, dim, bias=False, qat_enabled=qat_enabled, qat_bits=qat_bits)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class AdvancedMLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, *, qat_enabled: bool, qat_bits: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = AdvancedCastedLinear(dim, hidden, bias=False, qat_enabled=qat_enabled, qat_bits=qat_bits)
        self.proj = AdvancedCastedLinear(hidden, dim, bias=False, qat_enabled=qat_enabled, qat_bits=qat_bits)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class AdvancedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        *,
        qat_enabled: bool,
        qat_bits: int,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = AdvancedCausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            qat_enabled=qat_enabled,
            qat_bits=qat_bits,
        )
        self.mlp = AdvancedMLP(dim, mlp_mult, qat_enabled=qat_enabled, qat_bits=qat_bits)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class AdvancedGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        *,
        use_smeargate: bool,
        use_bigram_hash: bool,
        bigram_vocab_size: int,
        bigram_dim: int,
        init_mode: str,
        qat_enabled: bool,
        qat_bits: int,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if use_bigram_hash else None
        self.smeargate = SmearGate(model_dim) if use_smeargate else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                AdvancedBlock(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    qat_enabled=qat_enabled,
                    qat_bits=qat_bits,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights(init_mode)

    def _init_weights(self, init_mode: str) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
                continue
            if init_mode == "orthogonal" and isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)

    def encode(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids).to(dtype=x.dtype)
        if self.smeargate is not None:
            x = self.smeargate(x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.encode(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids).reshape(-1, self.tok_emb.num_embeddings)
        targets = target_ids.reshape(-1)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


def eval_val_sliding(
    args: object,
    model: AdvancedGPT,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    seq_len = int(getattr(args, "train_seq_len"))
    stride = int(getattr(args, "eval_stride"))
    eval_batch_seqs = int(getattr(args, "eval_batch_seqs"))
    total_targets = val_tokens.numel() - 1
    if total_targets <= 0:
        raise ValueError("Validation split is empty")
    if total_targets < seq_len:
        raise ValueError(
            f"Sliding evaluation requires at least TRAIN_SEQ_LEN={seq_len} targets, got {total_targets}"
        )

    max_start = total_targets - seq_len
    start_positions = list(range(0, max_start + 1, stride))
    if not start_positions or start_positions[-1] != max_start:
        start_positions.append(max_start)

    window_specs: list[tuple[int, int, int]] = []
    prev_target_end = 0
    for start in start_positions:
        abs_target_start = max(start + 1, prev_target_end + 1)
        abs_target_end = min(start + seq_len, total_targets)
        if abs_target_start > abs_target_end:
            continue
        local_start = abs_target_start - (start + 1)
        local_end = abs_target_end - (start + 1) + 1
        window_specs.append((start, local_start, local_end))
        prev_target_end = abs_target_end

    spec_start = (len(window_specs) * rank) // world_size
    spec_end = (len(window_specs) * (rank + 1)) // world_size
    local_specs = window_specs[spec_start:spec_end]

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(local_specs), eval_batch_seqs):
            batch_specs = local_specs[batch_start : batch_start + eval_batch_seqs]
            inputs = torch.stack(
                [val_tokens[start : start + seq_len] for start, _, _ in batch_specs],
                dim=0,
            ).to(device=device, dtype=torch.int64, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model.forward_logits(inputs)
            for sample_idx, (start, local_start, local_end) in enumerate(batch_specs):
                sample_logits = logits[sample_idx, local_start:local_end].reshape(-1, logits.size(-1))
                abs_target_start = start + 1 + local_start
                abs_target_end = start + local_end
                sample_targets = val_tokens[abs_target_start : abs_target_end + 1].to(
                    device=device,
                    dtype=torch.int64,
                    non_blocking=True,
                )
                sample_loss = F.cross_entropy(sample_logits.float(), sample_targets, reduction="sum")
                val_loss_sum += sample_loss.to(torch.float64)
                val_token_count += float(sample_targets.numel())

                prev_ids = val_tokens[abs_target_start - 1 : abs_target_end].to(
                    device=device,
                    dtype=torch.int64,
                    non_blocking=True,
                )
                tgt_ids = sample_targets
                token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes += (
                    has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
                ).to(dtype=torch.int16)
                val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def compress_bytes(raw: bytes, codec: str, level: int) -> bytes:
    if codec == "zlib":
        return zlib.compress(raw, level=level or 9)
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError("EXPORT_CODEC=zstd requires the 'zstandard' package") from exc
    compressor = zstd.ZstdCompressor(level=level or 22)
    return compressor.compress(raw)


def decompress_bytes(blob: bytes, codec: str) -> bytes:
    if codec == "zlib":
        return zlib.decompress(blob)
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError("EXPORT_CODEC=zstd requires the 'zstandard' package") from exc
    return zstd.ZstdDecompressor().decompress(blob)


def pick_tensor_bits(name: str, args: object) -> int:
    if name.startswith("blocks.") and ".attn." in name and name.endswith(".weight"):
        return int(getattr(args, "export_attn_weight_bits"))
    if name.startswith("blocks.") and ".mlp." in name and name.endswith(".weight"):
        return int(getattr(args, "export_mlp_weight_bits"))
    return 8


def quantize_state_dict_advanced(state_dict: dict[str, Tensor], args: object) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    block_indices = [
        int(name.split(".")[1])
        for name in state_dict
        if name.startswith("blocks.") and name.count(".") >= 2
    ]
    last_block_idx = max(block_indices) if block_indices else -1

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        keep_fp16 = False
        if name == "tok_emb.weight" and str(getattr(args, "export_tied_embed_mode")) == "fp16":
            keep_fp16 = True
        if (
            name == f"blocks.{last_block_idx}.attn.c_k.weight"
            and str(getattr(args, "export_late_k_mode")) == "fp16"
        ):
            keep_fp16 = True

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or keep_fp16:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            if keep_fp16 and name not in passthrough_orig_dtypes and t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        bits = pick_tensor_bits(name, args)
        q, s = quantize_float_tensor_nbits(t, bits=bits)
        qmeta[name] = {
            "scheme": "per_row" if s.ndim > 0 else "per_tensor",
            "axis": 0 if s.ndim > 0 else None,
            "bits": bits,
        }
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "advanced_mixed_precision_v1",
        "__codec__": str(getattr(args, "export_codec")),
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "qmeta": qmeta,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_advanced(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            scale = s.to(dtype=torch.float32)
            out[name] = (q.float() * scale.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


@dataclass
class SWAState:
    tensors: dict[str, Tensor] | None = None
    updates: int = 0

    def update(self, module: nn.Module) -> None:
        state = module.state_dict()
        if self.tensors is None:
            self.tensors = {
                name: tensor.detach().cpu().clone()
                for name, tensor in state.items()
            }
            self.updates = 1
            return
        assert self.tensors is not None
        self.updates += 1
        for name, tensor in state.items():
            avg = self.tensors[name]
            avg.mul_((self.updates - 1) / self.updates).add_(tensor.detach().cpu(), alpha=1.0 / self.updates)

    def apply(self, module: nn.Module) -> dict[str, Tensor] | None:
        if self.tensors is None:
            return None
        previous = {
            name: tensor.detach().cpu().clone()
            for name, tensor in module.state_dict().items()
        }
        module.load_state_dict(self.tensors, strict=True)
        return previous
