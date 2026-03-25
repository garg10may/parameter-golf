from __future__ import annotations

import os
from typing import Mapping


ADVANCED_ONLY_ENV_VARS: dict[str, str] = {
    "ADAM_WEIGHT_DECAY": "advanced decoupled AdamW weight decay",
    "BIGRAM_DIM": "advanced BigramHash input augmentation",
    "BIGRAM_VOCAB_SIZE": "advanced BigramHash input augmentation",
    "EVAL_BATCH_SEQS": "advanced sliding-window evaluation batching",
    "EVAL_MODE": "advanced evaluation mode selection",
    "EVAL_STRIDE": "advanced sliding-window evaluation stride",
    "EXPORT_ATTN_WEIGHT_BITS": "advanced export quantization policy",
    "EXPORT_CODEC": "advanced export compression codec",
    "EXPORT_CODEC_LEVEL": "advanced export compression level",
    "EXPORT_LATE_K_MODE": "advanced late-layer key export policy",
    "EXPORT_MLP_WEIGHT_BITS": "advanced export quantization policy",
    "EXPORT_TIED_EMBED_MODE": "advanced tied-embedding export policy",
    "INIT_MODE": "advanced initialization policy",
    "MUON_WEIGHT_DECAY": "advanced decoupled Muon weight decay",
    "QAT_BITS": "advanced quantization-aware training bit width",
    "QAT_ENABLED": "advanced quantization-aware training toggle",
    "SWA_ENABLED": "advanced stochastic weight averaging toggle",
    "SWA_EVERY": "advanced stochastic weight averaging cadence",
    "SWA_START_FRAC": "advanced stochastic weight averaging start point",
    "USE_BIGRAM_HASH": "advanced BigramHash input augmentation",
    "USE_SMEARGATE": "advanced SmearGate input augmentation",
}


def get_active_advanced_env_vars(env: Mapping[str, str] | None = None) -> list[str]:
    source = os.environ if env is None else env
    return sorted(name for name in ADVANCED_ONLY_ENV_VARS if name in source)


def reject_advanced_env_vars(script_name: str, env: Mapping[str, str] | None = None) -> None:
    active = get_active_advanced_env_vars(env)
    if not active:
        return
    joined = ", ".join(active)
    raise ValueError(
        f"{script_name} does not support advanced framework env vars. "
        f"Use train_gpt_advanced.py instead. Unsupported vars: {joined}"
    )


def validate_advanced_args(args: object) -> None:
    eval_mode = str(getattr(args, "eval_mode"))
    if eval_mode not in {"standard", "sliding"}:
        raise ValueError(f"EVAL_MODE must be 'standard' or 'sliding', got {eval_mode}")
    eval_stride = int(getattr(args, "eval_stride"))
    if eval_mode == "sliding" and eval_stride <= 0:
        raise ValueError("Sliding evaluation requires EVAL_STRIDE > 0")
    if eval_stride < 0:
        raise ValueError(f"EVAL_STRIDE must be >= 0, got {eval_stride}")
    if int(getattr(args, "eval_batch_seqs")) <= 0:
        raise ValueError("EVAL_BATCH_SEQS must be positive")

    use_bigram_hash = bool(getattr(args, "use_bigram_hash"))
    if use_bigram_hash and int(getattr(args, "bigram_vocab_size")) <= 0:
        raise ValueError("BigramHash requires BIGRAM_VOCAB_SIZE > 0")
    if use_bigram_hash and int(getattr(args, "bigram_dim")) <= 0:
        raise ValueError("BigramHash requires BIGRAM_DIM > 0")

    init_mode = str(getattr(args, "init_mode"))
    if init_mode not in {"default", "orthogonal"}:
        raise ValueError(f"INIT_MODE must be 'default' or 'orthogonal', got {init_mode}")

    qat_bits = int(getattr(args, "qat_bits"))
    if bool(getattr(args, "qat_enabled")) and qat_bits not in {4, 5, 6, 7, 8}:
        raise ValueError(f"QAT_BITS must be one of 4, 5, 6, 7, 8, got {qat_bits}")

    codec = str(getattr(args, "export_codec"))
    if codec not in {"zlib", "zstd"}:
        raise ValueError(f"EXPORT_CODEC must be 'zlib' or 'zstd', got {codec}")
    level = int(getattr(args, "export_codec_level"))
    if level < 0:
        raise ValueError(f"EXPORT_CODEC_LEVEL must be >= 0, got {level}")

    tied_embed_mode = str(getattr(args, "export_tied_embed_mode"))
    if tied_embed_mode not in {"int8", "fp16"}:
        raise ValueError(
            f"EXPORT_TIED_EMBED_MODE must be 'int8' or 'fp16', got {tied_embed_mode}"
        )
    late_k_mode = str(getattr(args, "export_late_k_mode"))
    if late_k_mode not in {"quantized", "fp16"}:
        raise ValueError(
            f"EXPORT_LATE_K_MODE must be 'quantized' or 'fp16', got {late_k_mode}"
        )

    for name in ("export_attn_weight_bits", "export_mlp_weight_bits"):
        bits = int(getattr(args, name))
        if bits not in {4, 5, 6, 7, 8}:
            env_name = name.upper()
            raise ValueError(f"{env_name} must be one of 4, 5, 6, 7, 8, got {bits}")

    if bool(getattr(args, "swa_enabled")):
        swa_every = int(getattr(args, "swa_every"))
        if swa_every <= 0:
            raise ValueError("SWA_EVERY must be positive when SWA is enabled")
        swa_start_frac = float(getattr(args, "swa_start_frac"))
        if not 0.0 <= swa_start_frac <= 1.0:
            raise ValueError("SWA_START_FRAC must be in [0, 1]")


def build_feature_tags(args: object) -> dict[str, str]:
    input_features: list[str] = []
    if bool(getattr(args, "use_smeargate")):
        input_features.append("smeargate")
    if bool(getattr(args, "use_bigram_hash")):
        input_features.append("bigram_hash")
    feature_input_stack = "+".join(input_features) if input_features else "baseline"

    train_features: list[str] = []
    init_mode = str(getattr(args, "init_mode"))
    if init_mode != "default":
        train_features.append(init_mode)
    if bool(getattr(args, "qat_enabled")):
        train_features.append(f"qat{int(getattr(args, 'qat_bits'))}")
    if bool(getattr(args, "swa_enabled")):
        train_features.append(
            f"swa{int(getattr(args, 'swa_every'))}@{float(getattr(args, 'swa_start_frac')):.2f}"
        )
    muon_weight_decay = float(getattr(args, "muon_weight_decay"))
    adam_weight_decay = float(getattr(args, "adam_weight_decay"))
    if muon_weight_decay > 0:
        train_features.append(f"muwd{muon_weight_decay:g}")
    if adam_weight_decay > 0:
        train_features.append(f"adamwd{adam_weight_decay:g}")
    feature_train_stack = "+".join(train_features) if train_features else "baseline"

    tied_embed_mode = str(getattr(args, "export_tied_embed_mode"))
    late_k_mode = str(getattr(args, "export_late_k_mode"))
    export_codec = str(getattr(args, "export_codec"))
    feature_export_stack = (
        f"{export_codec}_tied-{tied_embed_mode}_latek-{late_k_mode}"
    )

    attn_bits = int(getattr(args, "export_attn_weight_bits"))
    mlp_bits = int(getattr(args, "export_mlp_weight_bits"))
    feature_quant_profile = f"attn{attn_bits}_mlp{mlp_bits}_tied-{tied_embed_mode}"

    return {
        "feature_eval_mode": str(getattr(args, "eval_mode")),
        "feature_input_stack": feature_input_stack,
        "feature_train_stack": feature_train_stack,
        "feature_export_stack": feature_export_stack,
        "feature_quant_profile": feature_quant_profile,
    }
