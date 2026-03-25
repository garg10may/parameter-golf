from __future__ import annotations

import copy
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from experiment_tracking import SQLiteExperimentTracker, collect_hyperparameters
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from advanced_cuda_runtime import (
    AdvancedGPT,
    SWAState,
    compress_bytes,
    decompress_bytes,
    dequantize_state_dict_advanced,
    eval_val_sliding,
    quantize_state_dict_advanced,
)
from advanced_features import build_feature_tags, validate_advanced_args
from train_gpt import (
    CastedLinear,
    DistributedTokenLoader,
    Muon,
    build_sentencepiece_luts,
    eval_val,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
    zeropower_via_newtonschulz5,
)


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    fast_dev_run = bool(int(os.environ.get("FAST_DEV_RUN", "0")))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.0))
    adam_weight_decay = float(os.environ.get("ADAM_WEIGHT_DECAY", 0.0))

    use_smeargate = bool(int(os.environ.get("USE_SMEARGATE", "0")))
    use_bigram_hash = bool(int(os.environ.get("USE_BIGRAM_HASH", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 4096))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))

    eval_mode = os.environ.get("EVAL_MODE", "standard")
    eval_stride = int(os.environ.get("EVAL_STRIDE", "0"))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", "32"))

    init_mode = os.environ.get("INIT_MODE", "default")
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    qat_bits = int(os.environ.get("QAT_BITS", "6"))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "0")))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", "0.5"))
    swa_every = int(os.environ.get("SWA_EVERY", "50"))

    export_codec = os.environ.get("EXPORT_CODEC", "zlib")
    export_codec_level = int(os.environ.get("EXPORT_CODEC_LEVEL", "0"))
    export_tied_embed_mode = os.environ.get("EXPORT_TIED_EMBED_MODE", "int8")
    export_attn_weight_bits = int(os.environ.get("EXPORT_ATTN_WEIGHT_BITS", "8"))
    export_mlp_weight_bits = int(os.environ.get("EXPORT_MLP_WEIGHT_BITS", "8"))
    export_late_k_mode = os.environ.get("EXPORT_LATE_K_MODE", "quantized")


def apply_decoupled_weight_decay(param_groups: list[dict[str, object]]) -> None:
    with torch.no_grad():
        for group in param_groups:
            weight_decay = float(group.get("weight_decay", 0.0))
            lr = float(group.get("lr", 0.0))
            if weight_decay <= 0.0 or lr <= 0.0:
                continue
            for param in group["params"]:
                param.mul_(1.0 - weight_decay * lr)


def main() -> None:
    global zeropower_via_newtonschulz5

    args = Hyperparameters()
    validate_advanced_args(args)
    code = Path(__file__).read_text(encoding="utf-8")
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)
        tracker: SQLiteExperimentTracker | None = SQLiteExperimentTracker.from_env(
            script_name="train_gpt_advanced.py",
            backend="cuda",
            run_id=args.run_id,
            log_path=logfile,
        )
        tracker.start_run(notes=os.environ.get("EXPERIMENT_COMMENT"))
    else:
        tracker = None

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    def log_section(title: str, rows: list[tuple[str, object]]) -> None:
        log0(f"[{title}]")
        for key, value in rows:
            log0(f"  {key}: {value}")

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    feature_tags = build_feature_tags(args)
    if tracker is not None:
        tracker.log_params(collect_hyperparameters(args))
        tracker.log_params(
            {
                "dataset_name": dataset_dir.name,
                "actual_train_shards": actual_train_files,
                "world_size": world_size,
                "grad_accum_steps": grad_accum_steps,
                "experiment_group": os.environ.get("EXPERIMENT_GROUP"),
                "experiment_label": os.environ.get("EXPERIMENT_LABEL"),
                "experiment_comment": os.environ.get("EXPERIMENT_COMMENT"),
                "launch_source": os.environ.get("LAUNCH_SOURCE"),
                "launch_platform": os.environ.get("LAUNCH_PLATFORM"),
                "launch_device_kind": os.environ.get("LAUNCH_DEVICE_KIND"),
                "launch_device_count": int(os.environ["LAUNCH_DEVICE_COUNT"]) if "LAUNCH_DEVICE_COUNT" in os.environ else None,
                "launch_resolved_script": os.environ.get("LAUNCH_RESOLVED_SCRIPT"),
                "launch_trainer_mode": os.environ.get("LAUNCH_TRAINER_MODE"),
                **feature_tags,
            }
        )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log_section(
        "Run",
        [
            ("run_id", args.run_id),
            ("pytorch_version", torch.__version__),
            ("world_size", world_size),
            ("fast_dev_run", args.fast_dev_run),
            ("device", device),
        ],
    )
    log_section(
        "Features",
        [
            ("eval_mode", feature_tags["feature_eval_mode"]),
            ("input_stack", feature_tags["feature_input_stack"]),
            ("train_stack", feature_tags["feature_train_stack"]),
            ("export_stack", feature_tags["feature_export_stack"]),
            ("quant_profile", feature_tags["feature_quant_profile"]),
        ],
    )

    base_model = AdvancedGPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        use_smeargate=args.use_smeargate,
        use_bigram_hash=args.use_bigram_hash,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        init_mode=args.init_mode,
        qat_enabled=args.qat_enabled,
        qat_bits=args.qat_bits,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    if base_model.smeargate is not None and base_model.smeargate.gate.dtype != torch.float32:
        base_model.smeargate.gate.data = base_model.smeargate.gate.data.float()
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    token_params = [base_model.tok_emb.weight]
    if base_model.bigram is not None:
        token_params.append(base_model.bigram.embed.weight)
    head_params = [base_model.lm_head.weight] if base_model.lm_head is not None else []
    excluded_ids = {id(param) for param in token_params + head_params}
    matrix_params: list[nn.Parameter] = []
    scalar_params: list[nn.Parameter] = []
    for name, param in base_model.named_parameters():
        if id(param) in excluded_ids:
            continue
        if param.ndim == 2 and ".blocks." not in f".{name}." and name != "skip_weights":
            matrix_params.append(param)
            continue
        if param.ndim == 2 and not any(pattern in name for pattern in ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")):
            matrix_params.append(param)
        else:
            scalar_params.append(param)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": token_params, "lr": token_lr, "base_lr": token_lr, "weight_decay": args.adam_weight_decay}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
        group["weight_decay"] = args.muon_weight_decay
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr, "weight_decay": args.adam_weight_decay}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if head_params:
        optimizer_head = torch.optim.AdamW(
            [{"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr, "weight_decay": args.adam_weight_decay}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    if tracker is not None:
        tracker.log_params({"model_param_count": n_params})
    log_section(
        "Model",
        [
            ("params", n_params),
            ("vocab_size", args.vocab_size),
            ("layers", args.num_layers),
            ("dim", args.model_dim),
            ("heads", args.num_heads),
            ("kv_heads", args.num_kv_heads),
            ("seq_len", args.train_seq_len),
            ("tie_embeddings", args.tie_embeddings),
            ("attention_mode", "gqa"),
        ],
    )
    log_section(
        "Optimizer",
        [
            ("embed_lr", token_lr),
            ("matrix_lr", args.matrix_lr),
            ("scalar_lr", args.scalar_lr),
            ("muon_momentum", args.muon_momentum),
            ("muon_weight_decay", args.muon_weight_decay),
            ("adam_weight_decay", args.adam_weight_decay),
        ],
    )

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    swa_state = SWAState()
    swa_start_step = max(1, int(math.ceil(args.iterations * args.swa_start_frac)))

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if warmdown_start <= step < args.iterations
                else 1.0
            )
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    def run_eval() -> tuple[float, float]:
        if args.eval_mode == "sliding":
            return eval_val_sliding(
                args,
                base_model,
                rank,
                world_size,
                device,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
        return eval_val(
            args,
            base_model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            apply_decoupled_weight_decay(optimizer_muon.param_groups)
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = (not last_step and args.val_loss_every > 0 and step % args.val_loss_every == 0) or (
            last_step and not args.fast_dev_run
        )
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = run_eval()
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            if tracker is not None:
                tracker.log_metric(phase="val", name="val_loss", value=val_loss, step=step)
                tracker.log_metric(phase="val", name="val_bpb", value=val_bpb, step=step)
                tracker.log_metric(phase="val", name="train_time_ms", value=training_time_ms, step=step)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        apply_decoupled_weight_decay(optimizer_muon.param_groups)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        if args.swa_enabled and step >= swa_start_step and (step - swa_start_step) % args.swa_every == 0:
            swa_state.update(base_model)

        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
            if tracker is not None:
                tracker.log_metric(phase="train", name="train_loss", value=float(train_loss.item()), step=step)
                tracker.log_metric(phase="train", name="train_time_ms", value=approx_training_time_ms, step=step)

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    if tracker is not None:
        tracker.log_metric(
            phase="system",
            name="peak_memory_allocated_mib",
            value=torch.cuda.max_memory_allocated() // 1024 // 1024,
            step=step,
        )
        tracker.log_metric(
            phase="system",
            name="peak_memory_reserved_mib",
            value=torch.cuda.max_memory_reserved() // 1024 // 1024,
            step=step,
        )

    if args.fast_dev_run:
        log0("fast_dev_run:skipping final validation, serialization, and quantized roundtrip eval")
        if tracker is not None:
            tracker.finish(status="completed")
            tracker.close()
        if distributed:
            dist.destroy_process_group()
        return

    if args.swa_enabled:
        previous_state = swa_state.apply(base_model)
        if previous_state is None:
            log0("swa:enabled but no checkpoints collected; continuing with live weights")
    else:
        previous_state = None

    code_paths = [
        Path(__file__),
        Path(__file__).with_name("advanced_features.py"),
        Path(__file__).with_name("advanced_cuda_runtime.py"),
        Path(__file__).with_name("train_gpt.py"),
        Path(__file__).with_name("experiment_tracking.py"),
    ]
    code_bytes = sum(path.stat().st_size for path in code_paths if path.exists())

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code bundle size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")
        if tracker is not None:
            tracker.log_artifact(name="final_model_pt", num_bytes=model_bytes, metadata={"code_bytes": code_bytes})

    quant_obj, quant_stats = quantize_state_dict_advanced(base_model.state_dict(), args)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = compress_bytes(quant_raw, args.export_codec, args.export_codec_level)
    quant_raw_bytes = len(quant_raw)
    artifact_name = f"final_model.{args.export_codec}.ptz"
    if master_process:
        with open(artifact_name, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(artifact_name)
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model quantized+{args.export_codec}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size quantized+{args.export_codec}: {quant_file_bytes + code_bytes} bytes")
        if tracker is not None:
            tracker.log_artifact(
                name=artifact_name,
                num_bytes=quant_file_bytes,
                metadata={
                    "payload_bytes": quant_stats["int8_payload_bytes"],
                    "raw_torch_bytes": quant_raw_bytes,
                    "payload_ratio": ratio,
                    "code_bytes": code_bytes,
                    "codec": args.export_codec,
                },
            )

    if distributed:
        dist.barrier()
    with open(artifact_name, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(decompress_bytes(quant_blob_disk, args.export_codec)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_advanced(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = run_eval()
    torch.cuda.synchronize()
    q_eval_ms = 1000.0 * (time.perf_counter() - t_qeval)
    log0(
        f"final_quantized_roundtrip codec:{args.export_codec} val_loss:{q_val_loss:.4f} "
        f"val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms"
    )
    log0(f"final_quantized_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    if tracker is not None:
        tracker.log_metric(phase="roundtrip", name="val_loss", value=q_val_loss, step=step)
        tracker.log_metric(phase="roundtrip", name="val_bpb", value=q_val_bpb, step=step)
        tracker.log_metric(phase="roundtrip", name="eval_time_ms", value=q_eval_ms, step=step)
        tracker.finish(status="completed")
        tracker.close()

    if previous_state is not None:
        base_model.load_state_dict(previous_state, strict=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
