import argparse
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Optional
from copy import deepcopy

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from transformers import Qwen3VLForConditionalGeneration as Qwen3ModelForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters_qwen import (
    configure_adapter_clip_length,
    finalize_adalora,
    freeze_model_parameters,
    infer_visual_hidden_size,
    setup_adapter,
)
from data import (
    DEFAULT_ENDOVIS_TRAIN_SEQS,
    DEFAULT_ENDOVIS_VAL_SEQS,
    DEFAULT_ENDOVIS_EVAL_SEQS,
    EndoVisSFT,
    build_endovis_train_dataset,
    build_real_dataset,
    default_answer_prefixes,
    flatten_itemlists,
    infer_clip_length_from_dataset,
)
from training import (
    Qwen3VLBatchCollator,
    Qwen3VideoEvalCollator,
    do_qwen_evaluations,
    load_trained_weights,
    run_qwen_training_loop,
    save_model_state,
    save_run_settings,
    update_run_settings_with_trainable,
)

warnings.filterwarnings("ignore")





def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL on REAL VQA")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Pretrained model name or path.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--preview", action="store_true", help="run in preview mode with limited data")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print adapter debug info such as modules_to_save.",
    )

    # NEW: repeat 5 seeds
    parser.add_argument(
        "--repeat_5",
        action="store_true",
        help="Repeat full train+eval 5 times with seeds seed..seed+4 and save under out_dir_repeated_5/<run_name>/seedXX/",
    )

    parser.add_argument(
        "--real_annotations_path",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "datasets" / "real_colon_vqa"
            / "real_colon_vqa_in_domain.jsonl"
        ),
    )
    parser.add_argument(
        "--real_ood_annotations_path",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "datasets" / "real_colon_vqa"
            / "real_colon_vqa_out_of_domain.jsonl"
        ),
    )
    parser.add_argument(
        "--real_frames_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "frames_resized"),
    )
    parser.add_argument(
        "--endovis_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "EndoVis-18-VQA"),
    )
    parser.add_argument("--endovis_tail", type=str, default=os.path.join("vqa", "Sentence"))
    parser.add_argument(
        "--endovis_ood_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "Endovis18-VQA_Out-of-Template"),
    )
    parser.add_argument("--sequence_length", type=int, default=8)
    parser.add_argument(
        "--train_endovis",
        action="store_true",
        help="Train on EndoVis sequences (evaluation domain is set automatically).",
    )
    parser.add_argument(
        "--merge_val_train",
        action="store_true",
        help="Merge validation split into training and pick best epoch on train loss.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="If >0, limit REAL dataset clips to the first N frames.",
    )
    parser.add_argument(
        "--adapter_frames",
        type=int,
        default=8,
        help="Number of frames processed by vision adapters (<=0 uses dataset default).",
    )
    parser.add_argument("--video_fps", type=float, default=1.0)
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=0,
        help="Maximum number of tokens forwarded to the tokenizer (<=0 keeps HF default).",
    )

    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="Number of gradient accumulation steps before each optimizer update.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout probability")
    parser.add_argument("--adalora_r", type=int, default=8, help="Initial AdaLoRA rank")
    parser.add_argument("--adalora_tinit", type=int, default=200)
    parser.add_argument("--adalora_tfinal", type=int, default=1000)
    parser.add_argument("--adalora_deltaT", type=int, default=10)

    parser.add_argument(
        "--peft_enc",
        type=str,
        choices=[
            "none",
            "frozen",
            "lora_enc",
            "dora_enc",
            "vera_enc",
            "adalora_enc",
            "stlora_enc",
            "stdora_enc",
            "stlora_mhatt_enc",
            "stdora_mhatt_enc",
            "stlora_selfatt_enc",
            "stdora_selfatt_enc",
            "stlora_lstm_enc",
            "stdora_lstm_enc",
            "stmamba_enc",
            "stdora_mamba_enc",
            "timeshiftlora_enc",
            "timeshiftdora_enc",
            "gated_headwise",
            "gated_elementwise",
        ],
        default="none",
        help="Vision-side PEFT choice.",
    )
    parser.add_argument(
        "--peft_dec",
        type=str,
        choices=[
            "none",
            "frozen",
            "lora",
            "dora",
            "vera",
            "adalora",
            "stlora",
            "stdora",
            "stlora_mhatt",
            "stdora_mhatt",
            "stlora_selfatt",
            "stdora_selfatt",
            "stlora_lstm",
            "stdora_lstm",
            "stmamba",
            "stdora_mamba",
            "timeshiftlora",
            "timeshiftdora",
        ],
        default="lora",
        help="Decoder-side PEFT choice.",
    )

    parser.add_argument("--adapter_rank", type=int, default=8)
    parser.add_argument("--adapter_alpha", type=float, default=16.0)
    parser.add_argument("--adapter_dropout", type=float, default=0.1)
    parser.add_argument("--st_hidden", type=int, default=0)
    parser.add_argument("--st_rank", type=int, default=32)
    parser.add_argument("--st_bottleneck", type=int, default=64)
    parser.add_argument("--st_alpha", type=float, default=128.0)
    parser.add_argument("--st_kernel", type=int, default=3)
    parser.add_argument("--st_dropout", type=float, default=0.1)
    parser.add_argument("--st_adapter_channels", type=int, default=384)
    parser.add_argument("--st_adapter_kernel_t", type=int, default=3)
    parser.add_argument("--st_adapter_kernel_h", type=int, default=1)
    parser.add_argument("--st_adapter_kernel_w", type=int, default=1)
    parser.add_argument("--st_adapter_disable_cudnn", action="store_true")
    parser.add_argument("--tslora_hidden", type=int, default=0)
    parser.add_argument("--tslora_n_div", type=int, default=3)

    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="Attention implementation passed to `attn_implementation`.",
    )

    parser.add_argument("--no_eval_real", dest="eval_real", action="store_false")
    parser.add_argument("--no_eval_real_ood", dest="eval_real_ood", action="store_false")
    parser.add_argument("--no_eval_endovis", dest="eval_endovis", action="store_false")
    parser.add_argument("--no_eval_endovis_ood", dest="eval_endovis_ood", action="store_false")

    parser.add_argument("--max_new_tokens_eval", type=int, default=64)
    parser.add_argument("--max_samples_eval", type=int, default=None)

    parser.add_argument("--out_dir", type=str, default="./outputs/qwen3vl")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional directory for Hugging Face-format checkpoints (defaults to the run folder).",
    )
    parser.add_argument("--load_pretrained", type=str, default=None)
    parser.add_argument(
        "--reload_best",
        action="store_true",
        help="Reload best_model.pt from the run directory before training/eval.",
    )
    parser.add_argument(
        "--reload_last",
        action="store_true",
        help="Reload last_model.pt from the run directory before training/eval.",
    )
    parser.add_argument("--zero_shot_eval", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--debug_shapes",
        action="store_true",
        help="Print shapes of key tensors for the first batch and first model outputs.",
    )

    args = parser.parse_args()
    return args


def build_default_run_name(args) -> str:
    def _normalize_name(value: Optional[str]) -> str:
        if value:
            return value
        return "none"

    def _resolve_run_peft_names() -> tuple[str, str]:
        enc_name = _normalize_name(getattr(args, "peft_enc", None))
        dec_name = _normalize_name(getattr(args, "peft_dec", None))
        adapter = getattr(args, "adapter", None)
        adapter_encoder = getattr(args, "adapter_encoder", False)
        decoder_also = getattr(args, "decoder_also", False)
        skip_decoder_lora = getattr(args, "skip_decoder_lora", False)

        if adapter_encoder and enc_name in {"none", "frozen"} and adapter not in {None, "none"}:
            enc_name = adapter

        if skip_decoder_lora:
            return enc_name, dec_name

        if adapter not in {None, "none"}:
            if dec_name in {"none", "frozen"} and decoder_also:
                dec_name = adapter
            elif dec_name == "lora" and adapter != "lora":
                dec_name = adapter

        return enc_name, dec_name

    enc_name, dec_name = _resolve_run_peft_names()
    parts = ["run_qwen3vl"]
    parts.append("zero-shot" if args.zero_shot_eval else "finetune")
    parts.append("endovis-train" if args.train_endovis else "real-train")
    parts.append(f"enc{enc_name}")
    parts.append(f"dec{dec_name}")
    parts.append(f"lr{args.lr}")
    parts.append(f"gradacc{args.grad_accum}")
    parts.append(f"ep{args.epochs}")
    if getattr(args, "frames", 0) and args.frames > 0:
        parts.append(f"frames{args.frames}")
    return "_".join(parts)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DatasetWithFramePaths(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if "frame_paths" in item:
            return item
        if hasattr(self.base, "samples"):
            samples = getattr(self.base, "samples")
            if isinstance(samples, list) and idx < len(samples):
                frame_paths = samples[idx].get("frame_paths", [])
                item = dict(item)
                item["frame_paths"] = frame_paths
        return item


def run_once(args) -> None:
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose at most one of --load_in_4bit or --load_in_8bit.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # IMPORTANT: to match requested structure, run_dir is exactly args.out_dir
    run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[IO] Run directory: {run_dir}")
    save_run_settings(run_dir, vars(args))

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            use_fast=False,
        )
    processor.tokenizer = tokenizer

    if hasattr(processor, "video_processor"):
        processor.video_processor.do_sample_frames = False

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
    max_len_arg = getattr(args, "tokenizer_max_length", None)
    if max_len_arg is not None and max_len_arg > 0:
        tokenizer.model_max_length = max_len_arg
        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            init_kwargs["model_max_length"] = max_len_arg

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    quantization_config = None
    model_torch_dtype = dtype

    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_cuda else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_torch_dtype = None
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_torch_dtype = None

    print("Loading model...")
    model_cls = Qwen3ModelForConditionalGeneration or AutoModelForCausalLM
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": model_torch_dtype,
        "low_cpu_mem_usage": True,
        "device_map": "cuda" if use_cuda else None,
        "quantization_config": quantization_config,
    }
    if args.attn_impl and args.attn_impl.lower() != "auto":
        model_kwargs["attn_implementation"] = args.attn_impl
    model = model_cls.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    inferred_hidden = infer_visual_hidden_size(model, args.st_hidden)
    if args.st_hidden <= 0:
        args.st_hidden = inferred_hidden
    elif args.st_hidden != inferred_hidden:
        print(
            f"[ST] Requested hidden size {args.st_hidden} but model uses {inferred_hidden}; "
            f"overriding to match."
        )
        args.st_hidden = inferred_hidden

    freeze_model_parameters(model)
    model, need_adalora_setup = setup_adapter(model, args)
    if set(DEFAULT_ENDOVIS_TRAIN_SEQS) & set(DEFAULT_ENDOVIS_VAL_SEQS):
        raise RuntimeError("EndoVis train/val sequence overlap detected.")
    if set(DEFAULT_ENDOVIS_TRAIN_SEQS) & set(DEFAULT_ENDOVIS_EVAL_SEQS):
        raise RuntimeError("EndoVis train/eval sequence overlap detected.")
    if set(DEFAULT_ENDOVIS_VAL_SEQS) & set(DEFAULT_ENDOVIS_EVAL_SEQS):
        raise RuntimeError("EndoVis val/eval sequence overlap detected.")
    print(f"[Split] EndoVis train sequences: {DEFAULT_ENDOVIS_TRAIN_SEQS}")
    print(f"[Split] EndoVis val sequences: {DEFAULT_ENDOVIS_VAL_SEQS}")
    print(f"[Split] EndoVis eval sequences (in/out template): {DEFAULT_ENDOVIS_EVAL_SEQS}")

    valid_dataset = None
    if args.train_endovis:
        train_dataset = DatasetWithFramePaths(
            build_endovis_train_dataset(args, DEFAULT_ENDOVIS_TRAIN_SEQS)
        )
        valid_dataset = DatasetWithFramePaths(
            build_endovis_train_dataset(args, DEFAULT_ENDOVIS_VAL_SEQS)
        )
    else:
        train_dataset = DatasetWithFramePaths(
            build_real_dataset(args, args.real_annotations_path, mode="train")
        )
        try:
            valid_dataset = DatasetWithFramePaths(
                build_real_dataset(args, args.real_annotations_path, mode="val")
            )
        except ValueError as exc:
            print(f"[Val] REAL validation split unavailable; skipping val loss. Details: {exc}")
    if args.merge_val_train and valid_dataset is not None:
        train_dataset = ConcatDataset([train_dataset, valid_dataset])
        valid_dataset = None
    clip_length = args.adapter_frames if args.adapter_frames and args.adapter_frames > 0 else None
    if clip_length is None:
        if args.train_endovis:
            clip_length = args.sequence_length
            print(f"[Adapter] Using clip length from --sequence_length: {clip_length}")
        else:
            clip_length = infer_clip_length_from_dataset(train_dataset.base)
            print(f"[Adapter] Using clip length inferred from dataset: {clip_length}")
    else:
        print(f"[Adapter] Using clip length from --adapter_frames: {clip_length}")
    frames_limit = args.frames if getattr(args, "frames", 0) and args.frames > 0 else args.sequence_length
    effective_clip_length = min(clip_length, frames_limit) if frames_limit else clip_length
    configure_adapter_clip_length(model, effective_clip_length)

    train_prompt = getattr(args, "_train_system_prompt", "")
    collator = Qwen3VLBatchCollator(
        processor,
        video_fps=args.video_fps,
        max_frames=frames_limit,
        system_prompt=train_prompt,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collator,
        )
    eval_collator = Qwen3VideoEvalCollator(max_frames=frames_limit)
    test_dataset = None
    real_ood_dataset = None
    endovis_eval_dataset = None
    endovis_ood_eval_dataset = None

    real_eval_loader = None
    if args.eval_real:
        test_dataset = DatasetWithFramePaths(
            build_real_dataset(args, args.real_annotations_path, mode="test")
        )
        real_eval_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=eval_collator,
        )

    real_ood_eval_loader = None
    if args.eval_real_ood:
        if Path(args.real_ood_annotations_path).exists():
            real_ood_dataset = DatasetWithFramePaths(
                build_real_dataset(args, args.real_ood_annotations_path, mode="test")
            )
            real_ood_eval_loader = DataLoader(
                real_ood_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=eval_collator,
            )
        else:
            print(f"[Eval] REAL OOD annotations not found: {args.real_ood_annotations_path}")

    endovis_eval_loader = None
    if args.eval_endovis:
        endovis_eval_dataset = EndoVisSFT(
            DEFAULT_ENDOVIS_EVAL_SEQS,
            args.endovis_root,
            args.endovis_tail,
            sequence_length=args.sequence_length,
            use_answer_strip=False,
            strip_prefixes=default_answer_prefixes(),
            image_size=(448, 448),
        )
        endovis_eval_loader = DataLoader(
            endovis_eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=flatten_itemlists,
        )

    endovis_ood_eval_loader = None
    if getattr(args, "eval_endovis_ood", False):
        endovis_ood_root = Path(args.endovis_ood_root)
        if endovis_ood_root.exists():
            endovis_ood_eval_dataset = EndoVisSFT(
                DEFAULT_ENDOVIS_EVAL_SEQS,
                str(endovis_ood_root),
                args.endovis_tail,
                sequence_length=args.sequence_length,
                use_answer_strip=False,
                strip_prefixes=default_answer_prefixes(),
                image_size=(448, 448),
            )
            endovis_ood_eval_loader = DataLoader(
                endovis_ood_eval_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=flatten_itemlists,
            )
        else:
            print(f"[Eval] EndoVis OOT root not found: {endovis_ood_root}")

    print(f"[Data] Train samples: {len(train_dataset)}")
    if valid_dataset is not None:
        if args.train_endovis:
            print(f"[Data] EndoVis val samples: {len(valid_dataset)}")
        else:
            print(f"[Data] REAL val samples: {len(valid_dataset)}")
    elif not args.train_endovis:
        print("[Data] REAL val samples: unavailable")
    if test_dataset is not None:
        print(f"[Data] REAL test samples: {len(test_dataset)}")
    if real_ood_dataset is not None:
        print(f"[Data] REAL OOT test samples: {len(real_ood_dataset)}")
    if endovis_eval_dataset is not None:
        print(f"[Data] EndoVis in-template eval samples: {len(endovis_eval_dataset)}")
    if endovis_ood_eval_dataset is not None:
        print(f"[Data] EndoVis out-of-template eval samples: {len(endovis_ood_eval_dataset)}")

    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty; check annotation split filters.")

    total_steps = max(1, len(train_loader) * max(1, args.epochs))
    if need_adalora_setup:
        model = finalize_adalora(model, args, total_steps)

    if quantization_config is None:
        model.to(device)

    if args.reload_best and args.reload_last:
        raise ValueError("Choose at most one of --reload_best or --reload_last.")

    reload_path = None
    if args.reload_best or args.reload_last:
        reload_name = "best_model.pt" if args.reload_best else "last_model.pt"
        reload_path = run_dir / reload_name
        if args.load_pretrained:
            print("[Load] Ignoring --load_pretrained because --reload_* was set.")
        if reload_path.exists():
            model = load_trained_weights(model, str(reload_path), device)
            if quantization_config is None:
                model.to(device)
            configure_adapter_clip_length(model, effective_clip_length)
        else:
            print(f"[Load] Reload checkpoint not found: {reload_path}")
            reload_path = None

    if args.load_pretrained and reload_path is None:
        model = load_trained_weights(model, args.load_pretrained, device)
        if quantization_config is None:
            model.to(device)

    wandb_module = None
    wandb_run = None
    if args.use_wandb:
        import wandb  # type: ignore

        wandb_module = wandb
        wandb_kwargs = dict(
            project=args.wandb_project or "train_qwen3_vl",
            entity=args.wandb_entity,
            name=f"{args.run_name}_seed{args.seed}" if getattr(args, "repeat_5", False) else args.run_name,
            dir=str(run_dir),
            config=vars(args),
        )
        # optional grouping across seeds
        wandb_group = getattr(args, "_wandb_group", None)
        if wandb_group:
            wandb_kwargs["group"] = wandb_group
        wandb_run = wandb_module.init(**wandb_kwargs)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Trainable params percentage: {100 * trainable_params / total_params:.2f}% "
        f"({trainable_params}/{total_params})"
    )
    update_run_settings_with_trainable(run_dir, total_params, trainable_params)

    if args.zero_shot_eval or args.epochs <= 0:
        print("[Mode] Zero-shot evaluation only.")
        do_qwen_evaluations(
            model,
            processor,
            device,
            args,
            run_dir,
            real_eval_loader=real_eval_loader,
            real_ood_eval_loader=real_ood_eval_loader,
            endovis_eval_loader=endovis_eval_loader,
            endovis_ood_eval_loader=endovis_ood_eval_loader,
            wandb_module=wandb_module,
            wandb_step=None,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    vision_dtype = next(model.parameters()).dtype
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_qwen_training_loop(
        args,
        model,
        processor,
        train_loader,
        device,
        optimizer,
        vision_dtype,
        run_dir,
        valid_loader=valid_loader,
        wandb_module=wandb_module,
    )

    last_model_path = run_dir / "last_model.pt"
    save_model_state(model, last_model_path)
    with open(run_dir / "last.txt", "w") as f:
        f.write(f"{last_model_path.name}\n")

    best_state_path = run_dir / "best_model.pt"
    if best_state_path.exists():
        model = load_trained_weights(model, str(best_state_path), device)
    else:
        print("[Eval] Best model checkpoint not found; using last state.")

    do_qwen_evaluations(
        model,
        processor,
        device,
        args,
        run_dir,
        real_eval_loader=real_eval_loader,
        real_ood_eval_loader=real_ood_eval_loader,
        endovis_eval_loader=endovis_eval_loader,
        endovis_ood_eval_loader=endovis_ood_eval_loader,
        wandb_module=wandb_module,
        wandb_step=args.epochs,
    )

    if wandb_run is not None:
        wandb_run.finish()


def main():
    args = parse_args()

    output_root = Path(__file__).resolve().parents[1]
    dataset_tag = "endovis" if args.train_endovis else "real"
    out_suffix = os.environ.get("TRAIN_OUT_SUFFIX", "").strip()
    args.out_dir = str(output_root / "outputs" / f"qwen3vl_{dataset_tag}_{args.epochs}{out_suffix}")
    if args.merge_val_train:
        args.out_dir += "_mergedvaltrain"
    # Training uses no system prompt (real or EndoVis); evaluation keeps its own defaults.
    args._train_system_prompt = ""
    if args.train_endovis:
        args.eval_real = False
        args.eval_real_ood = False
    else:
        args.eval_endovis = False
        args.eval_endovis_ood = False

    # Validate EndoVis root; try fallback once, else abort.
    endovis_root_path = Path(args.endovis_root)
    if not endovis_root_path.exists():
        alt_root = Path(__file__).resolve().parents[1] / "REAL_dataset_annotation" / "EndoVis-18-VQA"
        if alt_root.exists():
            print(f"[EndoVis] endovis_root missing, switching to {alt_root}")
            args.endovis_root = str(alt_root)
        else:
            raise RuntimeError(f"EndoVis root not found at {endovis_root_path}")
    print(
        "[EvalPolicy] "
        f"real={args.eval_real} real_ood={args.eval_real_ood} "
        f"endovis={args.eval_endovis} endovis_ood={args.eval_endovis_ood}"
    )

    # preserve an experiment name; folder naming is handled below
    base_run_name = args.run_name or build_default_run_name(args)
    args.run_name = base_run_name

    # Decide seeds
    seeds = [args.seed + i for i in range(5)] if args.repeat_5 else [args.seed]

    # Root output dir
    out_root = Path(f"{args.out_dir}_repeated_5") if args.repeat_5 else Path(args.out_dir)

    # Optional: group wandb runs when repeating
    if args.use_wandb and args.repeat_5:
        setattr(args, "_wandb_group", base_run_name)

    # Run loop
    for seed in seeds:
        run_args = deepcopy(args)
        run_args.seed = seed

        # Folder structure:
        # out_dir_repeated_5/<base_run_name>/seed42/
        if args.repeat_5:
            run_args.out_dir = str(out_root / base_run_name / f"seed{seed}")
        else:
            run_args.out_dir = str(out_root / base_run_name)

        run_once(run_args)


if __name__ == "__main__":
    main()
