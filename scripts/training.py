import json
import operator
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import evaluate
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from system_prompt_real_colon import SYSTEM_PROMPT
from system_prompt_endovis import SYSTEM_PROMPT_ENDOVIS
from data import extract_label_keywords

IGNORE_INDEX = -100


def _progress_meta(iterable: Iterable[Any], default_interval: int = 50) -> Tuple[Optional[int], int]:
    total: Optional[int] = None
    if hasattr(iterable, "__len__"):
        total = len(iterable)  # type: ignore[arg-type]
    else:
        hint = operator.length_hint(iterable, -1)
        total = hint if hint >= 0 else None
    log_every = default_interval if not total else max(1, total // 10)
    return total, log_every


def _should_log(step: int, log_every: int, total: Optional[int]) -> bool:
    if step == 1:
        return True
    if log_every and step % log_every == 0:
        return True
    if total is not None and step == total:
        return True
    return False


def _log_progress(desc: str, step: int, total: Optional[int]) -> None:
    if total:
        print(f"[{desc}] Step {step}/{total}")
    else:
        print(f"[{desc}] Step {step}")


def set_model_img_context_token_id(model: nn.Module, token_id: Optional[int]) -> None:
    if token_id is None:
        raise ValueError("Tokenizer is missing the <IMG_CONTEXT> token.")
    if token_id == getattr(model, "unk_token_id", None):
        raise ValueError("<IMG_CONTEXT> token resolves to unk_token_id; add it to the tokenizer.")

    visited: set[int] = set()

    def _assign(target: Any) -> None:
        if target is None:
            return
        obj_id = id(target)
        if obj_id in visited:
            return
        visited.add(obj_id)
        setattr(target, "img_context_token_id", token_id)

        base_model = getattr(target, "base_model", None)
        if base_model is not None:
            inner = getattr(base_model, "model", None)
            _assign(inner if inner is not None else base_model)

        get_base = getattr(target, "get_base_model", None)
        if callable(get_base):
            _assign(get_base())

        inner_model = getattr(target, "model", None)
        if inner_model is not None and inner_model is not target:
            _assign(inner_model)

    _assign(model)


bleu = evaluate.load("bleu", keep_in_memory=True)
rouge = evaluate.load("rouge", keep_in_memory=True)
meteor = evaluate.load("meteor", keep_in_memory=True)


def get_nlp_metrics(references: List[str], hypotheses: List[str]):
    results_bleu = bleu.compute(predictions=hypotheses, references=references)
    results_rouge = rouge.compute(predictions=hypotheses, references=references)
    results_meteor = meteor.compute(predictions=hypotheses, references=references)
    return results_bleu, results_rouge, results_meteor


class VideoQACollator:
    """Collator that inserts one <image> per frame, InternVL-style."""

    def __init__(
        self,
        tokenizer,
        num_image_token: int,
        system_message: str = "",
        max_length: Optional[int] = None,
        max_frames: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_image_token = num_image_token
        self.max_frames = max_frames if isinstance(max_frames, int) and max_frames > 0 else None

        self.system_message = system_message

        self.img_start_token = "<img>"
        self.img_end_token = "</img>"
        self.img_context_token = "<IMG_CONTEXT>"
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        for tok in (self.img_start_token, self.img_end_token, self.img_context_token):
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            if tok_id is None or (unk_id is not None and tok_id == unk_id):
                raise ValueError(f"Tokenizer is missing required special token: {tok}")

    def __call__(self, batch):
        pixel_values_list = []
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        num_patches_list_batch = []
        image_flags_list = []
        short_answers = []
        questions = []
        answers = []
        num_frames_batch = []

        for sample in batch:
            pixel_values = sample["pixel_values"]
            question = sample["question"]
            answer = sample["answer"]
            num_frames = int(sample.get("num_frames") or pixel_values.size(0))
            if self.max_frames is not None:
                num_frames = min(num_frames, self.max_frames)
                pixel_values = pixel_values[:num_frames]
            num_patches_per_frame = sample.get("num_patches_list") or [1] * num_frames
            if len(num_patches_per_frame) != num_frames:
                num_patches_per_frame = [1] * num_frames
            short_ans = sample.get("short_answer") or ""
            image_flags = sample.get("image_flags")
            if isinstance(image_flags, torch.Tensor) and image_flags.size(0) >= num_frames:
                image_flags = image_flags[:num_frames]
            else:
                image_flags = torch.ones(num_frames, dtype=torch.long)

            frame_lines = [f"Frame{i+1}: <image>" for i in range(num_frames)]
            user_part = "\n".join(frame_lines) + "\n" + question

            if self.system_message and self.system_message.strip():
                input_text = self.system_message + "\nUser: " + user_part + "\nAssistant:"
            else:
                input_text = "User: " + user_part + "\nAssistant:"
            full_text = input_text + " " + answer

            expanded_input_text = input_text
            expanded_full_text = full_text

            for patches_for_this_frame in num_patches_per_frame:
                tokens_for_frame = self.num_image_token * max(1, patches_for_this_frame)
                frame_block = (
                    self.img_start_token
                    + self.img_context_token * tokens_for_frame
                    + self.img_end_token
                )
                expanded_input_text = expanded_input_text.replace("<image>", frame_block, 1)
                expanded_full_text = expanded_full_text.replace("<image>", frame_block, 1)

            input_encoding = self.tokenizer(
                expanded_input_text,
                return_tensors="pt",
                truncation=False,
            )
            full_encoding = self.tokenizer(
                expanded_full_text,
                return_tensors="pt",
                truncation=False,
            )

            input_ids = full_encoding["input_ids"].squeeze(0)
            attention_mask = full_encoding["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            input_actual_length = input_encoding["input_ids"].size(-1)
            full_actual_length = full_encoding["input_ids"].size(-1)

            if self.max_length is not None and full_actual_length > self.max_length:
                raise ValueError(
                    f"Sequence length {full_actual_length} exceeds maximum {self.max_length}. "
                    "Increase max_length to accommodate vision tokens."
                )

            if input_actual_length >= full_actual_length:
                target_start = max(0, full_actual_length - 5)
                labels[:target_start] = IGNORE_INDEX
            else:
                labels[:input_actual_length] = IGNORE_INDEX

            labels[full_actual_length:] = IGNORE_INDEX

            pixel_values_list.append(pixel_values)
            image_flags_list.append(image_flags)
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
            num_patches_list_batch.append(num_patches_per_frame)
            short_answers.append(short_ans)
            questions.append(question)
            answers.append(answer)
            num_frames_batch.append(num_frames)

        pad_token_id = self.tokenizer.pad_token_id
        max_seq_len = max(ids.size(0) for ids in input_ids_list)

        def _pad_tensor(tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
            if tensor.size(0) == max_seq_len:
                return tensor
            pad_size = max_seq_len - tensor.size(0)
            pad_tensor = tensor.new_full((pad_size,), pad_value)
            return torch.cat([tensor, pad_tensor], dim=0)

        input_ids_list = [_pad_tensor(ids, pad_token_id) for ids in input_ids_list]
        attention_mask_list = [_pad_tensor(mask, 0) for mask in attention_mask_list]
        labels_list = [_pad_tensor(lbl, IGNORE_INDEX) for lbl in labels_list]

        batch_pixel_values = torch.cat(pixel_values_list, dim=0)
        batch_image_flags = torch.cat(image_flags_list, dim=0)

        return {
            "pixel_values": batch_pixel_values,
            "input_ids": torch.stack(input_ids_list),
            "labels": torch.stack(labels_list),
            "attention_mask": torch.stack(attention_mask_list),
            "image_flags": batch_image_flags,
            "num_patches_list": num_patches_list_batch,
            "short_answer": short_answers,
            "question": questions,
            "answer": answers,
            "num_frames": num_frames_batch,
            "pixel_values_samples": pixel_values_list,
        }


def _extract_question_id(sample: Dict[str, Any]) -> Any:
    for key in ("qa_id", "question_id", "id"):
        if key in sample:
            return sample.get(key)
    return None


class VideoQAEvalCollator:
    """Pads frame tensors so generation can see per-sample clips."""

    def __init__(self, max_frames: Optional[int] = None) -> None:
        self.max_frames = max_frames if isinstance(max_frames, int) and max_frames > 0 else None

    def __call__(self, batch):
        if not batch:
            raise ValueError("Empty batch received in VideoQAEvalCollator.")

        num_frames_list = []
        trimmed_samples = []
        for sample in batch:
            frames = sample["pixel_values"]
            num_frames = int(sample.get("num_frames") or frames.size(0))
            if self.max_frames is not None:
                num_frames = min(num_frames, self.max_frames)
                frames = frames[:num_frames]
            trimmed = dict(sample)
            trimmed["pixel_values"] = frames
            trimmed["num_frames"] = num_frames
            trimmed_samples.append(trimmed)
            num_frames_list.append(num_frames)
        max_frames = max(num_frames_list)

        images = []
        num_patches = []
        questions = []
        answers = []
        shorts = []
        qa_ids = []

        for sample in trimmed_samples:
            frames = sample["pixel_values"]
            num_frames = int(sample.get("num_frames") or frames.size(0))
            if num_frames < max_frames:
                pad = torch.zeros(
                    max_frames - num_frames,
                    *frames.shape[1:],
                    dtype=frames.dtype,
                    device=frames.device,
                )
                frames = torch.cat([frames, pad], dim=0)

            images.append(frames)
            patches = sample.get("num_patches_list") or [1] * num_frames
            if len(patches) != num_frames:
                patches = [1] * num_frames
            num_patches.append(patches)
            questions.append(sample["question"])
            answers.append(sample["answer"])
            shorts.append(sample.get("short_answer") or "")
            qa_ids.append(_extract_question_id(sample))

        return {
            "image": torch.stack(images, dim=0),
            "num_patches_list": num_patches,
            "num_frames": num_frames_list,
            "question": questions,
            "answer": answers,
            "short_answer": shorts,
            "qa_id": qa_ids,
        }


def generate_model_response(
    model,
    processor,
    device: torch.device,
    video_tensor: torch.Tensor,
    question: str,
    max_new_tokens: int = 64,
    system_message: str = SYSTEM_PROMPT,
    num_patches_list: Optional[List[int]] = None,
) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer is None:
        raise ValueError("Tokenizer is required for text generation.")

    num_frames = int(video_tensor.size(0)) if isinstance(video_tensor, torch.Tensor) else 1

    num_image_token = getattr(model, "num_image_token", None)
    if num_image_token is None:
        raise ValueError("Model must have num_image_token attribute.")

    frame_lines = [f"Frame{i+1}: <image>" for i in range(num_frames)]
    user_part = "\n".join(frame_lines) + "\n" + question
    if system_message and system_message.strip():
        input_text = system_message + "\nUser: " + user_part + "\nAssistant:"
    else:
        input_text = "User: " + user_part + "\nAssistant:"

    if num_patches_list is None:
        num_patches_list = [1] * num_frames

    expanded_text = input_text
    for patches_for_this_frame in num_patches_list:
        tokens_for_frame = num_image_token * max(1, patches_for_this_frame)
        block = "<img>" + "<IMG_CONTEXT>" * tokens_for_frame + "</img>"
        expanded_text = expanded_text.replace("<image>", block, 1)

    tokenizer.padding_side = "left"
    enc = tokenizer(expanded_text, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    if getattr(model, "img_context_token_id", None) is None:
        set_model_img_context_token_id(model, tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))

    vision_dtype = next(model.vision_model.parameters()).dtype
    pixel_values = video_tensor.to(device=device, dtype=vision_dtype)

    outputs = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def iter_real_loader(dataloader, desc: str = "Eval-REAL"):
    total, log_every = _progress_meta(dataloader)
    for step, batch in enumerate(dataloader, 1):
        if _should_log(step, log_every, total):
            _log_progress(desc, step, total)
        images = batch["image"]
        num_patches_list = batch.get("num_patches_list", [[] for _ in range(len(images))])
        num_frames_list = batch.get("num_frames")
        questions = batch["question"]
        answers = batch["answer"]
        shorts = batch.get("short_answer", [""] * len(questions))
        question_ids = batch.get("qa_id")
        bsz = images.size(0)
        for i in range(bsz):
            if isinstance(num_frames_list, list) and i < len(num_frames_list):
                num_frames = int(num_frames_list[i])
            elif num_patches_list and len(num_patches_list[i]) > 0:
                num_frames = len(num_patches_list[i])
            else:
                num_frames = images.size(1)
            video_tensor = images[i, :num_frames]
            patches_for_item = (
                num_patches_list[i] if num_patches_list and len(num_patches_list[i]) > 0 else None
            )
            qid = None
            if isinstance(question_ids, (list, tuple)):
                if i < len(question_ids):
                    qid = question_ids[i]
            elif question_ids is not None:
                qid = question_ids
            yield video_tensor, questions[i], answers[i], shorts[i], patches_for_item, qid


def iter_endovis_loader(dataloader):
    total, log_every = _progress_meta(dataloader)
    for step, batch in enumerate(dataloader, 1):
        if _should_log(step, log_every, total):
            _log_progress("Eval-EndoVis", step, total)
        for item in batch:
            if isinstance(item, (list, tuple)):
                if len(item) == 4:
                    video_tensor, q, a, cls = item
                elif len(item) == 3:
                    video_tensor, q, a = item
                    cls = ""
                else:
                    raise ValueError(f"Unexpected EndoVis item structure length {len(item)}")
                yield video_tensor, q, a, cls
            else:
                raise ValueError(f"Unexpected EndoVis item type: {type(item)}")


def evaluate_items_generator(
    model,
    processor,
    device,
    items_iter,
    max_new_tokens=64,
    max_samples=None,
    system_message: str = SYSTEM_PROMPT,
):
    model.eval()
    gts, preds, shorts, questions, question_ids = [], [], [], [], []
    seen = 0
    with torch.no_grad():
        for item in items_iter:
            qid = None
            if isinstance(item, (list, tuple)) and len(item) == 6:
                video_tensor, q, a, short, patches, qid = item
            elif isinstance(item, (list, tuple)) and len(item) == 5:
                video_tensor, q, a, short, patches = item
            elif isinstance(item, (list, tuple)) and len(item) == 4:
                video_tensor, q, a, short = item
                patches = None
            else:
                raise ValueError(f"Unexpected item structure: {item}")

            if max_samples is not None and seen >= max_samples:
                break
            resp = generate_model_response(
                model=model,
                processor=processor,
                device=device,
                video_tensor=video_tensor,
                question=q,
                max_new_tokens=max_new_tokens,
                system_message=system_message,
                num_patches_list=patches,
            )
            gts.append(a)
            preds.append(resp)
            shorts.append(short)
            questions.append(q)
            question_ids.append(qid)
            seen += 1
    return gts, preds, shorts, questions, question_ids


def is_keyword_present(gen_ans: str, keyword: str) -> bool:
    gen_ans = (gen_ans or "").lower().strip()
    keyword = (keyword or "").lower().strip()

    if not gen_ans or not keyword:
        return False

    if " " in keyword:
        pattern = re.escape(keyword)
    else:
        pattern = r"\b" + re.escape(keyword) + r"\b"

    return bool(re.search(pattern, gen_ans))


def compute_endovis_classification_accuracy(labels: List[Any], preds: List[str]) -> Optional[Dict[str, float]]:
    pairs = []
    for lab, pred in zip(labels, preds):
        keywords = extract_label_keywords(lab)
        if not keywords:
            continue
        pairs.append((keywords, pred or ""))
    if not pairs:
        return None
    total = len(pairs)
    correct = 0
    for keywords, pred in pairs:
        if all(is_keyword_present(pred, kw) for kw in keywords):
            correct += 1
    skipped = len(labels) - total
    return {"acc": correct / total, "correct": correct, "total": total, "skipped": skipped}


def _compute_short_contains_acc(shorts: List[Any], preds: List[str]) -> Optional[Dict[str, float]]:
    if shorts is None or preds is None:
        return None

    pairs = []
    for s, p in zip(shorts, preds):
        keywords = extract_label_keywords(s)
        if not keywords:
            continue
        pairs.append((keywords, p))

    if not pairs:
        return None
    total = len(pairs)
    correct = 0
    for keywords, pred in pairs:
        if all(is_keyword_present(pred, kw) for kw in keywords):
            correct += 1
    skipped = len(shorts) - total
    return {"acc": correct / total, "correct": correct, "total": total, "skipped": skipped}


def eval_and_report(
    tag: str,
    gts: List[str],
    preds: List[str],
    save_root: Path,
    suffix: str,
    shorts: Optional[List[str]] = None,
):
    if len(gts) != len(preds):
        raise ValueError(f"Mismatch between gts({len(gts)}) and preds({len(preds)})")
    bleu_score, rouge_score, meteor_score = get_nlp_metrics(gts, preds)
    short_acc = _compute_short_contains_acc(shorts, preds) if shorts is not None else None
    result = {
        "tag": tag,
        "bleu": bleu_score,
        "rouge": rouge_score,
        "meteor": meteor_score,
        "short_contains_acc": short_acc,
        "n": len(gts),
    }
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    with open(save_root / f"metrics_{suffix}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{tag}] BLEU: {bleu_score}")
    print(f"[{tag}] ROUGE: {rouge_score}")
    print(f"[{tag}] METEOR: {meteor_score}")
    if short_acc:
        print(
            f"[{tag}] Short answer accuracy: {short_acc['acc']:.4f} "
            f"(correct={short_acc['correct']}/{short_acc['total']}, skipped={short_acc['skipped']})"
        )
    return result


def log_metrics_to_wandb(wandb_module, metrics: Dict[str, Any], prefix: str, step: Optional[int]):
    if wandb_module is None or metrics is None:
        return
    payload = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                payload[f"{prefix}/{k}/{subk}"] = subv
        else:
            payload[f"{prefix}/{k}"] = v
    wandb_module.log(payload, step=step)


def _normalize_question_id(qid: Any, fallback_idx: int) -> Any:
    if qid is None or qid == "":
        return fallback_idx
    if isinstance(qid, torch.Tensor):
        if qid.numel() == 1:
            return int(qid.item())
        return qid.tolist()
    if isinstance(qid, np.generic):
        return qid.item()
    return qid


def save_eval_predictions(
    save_root: Path,
    suffix: str,
    question_ids: Optional[List[Any]],
    questions: List[str],
    gts: List[str],
    preds: List[str],
) -> Path:
    if len(questions) != len(gts) or len(questions) != len(preds):
        raise ValueError(
            f"Mismatch in evaluation outputs (questions={len(questions)}, gts={len(gts)}, preds={len(preds)})"
        )
    if question_ids is not None and len(question_ids) != len(questions):
        raise ValueError(
            f"Mismatch in question_ids length ({len(question_ids)}) vs questions ({len(questions)})"
        )
    records = []
    for idx, (question, gt, pred) in enumerate(zip(questions, gts, preds)):
        qid = question_ids[idx] if question_ids is not None else None
        records.append(
            {
                "question_idx": _normalize_question_id(qid, idx),
                "question": question,
                "gt": gt,
                "pred": pred,
            }
        )
    save_root = Path(save_root) / "metrics"
    save_root.mkdir(parents=True, exist_ok=True)
    save_path = save_root / f"predictions_{suffix}.json"
    with open(save_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    return save_path


def load_trained_weights(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, nn.Module):
        model.load_state_dict(checkpoint.state_dict(), strict=True)
        loaded_model = model
        print(f"Loaded full model state from {checkpoint_path}")
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        model.load_state_dict(checkpoint, strict=True)
        loaded_model = model
        print(f"Loaded state_dict from {checkpoint_path}")
    else:
        loaded_model = checkpoint
        print(f"Loaded full model from {checkpoint_path}")
    return loaded_model


def save_model_state(model: nn.Module, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Save state_dict to avoid pickling issues with dynamically loaded HF modules.
    torch.save(model.state_dict(), target_path)
    print(f"Saved model state_dict to {target_path}")


def save_run_settings(run_dir: Path, settings: Dict[str, Any], filename: str = "settings.json") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return str(obj)

    payload = {k: _convert(v) for k, v in settings.items()}
    with open(run_dir / filename, "w") as fp:
        json.dump(payload, fp, indent=2)


def update_run_settings_with_trainable(
    run_dir: Path,
    total_params: int,
    trainable_params: int,
    filename: str = "settings.json",
) -> None:
    path = run_dir / filename
    if path.exists():
        with open(path, "r") as fp:
            payload = json.load(fp)
    else:
        payload = {}
    payload["total_params"] = int(total_params)
    payload["trainable_params"] = int(trainable_params)
    payload["trainable_percent"] = (
        trainable_params / total_params * 100.0 if total_params else 0.0
    )
    with open(path, "w") as fp:
        json.dump(payload, fp, indent=2)


def validate_intern(model, dataloader, device, vision_dtype):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            if not batch:
                continue
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(
                device, dtype=vision_dtype, non_blocking=True
            )
            image_flags = batch["image_flags"].to(device, non_blocking=True)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_flags=image_flags,
            )
            loss = outputs.loss
            total_loss += float(loss.detach().cpu())
            steps += 1
    return total_loss / max(1, steps)


def run_training_loop(
    args,
    model,
    processor,
    train_loader,
    device,
    optimizer,
    vision_dtype,
    tokenizer,
    run_dir: Path,
    valid_loader=None,
    wandb_module=None,
):
    use_cuda = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if vision_dtype == torch.bfloat16 else torch.float16
    use_autocast = use_cuda
    use_fp16_scaler = use_cuda and autocast_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_scaler)
    best_monitor: Optional[float] = None
    best_epoch_num: Optional[int] = None
    best_model_path = run_dir / "best_model.pt"
    set_model_img_context_token_id(model, tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))
    grad_accum_steps = max(1, args.grad_accum)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_log_losses: List[float] = []
        total_batches, log_every = _progress_meta(train_loader)
        optimizer.zero_grad(set_to_none=True)

        old_adapter_param = None
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(
                device, dtype=vision_dtype, non_blocking=True
            )
            image_flags = batch["image_flags"].to(device, non_blocking=True)

            if args.debug_shapes and step == 0 and epoch == 0:
                print("\n[DEBUG SHAPES] First training batch")
                print(f"  input_ids shape:      {tuple(input_ids.shape)}")
                print(f"  attention_mask shape: {tuple(attention_mask.shape)}")
                print(f"  labels shape:         {tuple(labels.shape)}")
                print(f"  pixel_values shape:   {tuple(pixel_values.shape)}")
                print(f"  image_flags shape:    {tuple(image_flags.shape)}")

            if getattr(model, "_force_vision_input_grads", False) and not pixel_values.requires_grad:
                pixel_values = pixel_values.detach().requires_grad_(True)


            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_flags=image_flags,
            )
            loss = outputs.loss


            loss_value = float(loss.detach().cpu())
            running_loss += loss_value
            running_log_losses.append(loss_value)
            avg_disp = float(
                np.mean(running_log_losses[-max(1, min(50, len(running_log_losses))) :])
            )

            loss_to_backward = loss / grad_accum_steps
            if use_fp16_scaler:
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            if args.debug and epoch == 0:
                adapter_param = None
                adapter_name = None
                for name, param in model.named_parameters():
                    if any(key in name for key in ("stlora_module", "timeshift_module", "stadapter_module")) and param.requires_grad:
                        adapter_param = param
                        adapter_name = name
                        break

                sample_param = adapter_param
                sample_name = adapter_name
                label = "vision adapter"

                if sample_param is None:
                    for name, param in model.named_parameters():
                        if "mlp1" in name and "lora_" in name and param.requires_grad:
                            sample_param = param
                            sample_name = name
                            label = "projector LoRA"
                            break

                if sample_param is None:
                    for name, param in model.named_parameters():
                        if "vision_model" in name and param.requires_grad:
                            sample_param = param
                            sample_name = name
                            label = "vision_model"
                            break

                if sample_param is not None:
                    adapter_name = getattr(model, "_vision_adapter_name", None)
                    adapter_name = adapter_name or getattr(
                        getattr(model, "stlora_module", None), "_adapter_name", None
                    )
                    if adapter_name:
                        label = f"{label} ({adapter_name})"
                    if sample_param.grad is None:
                        print(
                            "\n[DEBUG AUTOGRAD] No gradient computed for "
                            f"{label} parameter: {sample_name}"
                        )
                        if old_adapter_param is None:
                            old_adapter_param = sample_param.detach().clone()
                        abs_diff = (sample_param.detach() - old_adapter_param).abs().max().item()
                        print(f"Maximum absolute difference in gradients: {abs_diff:.4e}")
                    else:
                        print(
                            "\n[DEBUG AUTOGRAD] Gradient computed for "
                            f"{label} parameter: {sample_name}"
                        )
                        if old_adapter_param is None:
                            old_adapter_param = sample_param.detach().clone()
                        abs_diff = (sample_param.detach() - old_adapter_param).abs().max().item()
                        print(f"Maximum absolute difference in gradients: {abs_diff:.4e}")
                else:
                    print(
                        "\n[DEBUG AUTOGRAD] No trainable parameter found "
                        "inside vision adapters, mlp1 LoRA, or vision_model."
                    )

            should_step = (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader)
            if should_step:
                if use_fp16_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            display_step = step + 1
            if _should_log(display_step, log_every, total_batches):
                total_display = total_batches if total_batches is not None else "?"
                print(
                    f"[Train] Epoch {epoch + 1}/{args.epochs} "
                    f"Step {display_step}/{total_display} - loss={avg_disp:.2f}"
                )

            batch_size = len(batch.get("question", []))
            if batch_size > 0 and step % 100 == 0 and args.preview:
                sample_question = batch["question"][0]
                sample_answer = batch["answer"][0]
                sample_frames_list = batch.get("pixel_values_samples", [])
                sample_frames = sample_frames_list[0] if sample_frames_list else None
                train_prompt = getattr(args, "_train_system_prompt", "")

                sample_pred = ""
                if sample_frames is not None:
                    model.eval()
                    with torch.no_grad():
                        sample_pred = generate_model_response(
                            model=model,
                            processor=processor,
                            device=device,
                            video_tensor=sample_frames,
                            question=sample_question,
                            max_new_tokens=min(args.max_new_tokens_eval, 64),
                            system_message=train_prompt,
                        )
                    model.train()
                print(
                    f"\n[Preview] Epoch {epoch+1} Step {step}\n"
                    f"Q: {sample_question}\nGT: {sample_answer.strip()}\nPD: {sample_pred}\n"
                )
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6e}")
                torch.cuda.empty_cache()

        avg_train_loss = running_loss / max(1, len(train_loader))
        val_loss = None
        if valid_loader is not None:
            val_loss = validate_intern(model, valid_loader, device, vision_dtype)
            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"- train loss: {avg_train_loss:.4f}, val loss: {val_loss:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"- train loss: {avg_train_loss:.4f}"
            )

        if wandb_module is not None:
            log_payload = {"train/loss": avg_train_loss}
            if val_loss is not None:
                log_payload["val/loss"] = val_loss
            wandb_module.log(log_payload, step=epoch + 1)

        monitor_value = val_loss if val_loss is not None else avg_train_loss
        if best_monitor is None or monitor_value < best_monitor:
            best_monitor = monitor_value
            best_epoch_num = epoch + 1
            save_model_state(model, best_model_path)
            with open(run_dir / "best_epoch.txt", "w") as f:
                f.write(f"{best_epoch_num}\n")

    return best_monitor if best_monitor is not None else avg_train_loss


def do_evaluations(
    model,
    processor,
    device,
    args,
    run_dir: Path,
    real_eval_loader=None,
    real_ood_eval_loader=None,
    endovis_eval_loader=None,
    endovis_ood_eval_loader=None,
    wandb_module=None,
    wandb_step: Optional[int] = None,
):
    if real_eval_loader is not None and args.eval_real:
        gts_real, preds_real, shorts_real, questions_real, qids_real = evaluate_items_generator(
            model,
            processor,
            device,
            items_iter=iter_real_loader(real_eval_loader, desc="Eval-REAL"),
            max_new_tokens=args.max_new_tokens_eval,
            max_samples=args.max_samples_eval,
            system_message=SYSTEM_PROMPT,
        )
        metrics_real = eval_and_report(
            "REAL Eval (test only)", gts_real, preds_real, run_dir, "real_test", shorts=shorts_real
        )
        save_eval_predictions(run_dir, "real_test", qids_real, questions_real, gts_real, preds_real)
        log_metrics_to_wandb(wandb_module, metrics_real, "eval/real", wandb_step)
    elif args.eval_real:
        print("[Eval] REAL evaluation skipped (loader unavailable).")

    if real_ood_eval_loader is not None and args.eval_real_ood:
        gts_ood, preds_ood, shorts_ood, questions_ood, qids_ood = evaluate_items_generator(
            model,
            processor,
            device,
            items_iter=iter_real_loader(real_ood_eval_loader, desc="Eval-REAL-OOD"),
            max_new_tokens=args.max_new_tokens_eval,
            max_samples=args.max_samples_eval,
            system_message=SYSTEM_PROMPT,
        )
        metrics_ood = eval_and_report(
            "REAL OOD Eval (test only)",
            gts_ood,
            preds_ood,
            run_dir,
            "real_ood_test",
            shorts=shorts_ood,
        )
        save_eval_predictions(run_dir, "real_ood_test", qids_ood, questions_ood, gts_ood, preds_ood)
        log_metrics_to_wandb(wandb_module, metrics_ood, "eval/real_ood", wandb_step)
    elif args.eval_real_ood:
        print("[Eval] REAL-OOD evaluation skipped (loader unavailable).")

    if endovis_eval_loader is not None and args.eval_endovis:
        (
            gts_endovis,
            preds_endovis,
            labels_endovis,
            questions_endovis,
            qids_endovis,
        ) = evaluate_items_generator(
            model,
            processor,
            device,
            items_iter=iter_endovis_loader(endovis_eval_loader),
            max_new_tokens=args.max_new_tokens_eval,
            max_samples=args.max_samples_eval,
            system_message=SYSTEM_PROMPT_ENDOVIS,
        )

        metrics_endovis = eval_and_report(
            "EndoVis Eval",
            gts_endovis,
            preds_endovis,
            run_dir,
            "endovis",
            shorts=labels_endovis,
        )
        save_eval_predictions(
            run_dir, "endovis", qids_endovis, questions_endovis, gts_endovis, preds_endovis
        )

        cls_acc = compute_endovis_classification_accuracy(labels_endovis, preds_endovis)
        if cls_acc is not None:
            metrics_endovis["classification_accuracy"] = cls_acc
            print(
                f"EndoVis classification acc: {cls_acc['acc']:.4f} "
                f"(correct={cls_acc['correct']}/{cls_acc['total']}, skipped={cls_acc['skipped']})"
            )
        metrics_path = run_dir / "metrics" / "metrics_endovis.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics_endovis, f, indent=2)
        log_metrics_to_wandb(wandb_module, metrics_endovis, "eval/endovis", wandb_step)
    elif args.eval_endovis:
        print("[Eval] EndoVis evaluation skipped (loader unavailable).")

    if endovis_ood_eval_loader is not None and getattr(args, "eval_endovis_ood", False):
        (
            gts_endovis_ood,
            preds_endovis_ood,
            labels_endovis_ood,
            questions_endovis_ood,
            qids_endovis_ood,
        ) = evaluate_items_generator(
            model,
            processor,
            device,
            items_iter=iter_endovis_loader(endovis_ood_eval_loader),
            max_new_tokens=args.max_new_tokens_eval,
            max_samples=args.max_samples_eval,
            system_message=SYSTEM_PROMPT_ENDOVIS,
        )

        metrics_endovis_ood = eval_and_report(
            "EndoVis OOT Eval",
            gts_endovis_ood,
            preds_endovis_ood,
            run_dir,
            "endovis_ood",
            shorts=labels_endovis_ood,
        )
        save_eval_predictions(
            run_dir,
            "endovis_ood",
            qids_endovis_ood,
            questions_endovis_ood,
            gts_endovis_ood,
            preds_endovis_ood,
        )

        cls_acc_ood = compute_endovis_classification_accuracy(labels_endovis_ood, preds_endovis_ood)
        if cls_acc_ood is not None:
            metrics_endovis_ood["classification_accuracy"] = cls_acc_ood
            print(
                f"EndoVis OOT classification acc: {cls_acc_ood['acc']:.4f} "
                f"(correct={cls_acc_ood['correct']}/{cls_acc_ood['total']}, skipped={cls_acc_ood['skipped']})"
            )
        metrics_path_ood = run_dir / "metrics" / "metrics_endovis_ood.json"
        metrics_path_ood.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path_ood, "w") as f:
            json.dump(metrics_endovis_ood, f, indent=2)
        log_metrics_to_wandb(wandb_module, metrics_endovis_ood, "eval/endovis_ood", wandb_step)
    elif getattr(args, "eval_endovis_ood", False):
        print("[Eval] EndoVis OOT evaluation skipped (loader unavailable).")


# ---------------------- Qwen3 utilities ---------------------- #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
QWEN_VISION_TENSOR_KEYS = {"pixel_values", "pixel_values_videos"}


def _tensor_to_pil_frames(video_tensor: torch.Tensor) -> List[Any]:
    if video_tensor is None:
        return []
    frames: List[Any] = []
    mean = torch.tensor(IMAGENET_MEAN, dtype=video_tensor.dtype, device=video_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=video_tensor.dtype, device=video_tensor.device).view(3, 1, 1)
    if video_tensor.dim() == 3:
        video_tensor = video_tensor.unsqueeze(0)
    for frame in video_tensor:
        img = frame.detach().cpu() * std.cpu() + mean.cpu()
        img = img.clamp(0, 1)
        frames.append(to_pil_image(img))
    return frames


def _prepare_qwen3_vision_inputs(
    processor,
    conversations: List[List[Dict[str, Any]]],
):
    from qwen_vl_utils import process_vision_info  # type: ignore

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        conversations,
        return_video_kwargs=True,
    )
    if video_inputs is not None:
        video_kwargs["fps"] = 1.0
    return image_inputs, video_inputs, video_kwargs


def _cleanup_temp_dir(path: Optional[str]) -> None:
    if not path:
        return
    shutil.rmtree(path, ignore_errors=True)


def _compute_frame_indices(total: int, limit: Optional[int]) -> Optional[List[int]]:
    if limit is None or limit <= 0 or total <= limit:
        return None
    if limit == 1:
        return [total // 2]
    step = (total - 1) / float(limit - 1)
    indices: List[int] = []
    last = -1
    for i in range(limit):
        idx = int(round(i * step))
        idx = max(0, min(total - 1, idx))
        if idx <= last:
            idx = min(total - 1, last + 1)
        indices.append(idx)
        last = idx
    return indices


def _select_frame_paths(frame_paths: List[Any], limit: Optional[int]) -> List[Any]:
    if not frame_paths:
        return []
    if limit is None or limit <= 0 or len(frame_paths) <= limit:
        return [str(p) if isinstance(p, Path) else p for p in frame_paths]
    trimmed = frame_paths[:limit]
    return [str(p) if isinstance(p, Path) else p for p in trimmed]


def build_qwen3_conversation(
    question: str,
    frames: List[Any],
    fps: float,
    system_prompt: str,
    use_video_payload: bool = True,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    user_payload: List[Dict[str, Any]] = []
    temp_dir: Optional[str] = None

    if frames and use_video_payload:
        frames = [str(f) if isinstance(f, Path) else f for f in frames]
        user_payload.append(
            {
                "type": "video",
                "video": frames,
                "sample_fps": fps,
            }
        )
    user_payload.append({"type": "text", "text": question})

    messages: List[Dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )
    messages.append({"role": "user", "content": user_payload})
    return messages, temp_dir


class Qwen3VLBatchCollator:
    def __init__(
        self,
        processor,
        video_fps: float = 1.0,
        max_frames: Optional[int] = None,
        frame_pixel_size: int = 448,
        system_prompt: str = "",
    ) -> None:
        self.processor = processor
        self.video_fps = video_fps
        self.max_frames = max_frames if isinstance(max_frames, int) and max_frames > 0 else None
        self.frame_pixel_size = max(1, frame_pixel_size)
        self.system_prompt = system_prompt
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token.")
        self.pad_token_id = pad_token_id

    def _ensure_item_image(self, item: Dict[str, Any]) -> None:
        if "image" in item and isinstance(item["image"], torch.Tensor):
            return
        pixel_values = item.get("pixel_values")
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError("Qwen batch item is missing pixel_values/image tensors.")
        item["image"] = pixel_values

    def _trim_item_frames(self, item: Dict[str, Any]) -> None:
        tensor = item.get("image")
        if not isinstance(tensor, torch.Tensor):
            return
        total = tensor.size(0)
        if self.max_frames is not None and total > self.max_frames:
            item["image"] = tensor[: self.max_frames]
            frame_paths = item.get("frame_paths")
            if isinstance(frame_paths, list) and frame_paths:
                item["frame_paths"] = frame_paths[: self.max_frames]
        item["num_frames"] = int(item["image"].size(0))

    def _load_frames_for_prompt(self, item: Dict[str, Any]) -> List[Any]:
        frame_paths = item.get("frame_paths")
        if isinstance(frame_paths, list) and frame_paths:
            if self.max_frames is not None and self.max_frames > 0:
                return [str(p) if isinstance(p, Path) else p for p in frame_paths[: self.max_frames]]
            return [str(p) if isinstance(p, Path) else p for p in frame_paths]
        images = item.get("image")
        if isinstance(images, torch.Tensor):
            return _tensor_to_pil_frames(images)
        raise ValueError("Missing frame paths and image tensors for Qwen conversation building.")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}

        for item in batch:
            self._ensure_item_image(item)
            self._trim_item_frames(item)

        max_frames = max(item["image"].size(0) for item in batch)
        bsz = len(batch)
        c, h, w = batch[0]["image"].size(1), batch[0]["image"].size(2), batch[0]["image"].size(3)
        images = torch.zeros(bsz, max_frames, c, h, w, dtype=batch[0]["image"].dtype)
        frame_mask = torch.zeros(bsz, max_frames, dtype=torch.bool)
        for i, item in enumerate(batch):
            t_i = item["image"].size(0)
            images[i, :t_i] = item["image"]
            frame_mask[i, :t_i] = True

        conversations: List[List[Dict[str, Any]]] = []
        texts: List[str] = []
        answer_ids_list: List[torch.Tensor] = []
        temp_dirs: List[Optional[str]] = []
        for item in batch:
            frames = self._load_frames_for_prompt(item)
            messages, temp_dir = build_qwen3_conversation(
                question=item["question"],
                frames=frames,
                fps=self.video_fps,
                system_prompt=self.system_prompt,
                use_video_payload=True,
            )
            temp_dirs.append(temp_dir)
            conversations.append(messages)
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            answer_ids = self.processor.tokenizer(
                item["answer"].strip(),
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            answer_ids_list.append(answer_ids)

        image_inputs, video_inputs, video_kwargs = _prepare_qwen3_vision_inputs(
            self.processor,
            conversations,
        )
        processor_extra_kwargs: Dict[str, Any] = dict(video_kwargs or {})

        processor_kwargs: Dict[str, Any] = dict(processor_extra_kwargs)
        processor_kwargs.update(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        processor_kwargs["truncation"] = False

        processor_inputs = self.processor(**processor_kwargs)
        prompt_ids = processor_inputs.pop("input_ids")
        prompt_attention = processor_inputs.pop("attention_mask")

        merged_ids: List[torch.Tensor] = []
        merged_masks: List[torch.Tensor] = []
        merged_labels: List[torch.Tensor] = []
        for idx in range(len(batch)):
            prompt_seq = prompt_ids[idx]
            prompt_mask = prompt_attention[idx]
            ans_ids = answer_ids_list[idx]
            ans_mask = torch.ones_like(ans_ids)

            input_ids = torch.cat([prompt_seq, ans_ids], dim=0)
            attention = torch.cat([prompt_mask, ans_mask], dim=0)
            labels = torch.cat(
                [torch.full_like(prompt_seq, IGNORE_INDEX), ans_ids.clone()],
                dim=0,
            )

            merged_ids.append(input_ids)
            merged_masks.append(attention)
            merged_labels.append(labels)

        padded_input_ids = pad_sequence(merged_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_attention = pad_sequence(merged_masks, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(merged_labels, batch_first=True, padding_value=IGNORE_INDEX)

        model_inputs = dict(processor_inputs)
        model_inputs["input_ids"] = padded_input_ids
        model_inputs["attention_mask"] = padded_attention
        model_inputs["labels"] = padded_labels

        for temp_dir in temp_dirs:
            _cleanup_temp_dir(temp_dir)

        return {
            "image": images,
            "frame_mask": frame_mask,
            "question": [b["question"] for b in batch],
            "answer": [b["answer"] for b in batch],
            "short_answer": [b.get("short_answer", "") for b in batch],
            "num_frames": [b["num_frames"] for b in batch],
            "video_id": [b.get("video_id") for b in batch],
            "id": [b.get("id") for b in batch],
            "model_inputs": model_inputs,
            "frame_paths": [b.get("frame_paths", []) for b in batch],
        }


class Qwen3VideoEvalCollator:
    def __init__(self, max_frames: Optional[int] = None) -> None:
        self.max_frames = max_frames if isinstance(max_frames, int) and max_frames > 0 else None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Empty batch received.")

        trimmed: List[Tuple[torch.Tensor, int, Dict[str, Any]]] = []
        max_frames = 0
        for sample in batch:
            frames = sample.get("image", sample.get("pixel_values"))
            if not isinstance(frames, torch.Tensor):
                raise ValueError("Qwen eval sample missing image tensors.")
            num_frames = sample.get("num_frames", frames.size(0))
            if self.max_frames is not None and self.max_frames > 0:
                num_frames = min(num_frames, self.max_frames)
                frames = frames[:num_frames]
            sample["num_frames"] = num_frames
            frame_paths = sample.get("frame_paths")
            if isinstance(frame_paths, list) and frame_paths:
                sample["frame_paths"] = frame_paths[:num_frames]
            trimmed.append((frames, num_frames, sample))
            max_frames = max(max_frames, num_frames)

        images = []
        num_patches = []
        questions = []
        answers = []
        shorts = []
        qa_ids = []
        for frames, num_frames, sample in trimmed:
            if num_frames < max_frames:
                pad = torch.zeros(
                    max_frames - num_frames,
                    *frames.shape[1:],
                    dtype=frames.dtype,
                    device=frames.device,
                )
                frames = torch.cat([frames, pad], dim=0)
            images.append(frames)
            num_patches.append([1] * num_frames)
            questions.append(sample["question"])
            answers.append(sample["answer"])
            shorts.append(sample.get("short_answer") or "")
            qa_ids.append(_extract_question_id(sample))
        return {
            "image": torch.stack(images, dim=0),
            "num_patches_list": num_patches,
            "question": questions,
            "answer": answers,
            "short_answer": shorts,
            "frame_paths": [sample.get("frame_paths", []) for _, _, sample in trimmed],
            "qa_id": qa_ids,
        }


def tokenize_short_answer(processor, short_answer: str, lower: bool = False) -> List[int]:
    if not isinstance(short_answer, str) or short_answer.strip() == "":
        raise ValueError("short_answer must be a non-empty string")
    s = short_answer.strip().lower() if lower else short_answer.strip()
    if s.lower() in ["yes", "no"]:
        s += ","
    ids = processor.tokenizer(
        s,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    seen: set[int] = set()
    unique: List[int] = []
    for tok in ids.tolist():
        if tok not in seen:
            seen.add(tok)
            unique.append(tok)
    return unique


def robust_target_token_ids(processor, s: str, lower: bool = False) -> List[int]:
    if not isinstance(s, str) or not s.strip():
        return []
    variants = set()
    base = s.strip()
    variants.add(base)
    variants.add(base.rstrip(".").rstrip(","))
    variants.add(base.lower())
    variants.add(base.lower().rstrip(".").rstrip(","))
    for v in list(variants):
        variants.add(" " + v)

    ids: set[int] = set()
    tokenizer = processor.tokenizer
    for v in variants:
        tids = tokenizer(v, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        if len(tids) == 1:
            ids.add(tids[0])
        else:
            ids.update(tids)
            ids.add(tids[-1])
    return list(ids)


def smooth_presence_and_repeat_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_token_ids: List[int],
    temperature: float = 1.0,
    eps: float = 1e-6,
    rep_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not target_token_ids:
        raise ValueError("target_token_ids is empty")

    l_log = logits.size(1)
    l_lab = labels.size(1)
    if l_log < l_lab:
        pad_size = l_lab - l_log
        pad_tensor = logits.new_zeros((logits.size(0), pad_size, logits.size(2)))
        logits = torch.cat([logits, pad_tensor], dim=1)
    elif l_log > l_lab:
        pad_size = l_log - l_lab
        pad_tensor = labels.new_full((labels.size(0), pad_size), -100)
        labels = torch.cat([labels, pad_tensor], dim=1)

    mask = labels != -100
    if mask.sum() == 0:
        raise ValueError("No supervised tokens in labels")

    logprobs = torch.log_softmax(logits / max(temperature, 1e-6), dim=-1)
    probs = torch.exp(logprobs)
    supervised = mask.squeeze(0)

    presence_terms, repeat_terms = [], []
    for tok in target_token_ids:
        token_probs = probs[0, :, tok]
        supervised_probs = token_probs[supervised]
        if supervised_probs.numel() > 0:
            not_present = torch.clamp(1.0 - supervised_probs, eps, 1.0)
            presence_terms.append(
                -torch.log(torch.clamp(1.0 - torch.prod(not_present), eps, 1.0))
            )

        pred_ids = probs.argmax(dim=-1).squeeze(0)
        idxs = torch.nonzero((pred_ids == tok) & supervised, as_tuple=False).flatten()
        if idxs.numel() > 0:
            first_hit = idxs[0].item()
            future_mask = torch.zeros_like(supervised)
            future_mask[first_hit + 1 :] = True
            future_mask = future_mask & supervised
            if future_mask.any():
                future_probs = token_probs[future_mask]
                repeat_terms.append(
                    -torch.log(torch.clamp(1.0 - future_probs, eps, 1.0)).mean()
                )

    presence_loss = torch.stack(presence_terms).mean() if presence_terms else logits.new_zeros(())
    repeat_loss = torch.stack(repeat_terms).mean() if repeat_terms else logits.new_zeros(())

    return presence_loss, rep_weight * repeat_loss


def iter_real_loader_qwen(dataloader, desc: str = "Eval-REAL"):
    total, log_every = _progress_meta(dataloader)
    for step, batch in enumerate(dataloader, 1):
        if _should_log(step, log_every, total):
            _log_progress(desc, step, total)
        images = batch["image"]
        num_patches_list = batch.get("num_patches_list", [[] for _ in range(len(images))])
        questions = batch["question"]
        answers = batch["answer"]
        shorts = batch.get("short_answer", [""] * len(questions))
        frame_paths_batch = batch.get("frame_paths", [[] for _ in range(len(questions))])
        question_ids = batch.get("qa_id")
        for i in range(images.size(0)):
            if num_patches_list and len(num_patches_list[i]) > 0:
                num_frames = len(num_patches_list[i])
            else:
                num_frames = images.size(1)
            video_tensor = images[i, :num_frames]
            qid = None
            if isinstance(question_ids, (list, tuple)):
                if i < len(question_ids):
                    qid = question_ids[i]
            elif question_ids is not None:
                qid = question_ids
            yield video_tensor, questions[i], answers[i], shorts[i], frame_paths_batch[i], qid


def generate_qwen3_response(
    model,
    processor,
    device: torch.device,
    video_tensor: Optional[torch.Tensor],
    frame_paths: Optional[List[str]],
    question: str,
    max_new_tokens: int = 64,
    system_message: str = SYSTEM_PROMPT,
    fps: float = 1.0,
    max_frames: Optional[int] = None,
) -> str:
    frames: List[Any] = []
    if frame_paths:
        frames = _select_frame_paths(frame_paths, max_frames)
    elif video_tensor is not None:
        frames = _tensor_to_pil_frames(video_tensor)

    messages, temp_dir = build_qwen3_conversation(
        question=question,
        frames=frames,
        fps=fps,
        system_prompt=system_message,
        use_video_payload=True,
    )

    texts = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    ]

    image_inputs, video_inputs, video_kwargs = _prepare_qwen3_vision_inputs(processor, [messages])
    processor_extra_kwargs = dict(video_kwargs or {})

    chat_inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **processor_extra_kwargs,
    )
    chat_inputs = chat_inputs.to(device)
    prompt_len = chat_inputs["input_ids"].shape[-1]

    generated = model.generate(
        **chat_inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    continuation = generated[:, prompt_len:]
    decoded = processor.batch_decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    _cleanup_temp_dir(temp_dir)

    return decoded[0].strip()


def evaluate_items_generator_qwen(
    model,
    processor,
    device,
    items_iter,
    max_new_tokens=64,
    max_samples=None,
    system_message: str = SYSTEM_PROMPT,
    video_fps: float = 1.0,
    max_frames: Optional[int] = None,
):
    model.eval()
    gts, preds, shorts, questions, question_ids = [], [], [], [], []
    seen = 0
    with torch.no_grad():
        for item in items_iter:
            frame_paths = None
            qid = None
            if isinstance(item, (list, tuple)):
                if len(item) == 6:
                    video_tensor, q, a, short, frame_paths, qid = item
                elif len(item) == 5:
                    video_tensor, q, a, short, frame_paths = item
                elif len(item) == 4:
                    video_tensor, q, a, short = item
                else:
                    video_tensor, q, a = item
                    short = ""
            else:
                raise ValueError(f"Unexpected evaluation item type: {type(item)}")

            if max_samples is not None and seen >= max_samples:
                break
            resp = generate_qwen3_response(
                model=model,
                processor=processor,
                device=device,
                video_tensor=video_tensor,
                frame_paths=frame_paths,
                question=q,
                max_new_tokens=max_new_tokens,
                system_message=system_message,
                fps=video_fps,
                max_frames=max_frames,
            )
            gts.append(a)
            preds.append(resp)
            shorts.append(short)
            questions.append(q)
            question_ids.append(qid)
            seen += 1
    return gts, preds, shorts, questions, question_ids


def move_qwen_model_inputs_to_device(
    model_inputs: Dict[str, Any],
    device: torch.device,
    vision_dtype: torch.dtype,
):
    moved: Dict[str, Any] = {}
    for key, value in model_inputs.items():
        if isinstance(value, torch.Tensor):
            if key in QWEN_VISION_TENSOR_KEYS:
                moved[key] = value.to(device=device, dtype=vision_dtype)
            else:
                moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def validate_qwen(model, dataloader, device, vision_dtype):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            if not batch:
                continue
            model_inputs = move_qwen_model_inputs_to_device(
                batch["model_inputs"], device, vision_dtype
            )
            outputs = model(**model_inputs)
            loss = outputs.loss
            total_loss += float(loss.detach().cpu())
            steps += 1
    return total_loss / max(1, steps)


def run_qwen_training_loop(
    args,
    model,
    processor,
    train_loader,
    device,
    optimizer,
    vision_dtype,
    run_dir: Path,
    valid_loader=None,
    wandb_module=None,
):
    autocast_dtype = torch.bfloat16 if vision_dtype == torch.bfloat16 else torch.float16
    use_cuda = device.type == "cuda"
    use_autocast = use_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and autocast_dtype == torch.float16)
    best_monitor: Optional[float] = None
    best_epoch_num: Optional[int] = None
    best_model_path = run_dir / "best_model.pt"
    grad_accum_steps = max(1, args.grad_accum)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_log_losses: List[float] = []
        total_batches, log_every = _progress_meta(train_loader)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            if not batch:
                continue

            model_inputs = move_qwen_model_inputs_to_device(
                batch["model_inputs"], device, vision_dtype
            )
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(**model_inputs)
                loss = outputs.loss

            loss_value = float(loss.detach().cpu())
            running_loss += loss_value
            running_log_losses.append(loss_value)
            loss_to_backward = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            if getattr(args, "debug", False) and epoch == 0 and step == 0:
                found_trainable = False
                grad_reported = False
                for name, param in model.named_parameters():
                    if "visual" in name or "vision" in name:
                        if param.requires_grad:
                            found_trainable = True
                            if param.grad is None:
                                print(
                                    f"[DEBUG] No grad for encoder param: {name} "
                                    f"shape={tuple(param.shape)}"
                                )
                            else:
                                grad_norm = float(param.grad.detach().abs().max().cpu())
                                print(
                                    f"[DEBUG] Grad connected for encoder param: {name} "
                                    f"max_abs_grad={grad_norm:.4e}"
                                )
                            grad_reported = True
                            break
                if not found_trainable:
                    print("[DEBUG] No trainable encoder params found for gradient check.")
                elif not grad_reported:
                    print("[DEBUG] Trainable encoder params found, but no grad info reported.")

            should_step = (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader)
            if should_step:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            avg_disp = float(np.mean(running_log_losses[-max(1, min(50, len(running_log_losses))):]))
            display_step = step + 1
            if _should_log(display_step, log_every, total_batches):
                total_display = total_batches if total_batches is not None else "?"
                print(
                    f"[Train] Epoch {epoch + 1}/{args.epochs} "
                    f"Step {display_step}/{total_display} - loss={avg_disp:.2f}"
                )

            batch_size = len(batch.get("question", []))
            if batch_size > 0 and step % 100 == 0 and args.preview:
                preview_frames = batch["image"][0][: batch["num_frames"][0]]
                preview_question = batch["question"][0]
                preview_answer = batch["answer"][0]
                preview_paths = batch.get("frame_paths", [[]])[0]
                train_prompt = getattr(args, "_train_system_prompt", "")
                model.eval()
                with torch.no_grad():
                    frames_limit = (
                        args.frames
                        if getattr(args, "frames", 0) and args.frames > 0
                        else args.sequence_length
                    )
                    preview_text = generate_qwen3_response(
                        model=model,
                        processor=processor,
                        device=device,
                        video_tensor=preview_frames,
                        frame_paths=preview_paths,
                        question=preview_question,
                        max_new_tokens=min(args.max_new_tokens_eval, 64),
                        system_message=train_prompt,
                        fps=args.video_fps,
                        max_frames=frames_limit,
                    )
                print(
                    f"\n[Preview] Epoch {epoch+1} Step {step}\n"
                    f"Q: {preview_question}\nGT: {preview_answer.strip()}\nPD: {preview_text}\n"
                )
                torch.cuda.empty_cache()

        avg_train_loss = running_loss / max(1, len(train_loader))
        val_loss = None
        if valid_loader is not None:
            val_loss = validate_qwen(model, valid_loader, device, vision_dtype)
            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"- train loss: {avg_train_loss:.4f}, val loss: {val_loss:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"- train loss: {avg_train_loss:.4f}"
            )

        if wandb_module is not None:
            log_payload = {
                "train/loss": avg_train_loss,
            }
            if val_loss is not None:
                log_payload["val/loss"] = val_loss
            wandb_module.log(log_payload, step=epoch + 1)

        monitor_value = val_loss if val_loss is not None else avg_train_loss
        if best_monitor is None or monitor_value < best_monitor:
            best_monitor = monitor_value
            best_epoch_num = epoch + 1
            save_model_state(model, best_model_path)
            with open(run_dir / "best_epoch.txt", "w") as f:
                f.write(f"{best_epoch_num}\n")

    return best_monitor if best_monitor is not None else avg_train_loss


def do_qwen_evaluations(
    model,
    processor,
    device,
    args,
    run_dir: Path,
    real_eval_loader=None,
    real_ood_eval_loader=None,
    endovis_eval_loader=None,
    endovis_ood_eval_loader=None,
    wandb_module=None,
    wandb_step: Optional[int] = None,
):
    if real_eval_loader is not None and args.eval_real:
        frames_limit = args.frames if getattr(args, "frames", 0) and args.frames > 0 else args.sequence_length
        gts_real, preds_real, shorts_real, questions_real, qids_real = evaluate_items_generator_qwen(
            model,
            processor,
            device,
            items_iter=iter_real_loader_qwen(real_eval_loader, desc="Eval-REAL"),
            max_new_tokens=args.max_new_tokens_eval,
            max_samples=args.max_samples_eval,
            system_message=SYSTEM_PROMPT,
            video_fps=args.video_fps,
            max_frames=frames_limit,
        )
        metrics_real = eval_and_report(
            "REAL Eval (test only)", gts_real, preds_real, run_dir, "real_test", shorts=shorts_real
        )
        save_eval_predictions(run_dir, "real_test", qids_real, questions_real, gts_real, preds_real)
        log_metrics_to_wandb(wandb_module, metrics_real, "eval/real", wandb_step)
    elif args.eval_real:
        print("[Eval] REAL evaluation skipped (loader unavailable).")

    if real_ood_eval_loader is not None and args.eval_real_ood:
        frames_limit = args.frames if getattr(args, "frames", 0) and args.frames > 0 else args.sequence_length
        gts_ood, preds_ood, shorts_ood, questions_ood, qids_ood = evaluate_items_generator_qwen(
            model,
            processor,
            device,
            items_iter=iter_real_loader_qwen(real_ood_eval_loader, desc="Eval-REAL-OOD"),
            max_new_tokens=args.max_new_tokens_eval,
            max_samples=args.max_samples_eval,
            system_message=SYSTEM_PROMPT,
            video_fps=args.video_fps,
            max_frames=frames_limit,
        )
        metrics_ood = eval_and_report(
            "REAL OOD Eval (test only)",
            gts_ood,
            preds_ood,
            run_dir,
            "real_ood_test",
            shorts=shorts_ood,
        )
        save_eval_predictions(run_dir, "real_ood_test", qids_ood, questions_ood, gts_ood, preds_ood)
        log_metrics_to_wandb(wandb_module, metrics_ood, "eval/real_ood", wandb_step)
    elif args.eval_real_ood:
        print("[Eval] REAL-OOD evaluation skipped (loader unavailable).")

    if endovis_eval_loader is not None and args.eval_endovis:
        frames_limit = args.frames if getattr(args, "frames", 0) and args.frames > 0 else args.sequence_length
        (
            gts_endovis,
            preds_endovis,
            labels_endovis,
            questions_endovis,
            qids_endovis,
        ) = evaluate_items_generator_qwen(
            model,
            processor,
            device,
            items_iter=iter_endovis_loader(endovis_eval_loader),
            max_new_tokens=args.max_new_tokens_eval,
            max_samples=args.max_samples_eval,
            system_message=SYSTEM_PROMPT_ENDOVIS,
            video_fps=args.video_fps,
            max_frames=frames_limit,
        )

        metrics_endovis = eval_and_report(
            "EndoVis Eval",
            gts_endovis,
            preds_endovis,
            run_dir,
            "endovis",
            shorts=labels_endovis,
        )
        save_eval_predictions(
            run_dir, "endovis", qids_endovis, questions_endovis, gts_endovis, preds_endovis
        )

        cls_acc = compute_endovis_classification_accuracy(labels_endovis, preds_endovis)
        if cls_acc is not None:
            metrics_endovis["classification_accuracy"] = cls_acc
            print(
                f"EndoVis classification acc: {cls_acc['acc']:.4f} "
                f"(correct={cls_acc['correct']}/{cls_acc['total']}, skipped={cls_acc['skipped']})"
            )
        metrics_path = run_dir / "metrics" / "metrics_endovis.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics_endovis, f, indent=2)
        log_metrics_to_wandb(wandb_module, metrics_endovis, "eval/endovis", wandb_step)
    elif args.eval_endovis:
        print("[Eval] EndoVis evaluation skipped (loader unavailable).")

    if endovis_ood_eval_loader is not None and getattr(args, "eval_endovis_ood", False):
        frames_limit = args.frames if getattr(args, "frames", 0) and args.frames > 0 else args.sequence_length
        (
            gts_endovis_ood,
            preds_endovis_ood,
            labels_endovis_ood,
            questions_endovis_ood,
            qids_endovis_ood,
        ) = evaluate_items_generator_qwen(
            model,
            processor,
            device,
            items_iter=iter_endovis_loader(endovis_ood_eval_loader),
            max_new_tokens=args.max_new_tokens_eval,
            max_samples=args.max_samples_eval,
            system_message=SYSTEM_PROMPT_ENDOVIS,
            video_fps=args.video_fps,
            max_frames=frames_limit,
        )

        metrics_endovis_ood = eval_and_report(
            "EndoVis OOT Eval",
            gts_endovis_ood,
            preds_endovis_ood,
            run_dir,
            "endovis_ood",
            shorts=labels_endovis_ood,
        )
        save_eval_predictions(
            run_dir,
            "endovis_ood",
            qids_endovis_ood,
            questions_endovis_ood,
            gts_endovis_ood,
            preds_endovis_ood,
        )

        cls_acc_ood = compute_endovis_classification_accuracy(labels_endovis_ood, preds_endovis_ood)
        if cls_acc_ood is not None:
            metrics_endovis_ood["classification_accuracy"] = cls_acc_ood
            print(
                f"EndoVis OOT classification acc: {cls_acc_ood['acc']:.4f} "
                f"(correct={cls_acc_ood['correct']}/{cls_acc_ood['total']}, skipped={cls_acc_ood['skipped']})"
            )
        metrics_path_ood = run_dir / "metrics" / "metrics_endovis_ood.json"
        metrics_path_ood.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path_ood, "w") as f:
            json.dump(metrics_endovis_ood, f, indent=2)
        log_metrics_to_wandb(wandb_module, metrics_endovis_ood, "eval/endovis_ood", wandb_step)
    elif getattr(args, "eval_endovis_ood", False):
        print("[Eval] EndoVis OOT evaluation skipped (loader unavailable).")
