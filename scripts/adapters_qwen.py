import inspect
import weakref
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
from peft import AdaLoraConfig, LoraConfig, get_peft_model
from peft import VeraConfig


from temporal_modules import (
    GatedAttentionWrapper,
    HeadwiseGate,
    LayerwiseGate,
    PreprocessorWrapper,
    SpatioTemporalAdapterModule,
    SpatioTemporalConvAdapter,
    STLoRA3DInline,
    STMultiHeadAttentionLoRAInline,
    STSelfAttentionLoRAInline,
    STLSTMLoRAInline,
    STMambaLoRAInline,
    TimeShiftLoRA,
    TimeShiftedVisionLinear,
    STLoRAVisionLinear,
    STAttentionLoRAVisionLinear,
    STLSTMLoRAVisionLinear,
    STMambaVisionLinear,
    set_qwen_visual_context,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DECODER_SHARED_ADAPTERS: Set[str] = {
    "stlora",
    "stdora",
    "stlora_enc",
    "stdora_enc",
    "stlora_mhatt",
    "stdora_mhatt",
    "stlora_mhatt_enc",
    "stdora_mhatt_enc",
    "stlora_selfatt",
    "stdora_selfatt",
    "stlora_selfatt_enc",
    "stdora_selfatt_enc",
    "stlora_lstm",
    "stdora_lstm",
    "stlora_lstm_enc",
    "stdora_lstm_enc",
    "stmamba",
    "stdora_mamba",
    "stmamba_enc",
    "stdora_mamba_enc",
    "timeshiftlora",
    "timeshiftdora",
    "timeshiftlora_enc",
    "timeshiftdora_enc",
}

ENCODER_WRAPPER_TYPES = (
    STLoRAVisionLinear,
    STLSTMLoRAVisionLinear,
    STAttentionLoRAVisionLinear,
    STMambaVisionLinear,
    TimeShiftedVisionLinear,
)

# Try both names used across your codebases (Intern uses both patterns)
ENCODER_INNER_ATTR_CANDIDATES = ("stlora", "adapter")

# When wrappers are installed, qkv/proj are no longer nn.Linear.
QWEN_VISUAL_ATTN_PROJ_TYPES = (nn.Linear,) + ENCODER_WRAPPER_TYPES


def _resolve_qwen_base_model(model: nn.Module) -> Tuple[nn.Module, str]:
    """
    Resolve the Qwen3-VL base model and return (base_model, path_prefix).
    The prefix is used for module path strings relative to the outer model.
    """
    def _get_submodule_safe(path: str) -> Optional[nn.Module]:
        if not path:
            return model
        try:
            return model.get_submodule(path)
        except AttributeError:
            return None

    prefixes: List[str] = [""]
    seen = set(prefixes)
    for _ in range(3):
        for prefix in list(prefixes):
            for attr in ("model", "base_model"):
                candidate = f"{prefix}{attr}."
                if candidate not in seen:
                    seen.add(candidate)
                    prefixes.append(candidate)

    for prefix in prefixes:
        module = _get_submodule_safe(prefix[:-1] if prefix else "")
        if module is None:
            continue
        if hasattr(module, "visual") and hasattr(module, "language_model"):
            return module, prefix

    raise AttributeError(
        "[Adapter][Qwen] Expected Qwen3-VL model with `visual` and `language_model` "
        "(e.g., `model.visual` / `model.language_model`)."
    )


def _try_get_submodule(root: nn.Module, path: str) -> Optional[nn.Module]:
    try:
        return root.get_submodule(path)
    except Exception:
        return None


def _resolve_qwen_visual(model: nn.Module, strict: bool = True) -> Tuple[Optional[str], Optional[nn.Module]]:
    """
    Locate the Qwen visual tower path for both plain and PEFT-wrapped layouts.
    Returns (visual_path, visual_module).
    """
    candidates = [
        "visual",
        "model.visual",
        "base_model.model.visual",
        "base_model.model.model.visual",
    ]
    for path in candidates:
        mod = _try_get_submodule(model, path)
        if isinstance(mod, nn.Module):
            return path, mod

    for name, mod in model.named_modules():
        if name.endswith(".visual") and isinstance(mod, nn.Module):
            return name, mod

    if strict:
        raise AttributeError(
            "[ST-Enc] Could not locate Qwen visual tower (tried visual / model.visual / base_model...)."
        )
    return None, None


# -----------------------------------------------------------------------------
# Visual token chunking + ST module runner (unchanged core logic)
# -----------------------------------------------------------------------------

def _chunk_qwen_visual_tokens(
    visual_outputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    grid_thw: Union[torch.Tensor, List[List[int]]],
    spatial_merge_size: int,
) -> List[Dict[str, Any]]:
    main_tokens = visual_outputs[0] if isinstance(visual_outputs, (tuple, list)) else visual_outputs
    if grid_thw is None or main_tokens is None or main_tokens.numel() == 0:
        return []
    if isinstance(grid_thw, torch.Tensor):
        grid = grid_thw.detach().cpu().tolist()
    else:
        grid = grid_thw
    if not grid:
        return []
    chunks: List[Dict[str, Any]] = []
    offset = 0
    hidden = main_tokens.size(-1)
    for entry in grid:
        if len(entry) != 3:
            raise ValueError(f"grid_thw entry must have 3 elements (t,h,w), got {entry}")
        t, h, w = [max(1, int(v)) for v in entry]
        h_merged = max(1, h // spatial_merge_size)
        w_merged = max(1, w // spatial_merge_size)
        spatial_tokens = h_merged * w_merged
        total_tokens = t * spatial_tokens

        if offset + total_tokens > main_tokens.size(0):
            raise RuntimeError(
                f"[ST-Chunk] Trying to slice past visual token length "
                f"({offset + total_tokens} > {main_tokens.size(0)})"
            )

        chunk = main_tokens[offset : offset + total_tokens]
        offset += total_tokens
        chunks.append(
            {
                "tokens": chunk.view(t, spatial_tokens, hidden),
                "grid_hw": (h_merged, w_merged),
            }
        )
    if offset != main_tokens.size(0):
        raise RuntimeError(
            f"[ST-Chunk] Visual tokens not fully consumed: offset={offset}, total={main_tokens.size(0)}"
        )
    return chunks


def _run_qwen_st_module(module: nn.Module, chunks: List[Dict[str, Any]]) -> torch.Tensor:
    if not chunks:
        return torch.empty(0)

    hidden = chunks[0]["tokens"].size(-1)
    device = chunks[0]["tokens"].device
    module_hidden = getattr(module, "hidden_size", None)
    if module_hidden is not None and int(module_hidden) != int(hidden):
        raise ValueError(
            f"[ST] Module hidden size {module_hidden} does not match visual tokens ({hidden})."
        )

    if isinstance(module, SpatioTemporalAdapterModule):
        bsz = len(chunks)
        max_t = max(chunk["tokens"].shape[0] for chunk in chunks)
        max_h = max(chunk["grid_hw"][0] for chunk in chunks)
        max_w = max(chunk["grid_hw"][1] for chunk in chunks)
        spatial_tokens = max_h * max_w
        total_tokens = spatial_tokens + 1

        stacked = chunks[0]["tokens"].new_zeros(bsz, max_t, total_tokens, hidden)
        frame_mask = torch.zeros(bsz, max_t, dtype=torch.bool, device=device)
        for idx, chunk in enumerate(chunks):
            tokens = chunk["tokens"]
            t = tokens.shape[0]
            h = chunk["grid_hw"][0]
            w = chunk["grid_hw"][1]
            spatial = h * w
            seq = tokens
            if spatial < spatial_tokens:
                pad = tokens.new_zeros(t, spatial_tokens - spatial, hidden)
                seq = torch.cat([seq, pad], dim=1)
            cls = seq.mean(dim=1, keepdim=True)
            seq = torch.cat([cls, seq], dim=1)
            stacked[idx, :t, : seq.size(1)] = seq
            frame_mask[idx, :t] = True

        out = module(stacked, num_frames=max_t, frame_mask=frame_mask)
        restored: List[torch.Tensor] = []
        for idx, chunk in enumerate(chunks):
            t = chunk["tokens"].shape[0]
            h = chunk["grid_hw"][0]
            w = chunk["grid_hw"][1]
            spatial = h * w
            seq = out[idx, :t, 1 : spatial + 1]
            restored.append(seq.reshape(t * spatial, hidden))
        return torch.cat(restored, dim=0)

    if isinstance(module, TimeShiftLoRA):
        restored: List[torch.Tensor] = []
        for chunk in chunks:
            tokens = chunk["tokens"]  # (T, HW, C)
            t = tokens.shape[0]
            h, w = chunk["grid_hw"]
            spatial = h * w
            if spatial != tokens.shape[1]:
                raise ValueError("grid_hw and token shape mismatch")

            out_btpc = module(tokens, num_frames=t)
            if out_btpc.dim() != 3:
                raise ValueError("[TimeShift-LoRA] Expected 3D output (BT, L, C)")
            if out_btpc.size(0) != t or out_btpc.size(1) != spatial:
                raise ValueError("[TimeShift-LoRA] Output shape mismatch vs. chunks")
            residual = tokens + out_btpc
            restored.append(residual.reshape(t * spatial, hidden))
        return torch.cat(restored, dim=0)

    restored: List[torch.Tensor] = []
    for chunk in chunks:
        tokens = chunk["tokens"]  # (T, HW, C)
        t = tokens.shape[0]
        h, w = chunk["grid_hw"]
        spatial = h * w
        if spatial != tokens.shape[1]:
            raise ValueError("grid_hw and token shape mismatch")

        seq = tokens.unsqueeze(0)
        frame_mask = torch.ones(t, dtype=torch.bool, device=device)

        if isinstance(module, (STMultiHeadAttentionLoRAInline, STSelfAttentionLoRAInline)):
            out_btpc = module(
                seq,
                num_frames=t,
                grid_hw=(h, w),
            )
        else:
            out_btpc = module(
                seq,
                num_frames=t,
                frame_mask=frame_mask,
                grid_hw=(h, w),
            )
        spatial_out = tokens + out_btpc[0, :, :, :]
        restored.append(spatial_out.reshape(t * spatial, hidden))

    return torch.cat(restored, dim=0)


def _apply_qwen_st_modules(model, visual_outputs: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    if not getattr(model, "_st_apply_via_hook", True):
        return visual_outputs
    has_adapter = isinstance(getattr(model, "stadapter_module", None), nn.Module)
    has_lora_module = isinstance(getattr(model, "stlora_module", None), nn.Module)
    if not (has_adapter or has_lora_module):
        return visual_outputs

    base_model, _ = _resolve_qwen_base_model(model)
    visual = base_model.visual

    spatial_merge = getattr(visual, "spatial_merge_size", 1)
    chunks = _chunk_qwen_visual_tokens(visual_outputs, grid_thw, spatial_merge)
    if not chunks:
        return visual_outputs

    module = model.stadapter_module if has_adapter else model.stlora_module
    processed = _run_qwen_st_module(module, chunks)
    if isinstance(visual_outputs, (tuple, list)):
        extras = list(visual_outputs[1:])
        return (processed, *extras)
    return processed


def _apply_st_module_tokenwise(
    module: nn.Module,
    tokens: torch.Tensor,
    grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int,
) -> torch.Tensor:
    if grid_thw is None or module is None:
        return tokens
    chunks = _chunk_qwen_visual_tokens(tokens, grid_thw, spatial_merge_size)
    if not chunks:
        return tokens
    return _run_qwen_st_module(module, chunks)


def _compute_st_delta(
    module: nn.Module,
    tokens: torch.Tensor,
    grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int,
) -> torch.Tensor:
    if grid_thw is None:
        return torch.zeros_like(tokens)
    processed = _apply_st_module_tokenwise(module, tokens, grid_thw, spatial_merge_size)
    return processed - tokens


# -----------------------------------------------------------------------------
# Qwen vision hook (unchanged)
# -----------------------------------------------------------------------------

def infer_visual_hidden_size(model, fallback: Optional[int] = None) -> int:
    def _pick(value):
        return value if isinstance(value, int) and value > 0 else None

    base_model, _ = _resolve_qwen_base_model(model)
    visual = base_model.visual
    for attr in ("out_hidden_size", "embed_dim", "hidden_size", "dim"):
        val = _pick(getattr(visual, attr, None))
        if val is not None:
            return val
    visual_cfg = getattr(visual, "config", None)
    if visual_cfg is not None:
        for attr in ("out_hidden_size", "hidden_size", "embed_dim", "projection_dim"):
            val = _pick(getattr(visual_cfg, attr, None))
            if val is not None:
                return val

    cfg = getattr(model, "config", None)
    if cfg is not None:
        vision_cfg = getattr(cfg, "vision_config", None)
        if vision_cfg is not None:
            for attr in ("out_hidden_size", "hidden_size", "embed_dim", "projection_dim"):
                val = _pick(getattr(vision_cfg, attr, None))
                if val is not None:
                    return val
        model_type = getattr(cfg, "model_type", "") or ""
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and "qwen3" in model_type.lower():
            val = _pick(getattr(text_cfg, "hidden_size", None))
            if val is not None:
                return val
        for attr in ("hidden_size", "mm_hidden_size"):
            val = _pick(getattr(cfg, attr, None))
            if val is not None:
                return val

    if isinstance(fallback, int) and fallback > 0:
        return fallback
    raise RuntimeError(
        "[ST] Unable to infer visual hidden size; pass --st_hidden explicitly."
    )


def _ensure_qwen_visual_hook(model) -> None:
    base_model, _ = _resolve_qwen_base_model(model)
    visual = base_model.visual
    if getattr(visual, "_st_hooked", False):
        return

    original_forward = visual.forward
    parent_ref = weakref.ref(model)

    def forward_with_st(self, hidden_states, grid_thw, *args, **kwargs):
        parent = parent_ref()
        if parent is not None:
            if not hasattr(parent, "_st_apply_via_hook"):
                parent._st_apply_via_hook = True
            parent._st_last_grid_thw = grid_thw

        set_qwen_visual_context(grid_thw, getattr(self, "spatial_merge_size", None))
        self._st_last_grid_thw = grid_thw

        outputs = original_forward(hidden_states, grid_thw, *args, **kwargs)
        if parent is None or grid_thw is None:
            return outputs
        return _apply_qwen_st_modules(parent, outputs, grid_thw)

    visual.forward = forward_with_st.__get__(visual, visual.__class__)
    visual._st_hooked = True


# -----------------------------------------------------------------------------
# Find vision encoder attention modules (extended: includes parent attr for paths)
# -----------------------------------------------------------------------------

def _iter_qwen_visual_attn_layers_ex(model: nn.Module, strict: bool = True):
    """
    Yields (parent_name_under_visual, parent_attn_attr, attn_module)
    where parent_attn_attr is 'attn' or 'self_attn'.
    Includes wrapped qkv/proj modules so helpers can still see adapters post-wrap.
    """
    visual_path, visual = _resolve_qwen_visual(model, strict=strict)
    if visual is None:
        return

    seen: Set[int] = set()
    blocks = getattr(visual, "blocks", None)
    if isinstance(blocks, nn.ModuleList):
        for idx, blk in enumerate(blocks):
            attn = getattr(blk, "attn", None)
            if attn is None:
                continue
            qkv = getattr(attn, "qkv", None)
            proj = getattr(attn, "proj", None)
            if isinstance(qkv, nn.Module) and isinstance(proj, nn.Module):
                if id(attn) not in seen:
                    seen.add(id(attn))
                    yield f"blocks.{idx}", "attn", attn

    if not seen:
        for parent_name, parent_module in visual.named_modules():
            attn = getattr(parent_module, "attn", None)
            if attn is None:
                attn = getattr(parent_module, "self_attn", None)
                attn_attr = "self_attn"
            else:
                attn_attr = "attn"
            if attn is None:
                continue
            qkv = getattr(attn, "qkv", None)
            proj = getattr(attn, "proj", None)
            if isinstance(qkv, nn.Module) and isinstance(proj, nn.Module):
                if id(attn) not in seen:
                    seen.add(id(attn))
                    yield parent_name, attn_attr, attn

    if not seen and strict:
        raise RuntimeError(
            "[ST-Enc] No visual attention modules with (qkv, proj) found under visual tower. "
            "This usually means the visual tower path is wrong or the model is not Qwen3-VL."
        )


def _iter_qwen_visual_attn_layers(model: nn.Module, strict: bool = True):
    for parent_name, _attn_attr, attn in _iter_qwen_visual_attn_layers_ex(model, strict=strict):
        yield parent_name, attn


# -----------------------------------------------------------------------------
# Merger target helper
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Trainability + modules_to_save helpers (NEW)
# -----------------------------------------------------------------------------

def _collect_qwen_encoder_modules_to_save(model: nn.Module) -> List[str]:
    """
    Collect deep module paths for encoder adapter params so that PEFT save_pretrained
    will include them (and PEFT will keep them trainable via modules_to_save).
    """
    paths: List[str] = []
    visual_path, _visual = _resolve_qwen_visual(model, strict=False)
    if not visual_path:
        return paths

    for parent_name, attn_attr, attn in _iter_qwen_visual_attn_layers_ex(model, strict=False):
        for proj_attr in ("qkv", "proj"):
            wrapper = getattr(attn, proj_attr, None)
            if not isinstance(wrapper, ENCODER_WRAPPER_TYPES):
                continue

            wrapper_path = f"{visual_path}.{parent_name}.{attn_attr}.{proj_attr}"
            for inner_attr in ENCODER_INNER_ATTR_CANDIDATES:
                inner = getattr(wrapper, inner_attr, None)
                if isinstance(inner, nn.Module):
                    paths.append(f"{wrapper_path}.{inner_attr}")

    # Merger/after wrappers
    visual = _try_get_submodule(model, visual_path)
    if visual is not None:
        for name, module in visual.named_modules():
            if not isinstance(module, ENCODER_WRAPPER_TYPES):
                continue
            if not (name.startswith("merger.") or name.startswith("deepstack_merger_list.")):
                continue
            for inner_attr in ENCODER_INNER_ATTR_CANDIDATES:
                inner = getattr(module, inner_attr, None)
                if isinstance(inner, nn.Module):
                    paths.append(f"{visual_path}.{name}.{inner_attr}")
    after_mod = getattr(model, "_after_adapter_modules", None)
    if isinstance(after_mod, nn.ModuleList):
        for idx, mod in enumerate(after_mod):
            if any(True for _ in mod.parameters(recurse=True)):
                paths.append(f"_after_adapter_modules.{idx}")
            for inner_attr in ENCODER_INNER_ATTR_CANDIDATES:
                inner = getattr(mod, inner_attr, None)
                if isinstance(inner, nn.Module):
                    paths.append(f"_after_adapter_modules.{idx}.{inner_attr}")
    if isinstance(getattr(model, "stlora_module", None), nn.Module):
        paths.append("stlora_module")

    # Unique + stable order
    return sorted(set(paths))


def _collect_qwen_decoder_modules_to_save(model: nn.Module) -> List[str]:
    """
    Collect decoder adapter/gating module paths so PEFT save_pretrained includes them.
    """
    base_model, prefix = _resolve_qwen_base_model(model)
    language_model = base_model.language_model
    text_model = getattr(language_model, "model", None)
    if text_model is None:
        text_model = language_model
    layers = getattr(text_model, "layers", None)
    if layers is None:
        raise AttributeError("[Adapter][Qwen] language_model has no layers.")

    prefix_root = (
        f"{prefix}language_model.model.layers"
        if hasattr(language_model, "model")
        else f"{prefix}language_model.layers"
    )
    paths: List[str] = []
    for idx, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            raise AttributeError(f"[Adapter][Qwen] Missing self_attn in decoder layer {idx}.")
        if isinstance(self_attn, GatedAttentionWrapper):
            paths.append(f"{prefix_root}.{idx}.self_attn.gate")
        for proj_attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = getattr(self_attn, proj_attr, None)
            if isinstance(module, DecoderAdapterLinear):
                paths.append(f"{prefix_root}.{idx}.self_attn.{proj_attr}.adapter")

    return sorted(set(paths))


def _modules_to_save_for_peft_enc(peft_enc: str) -> Optional[List[str]]:
    peft_enc = (peft_enc or "").lower()
    # encoder-wrapped variants handled dynamically
    return None


def _resolve_modules_to_save_path(
    model: nn.Module,
    module_name: str,
    module_names: List[str],
    module_name_set: Set[str],
) -> Optional[str]:
    # Fast path: exact path still valid.
    if module_name in module_name_set:
        return module_name
    if _try_get_submodule(model, module_name) is not None:
        return module_name

    # Common case after PEFT wrapping: same tail, different root prefix.
    suffix = f".{module_name}"
    suffix_matches = [name for name in module_names if name.endswith(suffix)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        return sorted(suffix_matches, key=len)[0]

    # Retry by progressively stripping left-most path segments.
    parts = module_name.split(".")
    for i in range(1, len(parts)):
        tail = ".".join(parts[i:])
        if tail in module_name_set:
            return tail
        tail_suffix = f".{tail}"
        tail_matches = [name for name in module_names if name.endswith(tail_suffix)]
        if len(tail_matches) == 1:
            return tail_matches[0]
        if len(tail_matches) > 1:
            return sorted(tail_matches, key=len)[0]

    return None


def _mark_modules_trainable(model: nn.Module, modules_to_save: Optional[List[str]]) -> None:
    """
    Similar to your InternVL3 helper: ensure modules_to_save params are trainable.
    Also supports the case where a module contains a ModuleDict called modules_to_save.
    """
    if not modules_to_save:
        return

    module_names = [name for name, _ in model.named_modules()]
    module_name_set = set(module_names)
    resolved_cache: Dict[str, Optional[str]] = {}

    for module_name in modules_to_save:
        resolved = resolved_cache.get(module_name)
        if resolved is None and module_name not in resolved_cache:
            resolved = _resolve_modules_to_save_path(
                model,
                module_name,
                module_names=module_names,
                module_name_set=module_name_set,
            )
            resolved_cache[module_name] = resolved
        if resolved is None:
            raise AttributeError(
                f"Missing module '{module_name}' in model (could not resolve equivalent path after wrapping)."
            )
        module = model.get_submodule(resolved)

        modules_to_save_dict = getattr(module, "modules_to_save", None)
        original_module = getattr(module, "original_module", None)

        if isinstance(modules_to_save_dict, nn.ModuleDict):
            for p in modules_to_save_dict.parameters():
                p.requires_grad = True
            if original_module is not None:
                for p in original_module.parameters():
                    p.requires_grad = False
            continue

        for p in module.parameters():
            p.requires_grad = True


def _ensure_qwen_vision_encoder_adapters_trainable(model: nn.Module) -> bool:
    """
    Walks the Qwen vision encoder and:
    - sets inner adapter params (wrapper.stlora / wrapper.adapter) trainable
    - keeps the wrapped base linear frozen if we can locate it
    Returns True if any param flags were updated.
    """
    updated = False

    for _parent_name, _attn_attr, attn in _iter_qwen_visual_attn_layers_ex(model, strict=False):
        for proj_attr in ("qkv", "proj"):
            wrapper = getattr(attn, proj_attr, None)
            if not isinstance(wrapper, ENCODER_WRAPPER_TYPES):
                continue

            # Unfreeze inner adapter module parameters
            found_inner = False
            for inner_attr in ENCODER_INNER_ATTR_CANDIDATES:
                inner = getattr(wrapper, inner_attr, None)
                if not isinstance(inner, nn.Module):
                    continue
                found_inner = True

                modules_to_save_dict = getattr(inner, "modules_to_save", None)
                original_module = getattr(inner, "original_module", None)

                if isinstance(modules_to_save_dict, nn.ModuleDict):
                    for p in modules_to_save_dict.parameters():
                        if not p.requires_grad:
                            p.requires_grad = True
                            updated = True
                    if original_module is not None:
                        for p in original_module.parameters():
                            if p.requires_grad:
                                p.requires_grad = False
                                updated = True
                else:
                    for p in inner.parameters():
                        if not p.requires_grad:
                            p.requires_grad = True
                            updated = True
            if not found_inner:
                raise AttributeError(
                    f"[Adapter][Qwen] Encoder wrapper {type(wrapper).__name__} missing adapter/stlora."
                )

            # Freeze wrapper's base/original linear if present under common names
            for base_name in ("base_layer", "original_layer", "original", "linear", "base"):
                base = getattr(wrapper, base_name, None)
                if isinstance(base, nn.Module):
                    for p in base.parameters():
                        if p.requires_grad:
                            p.requires_grad = False
                            updated = True

    return updated


def _maybe_refresh_modules_to_save_paths(model: nn.Module, peft_enc: str) -> List[str]:
    """
    Build the final modules_to_save list:
      - root modules_to_save (after variants)
      - deep paths for encoder wrappers
    """
    modules_to_save: List[str] = []
    base = _modules_to_save_for_peft_enc(peft_enc)
    if base:
        modules_to_save.extend(base)

    # If you installed encoder-wrappers (stlora_enc / stlstm_enc / stmamba_enc / timeshift_enc / st-attn enc),
    # ensure their inner adapters are included as well.
    modules_to_save.extend(_collect_qwen_encoder_modules_to_save(model))
    modules_to_save.extend(_collect_qwen_decoder_modules_to_save(model))

    modules_to_save = sorted(set(modules_to_save))
    setattr(model, "_modules_to_save_paths", modules_to_save)
    return modules_to_save


# -----------------------------------------------------------------------------
# Gated attention (unchanged)
# -----------------------------------------------------------------------------

def _infer_attn_hidden(attn_module, default_hidden: Optional[int] = None) -> Optional[int]:
    for attr in ("hidden_size", "embed_dim"):
        val = getattr(attn_module, attr, None)
        if isinstance(val, (int, float)):
            return int(val)
    qkv = getattr(attn_module, "qkv", None)
    if qkv is not None and hasattr(qkv, "in_features"):
        return int(getattr(qkv, "in_features"))
    return default_hidden


def _infer_attn_heads(attn_module, default_heads: Optional[int] = None) -> Optional[int]:
    for attr in ("num_heads", "num_attention_heads"):
        val = getattr(attn_module, attr, None)
        if isinstance(val, (int, float)):
            return int(val)
    return default_heads


def _install_qwen_vision_gated_attention(model, gate_type: str, device, dtype) -> int:
    base_model, _ = _resolve_qwen_base_model(model)
    visual = base_model.visual
    if not hasattr(visual, "blocks"):
        raise AttributeError("[Gated-Attn] Qwen3 visual tower missing blocks for encoder gating.")

    blocks = getattr(visual, "blocks")
    config = getattr(visual, "config", None)
    default_hidden = getattr(config, "hidden_size", None) if config is not None else None
    default_heads = getattr(config, "num_heads", None) if config is not None else None

    wrapped = 0
    for idx, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is None:
            raise AttributeError(f"[Gated-Attn] Missing attn in vision block {idx}.")
        if isinstance(attn, GatedAttentionWrapper):
            continue
        hidden_size = _infer_attn_hidden(attn, default_hidden)
        if hidden_size is None:
            raise ValueError(f"[Gated-Attn] Unable to infer hidden_size for vision block {idx}.")

        if gate_type == "headwise":
            num_heads = _infer_attn_heads(attn, default_heads)
            if num_heads is None:
                raise ValueError(f"[Gated-Attn] Unable to infer num_heads for vision block {idx}.")
            gate_module = HeadwiseGate(hidden_size=hidden_size, num_heads=num_heads)
        else:
            gate_module = LayerwiseGate(hidden_size=hidden_size)

        gate_module = gate_module.to(device=device, dtype=dtype)
        for p in gate_module.parameters():
            p.requires_grad = True

        wrapped_module = GatedAttentionWrapper(attn, gate_module)
        setattr(block, "attn", wrapped_module)
        wrapped += 1

        gate_desc = (
            f"HeadwiseGate(hidden={hidden_size}, heads={getattr(gate_module, 'num_heads', 'n/a')})"
            if gate_type == "headwise"
            else f"LayerwiseGate(hidden={hidden_size})"
        )
        print(f"[Gated-Attn] Wrapped vision block {idx} with {gate_desc}.")

    if wrapped == 0:
        raise RuntimeError("[Gated-Attn] No Qwen3 vision blocks were wrapped.")
    return wrapped


def install_qwen_gated_attention(model, gate_type: str = "headwise", wrap_encoder: bool = False) -> None:
    gate_type = (gate_type or "headwise").lower()
    if gate_type not in {"headwise", "layerwise"}:
        raise ValueError(f"[Gated-Attn] Unknown gate_type '{gate_type}'.")

    base_model, _ = _resolve_qwen_base_model(model)
    language_model = base_model.language_model

    text_model = getattr(language_model, "model", language_model)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        raise AttributeError("[Gated-Attn] language_model.model has no layers.")

    ref_param = next(model.parameters())
    device = ref_param.device
    dtype = ref_param.dtype

    config = getattr(text_model, "config", None)
    default_hidden = getattr(config, "hidden_size", None)
    default_heads = getattr(config, "num_attention_heads", None)

    wrapped_layers = 0
    for idx, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            raise AttributeError(f"[Gated-Attn] Missing self_attn in decoder layer {idx}.")
        if isinstance(self_attn, GatedAttentionWrapper):
            continue

        hidden_size = _infer_attn_hidden(self_attn, default_hidden)
        if hidden_size is None:
            raise ValueError(f"[Gated-Attn] Unable to infer hidden_size for decoder layer {idx}.")

        if gate_type == "headwise":
            num_heads = _infer_attn_heads(self_attn, default_heads)
            if num_heads is None:
                raise ValueError(f"[Gated-Attn] Unable to infer num_heads for decoder layer {idx}.")
            if hidden_size % num_heads != 0:
                raise ValueError(
                    f"[Gated-Attn] hidden_size {hidden_size} not divisible by num_heads {num_heads}."
                )
            gate_module = HeadwiseGate(hidden_size=hidden_size, num_heads=num_heads)
        else:
            gate_module = LayerwiseGate(hidden_size=hidden_size)

        gate_module = gate_module.to(device=device, dtype=dtype)
        for p in gate_module.parameters():
            p.requires_grad = True

        wrapped = GatedAttentionWrapper(self_attn, gate_module)
        setattr(layer, "self_attn", wrapped)
        wrapped_layers += 1

        gate_desc = (
            f"HeadwiseGate(hidden={hidden_size}, heads={getattr(gate_module, 'num_heads', 'n/a')})"
            if gate_type == "headwise"
            else f"LayerwiseGate(hidden={hidden_size})"
        )
        print(f"[Gated-Attn] Wrapped decoder layer {idx} with {gate_desc}.")

    if wrapped_layers == 0:
        raise RuntimeError("[Gated-Attn] No decoder layers were wrapped.")

    wrapped_encoder = 0
    if wrap_encoder:
        wrapped_encoder = _install_qwen_vision_gated_attention(model, gate_type, device, dtype)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[Gated-Attn] Installed {gate_type} gating on {wrapped_layers} decoder layers"
        + (f" and {wrapped_encoder} vision layers." if wrap_encoder else ".")
        + f" Trainable={trainable_params:,} / Total={total_params:,}"
    )


# -----------------------------------------------------------------------------
# Decoder shared adapter (unchanged from your original)
# -----------------------------------------------------------------------------

class DecoderAdapterLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_module: nn.Module,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.adapter = adapter_module
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        for p in self.adapter.parameters():
            p.requires_grad = True

    def _apply_adapter(self, seq: torch.Tensor, frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if seq.numel() == 0:
            return seq.new_zeros(seq.size(0), seq.size(1), self.out_features)
        bsz, t, _ = seq.shape
        x_btpc = seq.view(bsz, t, 1, self.in_features)
        kwargs: Dict[str, Any] = {"num_frames": t}
        if frame_mask is not None:
            kwargs["frame_mask"] = frame_mask
        delta = self.adapter(x_btpc, **kwargs)
        return delta.view(bsz, t, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if x.dim() != 3:
            return base_out

        delta = self._apply_adapter(x)
        return base_out + delta


def _describe_qwen_attention_cls(attention_cls: Type[nn.Module]) -> str:
    if attention_cls is STSelfAttentionLoRAInline:
        return "ST-SelfAttn"
    if attention_cls is STMultiHeadAttentionLoRAInline:
        return "ST-MHAttn"
    return attention_cls.__name__


def _make_qwen_attention_module(
    attention_cls: Type[nn.Module],
    hidden_dim: int,
    rank: int,
    alpha: float,
    dropout: float,
    debug: bool,
    num_heads: Optional[int] = None,
    out_size: Optional[int] = None,
    use_dora: bool = False,
) -> nn.Module:
    kwargs = dict(hidden_size=hidden_dim, rank=rank, alpha=alpha, dropout=dropout, debug=debug, use_dora=use_dora)
    if out_size is not None:
        kwargs["out_size"] = out_size
    if issubclass(attention_cls, STMultiHeadAttentionLoRAInline) and num_heads is not None:
        kwargs["num_heads"] = num_heads
    return attention_cls(**kwargs)


def _select_qwen_attention_cls(adapter_name: str) -> Type[nn.Module]:
    adapter_name = adapter_name.lower()
    if "mhatt" in adapter_name:
        return STMultiHeadAttentionLoRAInline
    if "selfatt" in adapter_name:
        return STSelfAttentionLoRAInline
    return STMultiHeadAttentionLoRAInline


def _build_decoder_adapter_module(base_layer: nn.Linear, adapter_name: str, args) -> nn.Module:
    hidden = int(base_layer.in_features)
    out_features = int(base_layer.out_features)
    rank = int(getattr(args, "adapter_rank", 8))
    alpha = float(getattr(args, "adapter_alpha", 16.0))
    dropout = float(getattr(args, "adapter_dropout", 0.1))
    adapter_lower = adapter_name.lower()

    use_dora = "dora" in adapter_lower

    if "timeshiftdora" in adapter_lower:
        module = TimeShiftLoRA(
            hidden_size=hidden,
            rank=rank,
            alpha=alpha,
            n_div=getattr(args, "tslora_n_div", 3),
            dropout=dropout,
            out_dim=out_features,
            use_dora=True,
        )
    elif "timeshiftlora" in adapter_lower:
        module = TimeShiftLoRA(
            hidden_size=hidden,
            rank=rank,
            alpha=alpha,
            n_div=getattr(args, "tslora_n_div", 3),
            dropout=dropout,
            out_dim=out_features,
            use_dora=False,
        )
    elif "mamba" in adapter_lower:
        module = STMambaLoRAInline(
            hidden_size=hidden,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            out_size=out_features,
            use_dora=use_dora,
        )
    elif "lstm" in adapter_lower:
        module = STLSTMLoRAInline(
            hidden_size=hidden,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            out_size=out_features,
            use_dora=use_dora,
        )
    elif "mhatt" in adapter_lower or "selfatt" in adapter_lower:
        attention_cls = _select_qwen_attention_cls(adapter_lower)
        num_heads = max(4, rank // 4) if issubclass(attention_cls, STMultiHeadAttentionLoRAInline) else 1
        module = _make_qwen_attention_module(
            attention_cls,
            hidden_dim=hidden,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            debug=False,
            num_heads=num_heads,
            out_size=out_features,
            use_dora=use_dora,
        )
    else:
        module = STLoRA3DInline(
            hidden_size=hidden,
            rank=rank,
            alpha=alpha,
            k_t=getattr(args, "st_kernel", 3),
            k_s=getattr(args, "st_kernel", 3),
            dropout=dropout,
            debug=False,
            out_size=out_features,
            use_dora=use_dora,
        )

    module = module.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
    for p in module.parameters():
        p.requires_grad = True
    return module


def install_qwen_decoder_shared_adapter(model, args, adapter_name: str):
    base_model, _ = _resolve_qwen_base_model(model)
    language_model = base_model.language_model
    text_model = getattr(language_model, "model", language_model)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        raise AttributeError("[Decoder-Adapter] language_model.model has no layers.")

    wrapped = 0
    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            raise AttributeError("[Decoder-Adapter] Missing self_attn in decoder layer.")
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            base_layer = getattr(self_attn, attr, None)
            if base_layer is None:
                raise AttributeError(f"[Decoder-Adapter] Missing {attr} in decoder self_attn.")
            if not isinstance(base_layer, nn.Linear):
                raise TypeError(f"[Decoder-Adapter] Expected nn.Linear for {attr}, got {type(base_layer)}.")
            adapter_module = _build_decoder_adapter_module(base_layer, adapter_name, args)
            wrapped_layer = DecoderAdapterLinear(
                base_layer=base_layer,
                adapter_module=adapter_module,
            )
            setattr(self_attn, attr, wrapped_layer)
            wrapped += 1

    if wrapped == 0:
        raise RuntimeError("[Decoder-Adapter] No decoder projections were wrapped.")

    print(f"[Decoder-Adapter] Installed {adapter_name} on decoder projections ({wrapped} linears).")
    return model


# -----------------------------------------------------------------------------
# Encoder wrapper installers (FIXED: no bogus model.stlora_module=True, and we keep them trainable)
# -----------------------------------------------------------------------------

def install_qwen_st_lora_encoder(model, args, use_dora: bool = False):
    rank = args.adapter_rank
    alpha = args.adapter_alpha
    dropout = args.adapter_dropout

    count_qkv = 0
    count_proj = 0
    for name, attn in _iter_qwen_visual_attn_layers(model):
        original_qkv = attn.qkv
        original_proj = attn.proj

        if not isinstance(original_qkv, nn.Linear) or not isinstance(original_proj, nn.Linear):
            raise TypeError(
                f"[Qwen-ST-LoRA-Enc] Expected Linear qkv/proj in {name}, "
                f"got {type(original_qkv)} / {type(original_proj)}."
            )

        wrapped_qkv = STLoRAVisionLinear(
            original_qkv,
            rank=rank,
            alpha=alpha,
            k=args.st_kernel,
            dropout=dropout,
            debug=False,
            use_dora=use_dora,
        ).to(original_qkv.weight.device, dtype=original_qkv.weight.dtype)
        wrapped_qkv.layer_type = f"{name}_QKV_STLoRA"
        attn.qkv = wrapped_qkv
        count_qkv += 1

        wrapped_proj = STLoRAVisionLinear(
            original_proj,
            rank=rank,
            alpha=alpha,
            k=args.st_kernel,
            dropout=dropout,
            debug=False,
            use_dora=use_dora,
        ).to(original_proj.weight.device, dtype=original_proj.weight.dtype)
        wrapped_proj.layer_type = f"{name}_Proj_STLoRA"
        attn.proj = wrapped_proj
        count_proj += 1

    if count_qkv == 0 and count_proj == 0:
        raise RuntimeError("[Qwen-ST-LoRA-Enc] No visual encoder layers were wrapped.")

    # ensure trainability NOW (before any PEFT wrap)
    _ensure_qwen_vision_encoder_adapters_trainable(model)

    adapter_tag = "STDora" if use_dora else "STLoRA"
    print(f"[Qwen-{adapter_tag}-Enc] Wrapped {count_qkv} qkv and {count_proj} proj layers with {adapter_tag}.")


def install_qwen_st_attention_encoder(
    model,
    args,
    attention_cls: Type[nn.Module] = STMultiHeadAttentionLoRAInline,
    use_dora: bool = False,
):
    rank = args.adapter_rank
    alpha = args.adapter_alpha
    dropout = args.adapter_dropout

    count_qkv = 0
    count_proj = 0
    for name, attn in _iter_qwen_visual_attn_layers(model):
        original_qkv = attn.qkv
        original_proj = attn.proj

        if not isinstance(original_qkv, nn.Linear) or not isinstance(original_proj, nn.Linear):
            raise TypeError(
                f"[Qwen-ST-Att-Enc] Expected Linear qkv/proj in {name}, "
                f"got {type(original_qkv)} / {type(original_proj)}."
            )

        wrapped_qkv = STAttentionLoRAVisionLinear(
            original_qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            num_heads=max(4, rank // 4),
            debug=False,
            attention_cls=attention_cls,
            use_dora=use_dora,
        ).to(original_qkv.weight.device, dtype=original_qkv.weight.dtype)
        wrapped_qkv.layer_type = f"{name}_QKV_STAttn"
        attn.qkv = wrapped_qkv
        count_qkv += 1

        wrapped_proj = STAttentionLoRAVisionLinear(
            original_proj,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            num_heads=max(4, rank // 4),
            debug=False,
            attention_cls=attention_cls,
            use_dora=use_dora,
        ).to(original_proj.weight.device, dtype=original_proj.weight.dtype)
        wrapped_proj.layer_type = f"{name}_Proj_STAttn"
        attn.proj = wrapped_proj
        count_proj += 1

    if count_qkv == 0 and count_proj == 0:
        raise RuntimeError("[Qwen-ST-Att-Enc] No visual encoder layers were wrapped.")

    _ensure_qwen_vision_encoder_adapters_trainable(model)

    head_info = f", heads={max(4, rank // 4)}" if issubclass(attention_cls, STMultiHeadAttentionLoRAInline) else ""
    adapter_tag = "STDora-" + _describe_qwen_attention_cls(attention_cls) if use_dora else _describe_qwen_attention_cls(attention_cls)
    print(
        f"[{adapter_tag}-Enc] Wrapped {count_qkv} qkv and "
        f"{count_proj} proj layers (rank={rank}, alpha={alpha}, dropout={dropout}{head_info})."
    )


def install_qwen_stlstm_lora_encoder(model, args, use_dora: bool = False):
    rank = args.adapter_rank
    alpha = args.adapter_alpha
    dropout = args.adapter_dropout

    count_qkv = 0
    count_proj = 0
    for name, attn in _iter_qwen_visual_attn_layers(model):
        original_qkv = attn.qkv
        original_proj = attn.proj

        if not isinstance(original_qkv, nn.Linear) or not isinstance(original_proj, nn.Linear):
            raise TypeError(
                f"[Qwen-ST-LSTM-Enc] Expected Linear qkv/proj in {name}, "
                f"got {type(original_qkv)} / {type(original_proj)}."
            )

        wrapped_qkv = STLSTMLoRAVisionLinear(
            original_qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            num_layers=1,
            bidirectional=False,
            debug=False,
            use_dora=use_dora,
        ).to(original_qkv.weight.device, dtype=original_qkv.weight.dtype)
        wrapped_qkv.layer_type = f"{name}_QKV_STLSTM"
        attn.qkv = wrapped_qkv
        count_qkv += 1

        wrapped_proj = STLSTMLoRAVisionLinear(
            original_proj,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            num_layers=1,
            bidirectional=False,
            debug=False,
            use_dora=use_dora,
        ).to(original_proj.weight.device, dtype=original_proj.weight.dtype)
        wrapped_proj.layer_type = f"{name}_Proj_STLSTM"
        attn.proj = wrapped_proj
        count_proj += 1

    if count_qkv == 0 and count_proj == 0:
        raise RuntimeError("[Qwen-ST-LSTM-Enc] No visual encoder layers were wrapped.")

    _ensure_qwen_vision_encoder_adapters_trainable(model)

    tag = "STDora-LSTM" if use_dora else "ST-LSTM-LoRA"
    print(f"[Qwen-{tag}-Enc] Wrapped {count_qkv} qkv and {count_proj} proj layers with {tag}.")


def install_qwen_st_mamba_encoder(model, args, use_dora: bool = False):
    rank = args.adapter_rank
    alpha = args.adapter_alpha
    dropout = args.adapter_dropout

    count_qkv = 0
    count_proj = 0
    for name, attn in _iter_qwen_visual_attn_layers(model):
        original_qkv = attn.qkv
        original_proj = attn.proj

        if not isinstance(original_qkv, nn.Linear) or not isinstance(original_proj, nn.Linear):
            raise TypeError(
                f"[Qwen-ST-Mamba-Enc] Expected Linear qkv/proj in {name}, "
                f"got {type(original_qkv)} / {type(original_proj)}."
            )

        wrapped_qkv = STMambaVisionLinear(
            original_qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            d_state=16,
            d_conv=4,
            expand=2,
            debug=False,
            use_dora=use_dora,
        ).to(original_qkv.weight.device, dtype=original_qkv.weight.dtype)
        wrapped_qkv.layer_type = f"{name}_QKV_STMamba"
        attn.qkv = wrapped_qkv
        count_qkv += 1

        wrapped_proj = STMambaVisionLinear(
            original_proj,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            d_state=16,
            d_conv=4,
            expand=2,
            debug=False,
            use_dora=use_dora,
        ).to(original_proj.weight.device, dtype=original_proj.weight.dtype)
        wrapped_proj.layer_type = f"{name}_Proj_STMamba"
        attn.proj = wrapped_proj
        count_proj += 1

    if count_qkv == 0 and count_proj == 0:
        raise RuntimeError("[Qwen-ST-Mamba-Enc] No visual encoder layers were wrapped.")

    _ensure_qwen_vision_encoder_adapters_trainable(model)

    tag = "STDora-Mamba" if use_dora else "STMamba-LoRA"
    print(f"[Qwen-{tag}-Enc] Wrapped {count_qkv} qkv and {count_proj} proj layers with {tag}.")


def install_qwen_timeshift_lora_encoder(model, args, use_dora: bool = False):
    rank = args.adapter_rank
    alpha = args.adapter_alpha
    dropout = args.adapter_dropout
    num_frames = (
        int(args.frames)
        if getattr(args, "frames", 0) and args.frames > 0
        else int(getattr(args, "adapter_frames", 8))
    )

    count_qkv = 0
    count_proj = 0
    for name, attn in _iter_qwen_visual_attn_layers(model):
        original_qkv = attn.qkv
        original_proj = attn.proj

        if not isinstance(original_qkv, nn.Linear) or not isinstance(original_proj, nn.Linear):
            raise TypeError(
                f"[Qwen-TimeShift-Enc] Expected Linear qkv/proj in {name}, "
                f"got {type(original_qkv)} / {type(original_proj)}."
            )

        wrapped_qkv = TimeShiftedVisionLinear(
            original_qkv,
            rank=rank,
            alpha=alpha,
            n_div=args.tslora_n_div,
            dropout=dropout,
            num_frames=num_frames,
            use_dora=use_dora,
        ).to(original_qkv.weight.device, dtype=original_qkv.weight.dtype)
        wrapped_qkv.layer_type = f"{name}_QKV_TimeShift"
        attn.qkv = wrapped_qkv
        count_qkv += 1

        wrapped_proj = TimeShiftedVisionLinear(
            original_proj,
            rank=rank,
            alpha=alpha,
            n_div=args.tslora_n_div,
            dropout=dropout,
            num_frames=num_frames,
            use_dora=use_dora,
        ).to(original_proj.weight.device, dtype=original_proj.weight.dtype)
        wrapped_proj.layer_type = f"{name}_Proj_TimeShift"
        attn.proj = wrapped_proj
        count_proj += 1

    if count_qkv == 0 and count_proj == 0:
        raise RuntimeError("[Qwen-TimeShift-Enc] No visual encoder layers were wrapped.")

    _ensure_qwen_vision_encoder_adapters_trainable(model)

    print(
        f"[Qwen-TimeShift-Enc] Wrapped {count_qkv} qkv and {count_proj} proj layers with "
        f"TimeShift-{'DoRA' if use_dora else 'LoRA'}."
    )


# -----------------------------------------------------------------------------
# Merger wrapper helper (legacy) + ST adapter on tower
# -----------------------------------------------------------------------------

def _iter_qwen_mergers(model: nn.Module):
    _visual_path, visual = _resolve_qwen_visual(model, strict=True)
    base_merger = getattr(visual, "merger", None)
    if base_merger is not None:
        yield "merger", base_merger
    deepstack = getattr(visual, "deepstack_merger_list", None)
    if isinstance(deepstack, nn.ModuleList):
        for idx, merger in enumerate(deepstack):
            yield f"deepstack_merger_list.{idx}", merger


def _iter_qwen_merger_linears(model: nn.Module):
    mergers = list(_iter_qwen_mergers(model))
    if not mergers:
        raise AttributeError("[Adapter][Qwen] Visual tower missing merger modules.")
    for _name, merger in mergers:
        for attr in ("linear_fc1", "linear_fc2"):
            layer = getattr(merger, attr, None)
            if isinstance(layer, nn.Linear):
                yield merger, attr, layer


def _collect_qwen_merger_lora_targets(model: nn.Module) -> List[str]:
    targets: List[str] = []
    visual_path, visual = _resolve_qwen_visual(model, strict=False)
    if visual is None:
        return targets

    for name, module in visual.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.startswith("merger.") or name.startswith("deepstack_merger_list."):
            targets.append(f"{visual_path}.{name}")

    return sorted(set(targets))


def _install_qwen_merger_after_adapter(model, adapter_name: str, args) -> None:
    adapter_lower = adapter_name.lower()
    use_dora = "stdora" in adapter_lower
    use_mhatt = "mhatt" in adapter_lower
    rank = int(args.adapter_rank)
    alpha = float(args.adapter_alpha)
    dropout = float(args.adapter_dropout)
    num_heads = max(4, rank // 4)

    _ensure_qwen_visual_hook(model)
    base_model, _ = _resolve_qwen_base_model(model)
    visual = base_model.visual
    token_hidden_size = None
    patch_embed = getattr(visual, "patch_embed", None)
    if patch_embed is not None:
        val = getattr(patch_embed, "embed_dim", None)
        if isinstance(val, int) and val > 0:
            token_hidden_size = int(val)
    if token_hidden_size is None:
        pos_embed = getattr(visual, "pos_embed", None)
        if isinstance(pos_embed, nn.Embedding):
            token_hidden_size = int(pos_embed.embedding_dim)
    if token_hidden_size is None:
        blocks = getattr(visual, "blocks", None)
        if isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
            first_attn = getattr(blocks[0], "attn", None)
            qkv = getattr(first_attn, "qkv", None) if first_attn is not None else None
            if isinstance(qkv, nn.Linear):
                token_hidden_size = int(qkv.in_features)
    if token_hidden_size is None:
        raise RuntimeError("[Adapter][Qwen-After] Unable to infer visual token hidden size.")
    ref_param = next(model.parameters())
    installed_modules: List[nn.Module] = []
    wrapped_mergers = 0

    for merger_name, merger in _iter_qwen_mergers(model):
        fc1 = getattr(merger, "linear_fc1", None)
        if not isinstance(fc1, nn.Linear):
            raise AttributeError(f"[Adapter][Qwen-After] {merger_name} missing linear_fc1.")
        if use_mhatt:
            module = STMultiHeadAttentionLoRAInline(
                hidden_size=token_hidden_size,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                num_heads=num_heads,
                use_dora=use_dora,
            ).to(device=ref_param.device, dtype=ref_param.dtype)
        else:
            module = STLoRA3DInline(
                hidden_size=token_hidden_size,
                rank=rank,
                alpha=alpha,
                k_t=args.st_kernel,
                k_s=args.st_kernel,
                dropout=dropout,
                use_dora=use_dora,
            ).to(device=ref_param.device, dtype=ref_param.dtype)

        for param in module.parameters():
            param.requires_grad = True

        def preprocess(
            hidden_states: torch.Tensor,
            adapter_module: nn.Module = module,
            merger_tag: str = merger_name,
        ) -> torch.Tensor:
            if hidden_states.dim() != 2:
                raise ValueError(
                    f"[Adapter][Qwen-After] {merger_tag} expects 2D merger input, got {tuple(hidden_states.shape)}."
                )
            if hidden_states.size(-1) != token_hidden_size:
                raise ValueError(
                    f"[Adapter][Qwen-After] {merger_tag} expected hidden dim {token_hidden_size}, "
                    f"got {hidden_states.size(-1)}."
                )
            grid_thw = getattr(model, "_st_last_grid_thw", None)
            if grid_thw is None:
                raise ValueError(
                    f"[Adapter][Qwen-After] Missing grid_thw context while processing {merger_tag}."
                )

            # Merger inputs are pre-merge patch tokens (T * H * W), so do not apply spatial merge here.
            chunks = _chunk_qwen_visual_tokens(hidden_states, grid_thw, spatial_merge_size=1)
            if not chunks:
                raise ValueError(
                    f"[Adapter][Qwen-After] Empty visual chunk list for {merger_tag}."
                )
            processed = _run_qwen_st_module(adapter_module, chunks)
            if processed.shape != hidden_states.shape:
                raise ValueError(
                    f"[Adapter][Qwen-After] Processed shape {tuple(processed.shape)} "
                    f"does not match merger input {tuple(hidden_states.shape)} for {merger_tag}."
                )
            return processed - hidden_states

        wrapped = PreprocessorWrapper(preprocess, merger)
        if merger_name == "merger":
            visual.merger = wrapped
        elif merger_name.startswith("deepstack_merger_list."):
            idx = int(merger_name.rsplit(".", 1)[-1])
            visual.deepstack_merger_list[idx] = wrapped
        else:
            raise ValueError(f"[Adapter][Qwen-After] Unsupported merger path {merger_name}.")

        installed_modules.append(module)
        wrapped_mergers += 1

    if wrapped_mergers == 0:
        raise RuntimeError("[Adapter][Qwen-After] No merger modules were wrapped.")

    model._after_adapter_modules = nn.ModuleList(installed_modules)
    model.stlora_module = None
    model.stlora_num_frames = (
        int(args.frames)
        if getattr(args, "frames", 0) and args.frames > 0
        else int(getattr(args, "adapter_frames", 8))
    )
    model.stadapter_module = None
    model._st_apply_via_hook = False

    print(
        f"[Adapter][Qwen-After] Installed "
        f"{'STDoRA' if use_dora else 'STLoRA'}{'-MHAttn' if use_mhatt else ''} "
        f"on {wrapped_mergers} merger modules (after visual blocks, before merger MLP)."
    )


def install_qwen_st_adapter(model, args):
    hidden_size = infer_visual_hidden_size(model, args.st_hidden)
    _ensure_qwen_visual_hook(model)

    ref_param = next(model.parameters())
    adapter = SpatioTemporalAdapterModule(
        hidden_size=hidden_size,
        adapter_channels=args.st_adapter_channels,
        kernel_size=(args.st_adapter_kernel_t, args.st_adapter_kernel_h, args.st_adapter_kernel_w),
        disable_cudnn=args.st_adapter_disable_cudnn,
    ).to(device=ref_param.device, dtype=ref_param.dtype)

    for param in adapter.parameters():
        param.requires_grad = True

    model.stadapter_module = adapter
    model.stadapter_num_frames = (
        int(args.frames)
        if getattr(args, "frames", 0) and args.frames > 0
        else int(getattr(args, "adapter_frames", 8))
    )
    model.stlora_module = None
    model._st_apply_via_hook = True

    print(
        "[ST-Adapter] Installed on Qwen3-VL visual tower "
        f"(hidden={hidden_size}, channels={args.st_adapter_channels}, "
        f"kernel=({args.st_adapter_kernel_t}, {args.st_adapter_kernel_h}, {args.st_adapter_kernel_w}), "
        f"disable_cudnn={args.st_adapter_disable_cudnn})"
    )


def configure_adapter_clip_length(model, clip_length: int) -> None:
    if clip_length <= 0:
        return

    if isinstance(getattr(model, "stlora_module", None), nn.Module):
        model.stlora_num_frames = clip_length
    if isinstance(getattr(model, "stadapter_module", None), nn.Module):
        model.stadapter_num_frames = clip_length
    if isinstance(getattr(model, "timeshift_module", None), nn.Module):
        model.timeshift_num_frames = clip_length

    for _, attn in _iter_qwen_visual_attn_layers(model):
        for attr in ("qkv", "proj"):
            module = getattr(attn, attr, None)
            if isinstance(module, ENCODER_WRAPPER_TYPES):
                module.num_frames = clip_length

    for _, _, module in _iter_qwen_merger_linears(model):
        if isinstance(module, ENCODER_WRAPPER_TYPES):
            module.num_frames = clip_length


def freeze_model_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


# -----------------------------------------------------------------------------
# PEFT helpers (FIXED: modules_to_save + post-peft trainability pass)
# -----------------------------------------------------------------------------

def _lora_target_modules(include_decoder: bool, include_encoder: bool) -> List[str]:
    modules: List[str] = []
    if include_decoder:
        modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
    if include_encoder:
        modules.extend(["attn.proj", "attn.qkv"])
    return modules


def _build_lora_config(
    args,
    target_modules: List[str],
    use_dora: bool = False,
    modules_to_save: Optional[List[str]] = None,
) -> LoraConfig:
    if not target_modules:
        raise ValueError("[Adapter] No target modules provided for LoRA.")

    return LoraConfig(
        use_dora=use_dora,
        r=args.adapter_rank,
        lora_alpha=args.adapter_alpha,
        lora_dropout=args.adapter_dropout,
        target_modules=sorted(set(target_modules)),
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )


def _inject_peft_adapter(
    model: nn.Module,
    cfg: LoraConfig,
    adapter_name: Optional[str] = None,
) -> Tuple[nn.Module, str]:
    requested_name = adapter_name or "default"

    if hasattr(model, "peft_config"):
        if not hasattr(model, "add_adapter"):
            raise AttributeError(
                "[Adapter] Multiple LoRA/DoRA adapters requested, but this PEFT version "
                "does not support add_adapter(). Please upgrade peft to enable mixed adapters."
            )
        name = requested_name
        peft_cfg = getattr(model, "peft_config", None)
        if isinstance(peft_cfg, dict) and name in peft_cfg:
            suffix = 1
            while f"{requested_name}_{suffix}" in peft_cfg:
                suffix += 1
            name = f"{requested_name}_{suffix}"
        model.add_adapter(name, cfg)
        return model, name

    # First adapter injection
    try:
        model = get_peft_model(model, cfg, adapter_name=adapter_name)
    except TypeError:
        model = get_peft_model(model, cfg)

    name = requested_name
    peft_cfg = getattr(model, "peft_config", None)
    if isinstance(peft_cfg, dict) and name not in peft_cfg:
        keys = list(peft_cfg.keys())
        if keys:
            name = keys[0]
    return model, name


def _set_active_adapters(model: nn.Module, adapter_names: List[str]) -> None:
    if not adapter_names:
        return
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_names if len(adapter_names) > 1 else adapter_names[0])
        return
    if hasattr(model, "active_adapter"):
        model.active_adapter = adapter_names if len(adapter_names) > 1 else adapter_names[0]
        return
    if hasattr(model, "active_adapters"):
        model.active_adapters = adapter_names
        return
    raise AttributeError(
        "[Adapter] PEFT model does not support selecting multiple adapters. "
        "Please upgrade peft to enable mixed adapters."
    )


def _apply_lora(
    model: nn.Module,
    args,
    target_modules: List[str],
    use_dora: bool = False,
    modules_to_save: Optional[List[str]] = None,
    adapter_name: Optional[str] = None,
) -> Tuple[nn.Module, str]:
    cfg = _build_lora_config(args, target_modules, use_dora=use_dora, modules_to_save=modules_to_save)
    model, adapter_name = _inject_peft_adapter(model, cfg, adapter_name=adapter_name)

    # Ensure modules_to_save remain trainable (PEFT uses these for saving)
    _mark_modules_trainable(model, modules_to_save)

    # Re-ensure encoder wrapper adapters remain trainable after PEFT wrap
    _ensure_qwen_vision_encoder_adapters_trainable(model)

    if getattr(args, "debug", False):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[Adapter][DEBUG] After PEFT ({adapter_name}): trainable={trainable:,} / total={total:,}")
        # Print a few trainable names as sanity check
        shown = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"  [trainable] {n} ({p.numel():,})")
                shown += 1
                if shown >= 15:
                    break

    return model, adapter_name


def _build_vera_config(
    args,
    target_modules: List[str],
    modules_to_save: Optional[List[str]] = None,
    task_type: Optional[str] = None,
):
    if VeraConfig is None:
        raise ImportError(
            "VeraConfig is not available in the installed peft version. "
            "Please upgrade peft to a release that includes VeRA support."
        )
    if not target_modules:
        raise ValueError("[Adapter] No target modules provided for VeRA.")

    requested = {
        "r": args.adapter_rank,
        "rank": args.adapter_rank,
        "alpha": args.adapter_alpha,
        "vera_alpha": args.adapter_alpha,
        "lora_alpha": args.adapter_alpha,
        "dropout": args.adapter_dropout,
        "lora_dropout": args.adapter_dropout,
        "target_modules": sorted(set(target_modules)),
        "modules_to_save": modules_to_save,
        "bias": "none",
        "task_type": task_type,
    }

    sig = inspect.signature(VeraConfig)
    vera_kwargs = {
        name: value
        for name, value in requested.items()
        if name in sig.parameters and value is not None
    }

    missing = [
        name
        for name, param in sig.parameters.items()
        if name != "self" and param.default is inspect._empty and name not in vera_kwargs
    ]
    if missing:
        raise ValueError(
            f"VeraConfig is missing required parameters {missing}. "
            "Please update the adapter installer to match your peft version."
        )

    return VeraConfig(**vera_kwargs)


def _apply_vera(
    model: nn.Module,
    args,
    target_modules: List[str],
    modules_to_save: Optional[List[str]] = None,
    task_type: Optional[str] = None,
) -> nn.Module:
    cfg = _build_vera_config(
        args,
        target_modules,
        modules_to_save=modules_to_save,
        task_type=task_type,
    )
    model = get_peft_model(model, cfg)

    _mark_modules_trainable(model, modules_to_save)
    _ensure_qwen_vision_encoder_adapters_trainable(model)

    if getattr(args, "debug", False):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[Adapter][DEBUG] After VeRA: trainable={trainable:,} / total={total:,}")

    return model


# -----------------------------------------------------------------------------
# Train state helpers (unchanged)
# -----------------------------------------------------------------------------

def _set_module_requires_grad(module: Optional[nn.Module], requires_grad: bool) -> None:
    if module is None:
        raise AttributeError("[Adapter][Qwen] Expected module for train-state update, got None.")
    for param in module.parameters():
        param.requires_grad = requires_grad


def _set_encoder_train_state(model: nn.Module, peft_enc: str) -> None:
    base_model, _ = _resolve_qwen_base_model(model)
    visual = base_model.visual
    if peft_enc == "none":
        _set_module_requires_grad(visual, True)
    elif peft_enc == "frozen":
        _set_module_requires_grad(visual, False)


def _set_decoder_train_state(model: nn.Module, peft_dec: str) -> None:
    base_model, _ = _resolve_qwen_base_model(model)
    language_model = base_model.language_model
    if peft_dec == "none":
        _set_module_requires_grad(language_model, True)
    elif peft_dec == "frozen":
        _set_module_requires_grad(language_model, False)


# -----------------------------------------------------------------------------
# Main setup_adapter (FIXED: modules_to_save + ensure trainable encoder wrappers)
# -----------------------------------------------------------------------------

def setup_adapter(model, args):
    peft_enc = args.peft_enc.lower()
    peft_dec = args.peft_dec.lower()
    need_adalora_setup = False
    enable_vision_grad = False

    # Start from fully frozen model
    for p in model.parameters():
        p.requires_grad = False

    enc_lora = peft_enc in {"lora_enc"}
    enc_dora = peft_enc in {"dora_enc"}
    enc_vera = peft_enc in {"vera_enc"}
    enc_adalora = peft_enc in {"adalora_enc"}
    dec_lora = peft_dec == "lora"
    dec_dora = peft_dec == "dora"
    dec_vera = peft_dec == "vera"

    if enc_vera or dec_vera:
        if enc_lora or enc_dora or enc_adalora or dec_lora or dec_dora or peft_dec == "adalora":
            raise ValueError("VeRA cannot be combined with LoRA/DoRA/AdaLoRA in this setup.")
    if peft_dec == "adalora" and (enc_lora or enc_dora):
        raise ValueError("AdaLoRA cannot be combined with LoRA/DoRA on the encoder.")
    if enc_adalora and peft_dec in {"lora", "dora"}:
        raise ValueError("AdaLoRA encoder cannot be combined with decoder LoRA/DoRA.")

    # Optional gated attention
    if peft_enc in {"gated_headwise", "gated_elementwise"}:
        install_qwen_gated_attention(
            model,
            gate_type="headwise" if peft_enc == "gated_headwise" else "layerwise",
            wrap_encoder=True,
        )
        enable_vision_grad = True

    hidden_size = infer_visual_hidden_size(model, args.st_hidden)

    # ------------------------
    # Encoder-side custom adapters (encoder wrappers)
    # ------------------------

    if peft_enc in {
        "stlora_enc",
        "stdora_enc",
        "stlora_mhatt_enc",
        "stdora_mhatt_enc",
        "stlora_selfatt_enc",
        "stdora_selfatt_enc",
    }:
        use_dora = peft_enc.startswith("stdora")
        if peft_enc in {"stlora_enc", "stdora_enc"}:
            install_qwen_st_lora_encoder(model, args, use_dora=use_dora)
        else:
            attention_cls = _select_qwen_attention_cls(peft_enc)
            install_qwen_st_attention_encoder(model, args, attention_cls=attention_cls, use_dora=use_dora)
        enable_vision_grad = True

    elif peft_enc in {"stlora_lstm_enc", "stdora_lstm_enc"}:
        install_qwen_stlstm_lora_encoder(model, args, use_dora=peft_enc.startswith("stdora"))
        enable_vision_grad = True

    elif peft_enc in {"stmamba_enc", "stdora_mamba_enc"}:
        install_qwen_st_mamba_encoder(model, args, use_dora=peft_enc.startswith("stdora"))
        enable_vision_grad = True

    elif peft_enc in {"timeshiftlora_enc", "timeshiftdora_enc"}:
        install_qwen_timeshift_lora_encoder(model, args, use_dora=peft_enc.startswith("timeshift") and "dora" in peft_enc)
        enable_vision_grad = True

    elif peft_enc in {"stlora_after", "stdora_after", "stlora_mhatt_after", "stdora_mhatt_after"}:
        _install_qwen_merger_after_adapter(model, peft_enc, args)
        enable_vision_grad = True

    elif peft_enc not in {"none", "frozen", "lora_enc", "dora_enc", "vera_enc", "adalora_enc"}:
        raise ValueError(f"Unknown encoder PEFT type: {args.peft_enc}")

    # Always rebuild modules_to_save paths after encoder modifications
    modules_to_save = _maybe_refresh_modules_to_save_paths(model, peft_enc)

    # ------------------------
    # PEFT (LoRA/DoRA/AdaLoRA) targets
    # ------------------------
    lora_targets: List[str] = []
    enc_lora_targets: List[str] = []
    dec_lora_targets: List[str] = []
    vera_targets: List[str] = []

    if enc_lora or enc_dora or enc_adalora:
        enable_vision_grad = True
        enc_lora_targets.extend(_lora_target_modules(include_decoder=False, include_encoder=True))
    if enc_vera:
        enable_vision_grad = True
        vera_targets.extend(_lora_target_modules(include_decoder=False, include_encoder=True))

    if peft_dec in {"lora", "dora", "adalora"}:
        dec_lora_targets.extend(_lora_target_modules(include_decoder=True, include_encoder=False))
    if dec_vera:
        vera_targets.extend(_lora_target_modules(include_decoder=True, include_encoder=False))

    if peft_dec == "adalora" or enc_adalora:
        need_adalora_setup = True
        lora_targets.extend(enc_lora_targets)
        lora_targets.extend(dec_lora_targets)
        args._lora_target_modules = sorted(set(lora_targets))
    elif enc_lora_targets or dec_lora_targets:
        mixed_lora_dora = (enc_dora and dec_lora) or (enc_lora and dec_dora)
        if mixed_lora_dora and enc_lora_targets and dec_lora_targets:
            model, enc_adapter = _apply_lora(
                model,
                args,
                enc_lora_targets,
                use_dora=enc_dora,
                modules_to_save=modules_to_save,
                adapter_name="encoder",
            )
            model, dec_adapter = _apply_lora(
                model,
                args,
                dec_lora_targets,
                use_dora=dec_dora,
                modules_to_save=modules_to_save,
                adapter_name="decoder",
            )
            _set_active_adapters(model, [enc_adapter, dec_adapter])
        else:
            lora_targets.extend(enc_lora_targets)
            lora_targets.extend(dec_lora_targets)
            model, adapter_name = _apply_lora(
                model,
                args,
                lora_targets,
                use_dora=(enc_dora or dec_dora),
                modules_to_save=modules_to_save,
            )
    elif vera_targets:
        model = _apply_vera(
            model,
            args,
            vera_targets,
            modules_to_save=modules_to_save,
            task_type="CAUSAL_LM",
        )

    # Decoder shared adapter variants (your custom DecoderAdapterLinear path)
    if peft_dec in DECODER_SHARED_ADAPTERS:
        model = install_qwen_decoder_shared_adapter(model, args, peft_dec)
    elif peft_dec not in {"none", "frozen", "lora", "dora", "adalora", "vera"}:
        raise ValueError(f"Unknown decoder PEFT type: {args.peft_dec}")

    # Finally, enforce frozen/none on base towers if requested
    _set_encoder_train_state(model, peft_enc)
    _set_decoder_train_state(model, peft_dec)
    _ensure_qwen_vision_encoder_adapters_trainable(model)

    # If vision adapters exist, Qwen may require this for gradient flow through vision inputs
    if enable_vision_grad:
        setattr(model, "_force_vision_input_grads", True)

    model._vision_adapter_name = peft_enc
    return model, need_adalora_setup


# -----------------------------------------------------------------------------
# AdaLoRA finalization (unchanged; note: modules_to_save not used here)
# -----------------------------------------------------------------------------

def finalize_adalora(model, args, total_steps: int):
    target_modules = getattr(args, "_lora_target_modules", None)
    if not target_modules:
        target_modules = _lora_target_modules(include_decoder=True, include_encoder=False)

    peft_cfg = AdaLoraConfig(
        init_r=args.adalora_r,
        target_r=args.adalora_r,
        tinit=args.adalora_tinit,
        tfinal=args.adalora_tfinal,
        deltaT=args.adalora_deltaT,
        beta1=0.85,
        beta2=0.85,
        orth_reg_weight=0.5,
        total_step=total_steps,
        lora_alpha=args.adapter_alpha,
        lora_dropout=args.adapter_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    # Re-ensure custom encoder wrapper adapters (if present) are still trainable after AdaLoRA wrap
    _mark_modules_trainable(model, getattr(model, "_modules_to_save_paths", None))
    _ensure_qwen_vision_encoder_adapters_trainable(model)

    return model
