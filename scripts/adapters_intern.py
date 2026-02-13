import inspect
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import torch
import torch.nn as nn
from peft import AdaLoraConfig, LoraConfig, get_peft_model
from peft import VeraConfig


from temporal_modules import (
    GatedAttentionWrapper,
    HeadwiseGate,
    LayerwiseGate,
    PreprocessorWrapper,
    PreprocessorWrapperAfter,
    DecoderAdapterLinear,
    SpatioTemporalAdapterModule,
    STAttentionLoRAVisionLinear,
    STLSTMLoRAVisionLinear,
    STLoRAVisionLinear,
    STLoRA3DInline,
    STLSTMLoRAInline,
    STMambaLoRAInline,
    STMambaVisionLinear,
    STMultiHeadAttentionLoRAInline,
    STSelfAttentionLoRAInline,
    TimeShiftLoRA,
    TimeShiftedVisionLinear,
)

ENCODER_WRAPPER_TYPES = (
    STLoRAVisionLinear,
    STLSTMLoRAVisionLinear,
    STAttentionLoRAVisionLinear,
    STMambaVisionLinear,
    TimeShiftedVisionLinear,
)

ENCODER_INNER_ATTR_CANDIDATES = ("stlora", "adapter")


def _is_peft_wrapper(model: nn.Module) -> bool:
    return hasattr(model, "peft_config")


def _resolve_internvl_base(model: nn.Module) -> nn.Module:
    base_model = model
    get_base = getattr(base_model, "get_base_model", None)
    if callable(get_base) and _is_peft_wrapper(base_model):
        base_model = get_base()
    wrapped = getattr(base_model, "base_model", None)
    if wrapped is not None:
        inner = getattr(wrapped, "model", None)
        base_model = inner if inner is not None else wrapped
    inner_model = getattr(base_model, "model", None)
    if inner_model is not None and inner_model is not base_model:
        base_model = inner_model
    return base_model


def _require_intern_vision_model(model: nn.Module) -> nn.Module:
    # Prefer the top-level model to avoid base_model_prefix resolving to language_model.
    candidates = [model]
    base_model = _resolve_internvl_base(model)
    if base_model is not model:
        candidates.append(base_model)
    wrapped = getattr(model, "base_model", None)
    if wrapped is not None:
        candidates.append(wrapped)
        inner = getattr(wrapped, "model", None)
        if inner is not None:
            candidates.append(inner)

    for candidate in candidates:
        vision_model = getattr(candidate, "vision_model", None)
        if vision_model is not None and hasattr(vision_model, "encoder"):
            return vision_model

    raise AttributeError(
        "[Adapter][InternVL] Expected vision_model.encoder on InternVLChatModel."
    )


def _require_intern_language_model(model: nn.Module) -> nn.Module:
    # Prefer the top-level model to avoid base_model_prefix resolving to language_model.
    candidates = [model]
    base_model = _resolve_internvl_base(model)
    if base_model is not model:
        candidates.append(base_model)
    wrapped = getattr(model, "base_model", None)
    if wrapped is not None:
        candidates.append(wrapped)
        inner = getattr(wrapped, "model", None)
        if inner is not None:
            candidates.append(inner)

    for candidate in candidates:
        language_model = getattr(candidate, "language_model", None)
        if language_model is not None:
            return language_model

    raise AttributeError(
        "[Adapter][InternVL] Expected language_model on InternVLChatModel."
    )


def _qualified_name(root: nn.Module, target: nn.Module) -> str:
    """Return module path (as used by get_submodule) for target inside root."""
    for name, mod in root.named_modules():
        if mod is target:
            return name
    raise ValueError(
        f"Target module {type(target).__name__} not found under root {type(root).__name__}"
    )


def _join_module_path(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix


def _locate_mlp1_owner(model: nn.Module) -> Tuple[nn.Module, str, nn.Module]:
    """
    Return (owner_module, attr_name, mlp1_module) even if model is PEFT-wrapped.
    """
    candidates = [model]
    base = _resolve_internvl_base(model)
    if base is not model:
        candidates.append(base)

    wrapped = getattr(model, "base_model", None)
    if isinstance(wrapped, nn.Module):
        candidates.append(wrapped)
        inner = getattr(wrapped, "model", None)
        if isinstance(inner, nn.Module):
            candidates.append(inner)

    for owner in candidates:
        mlp1 = getattr(owner, "mlp1", None)
        if isinstance(mlp1, nn.Module):
            return owner, "mlp1", mlp1

    for name, mod in model.named_modules():
        if name.endswith(".mlp1") and isinstance(mod, nn.Module):
            owner_path = name.rsplit(".", 1)[0]
            owner = model.get_submodule(owner_path) if owner_path else model
            return owner, "mlp1", mod

    raise AttributeError(
        "[Adapter][InternVL] Could not locate projector mlp1 (even though architecture shows it)."
    )


class AdapterStore(nn.Module):
    """Container to register adapter modules for PEFT modules_to_save."""

    def __init__(self, modules: List[nn.Module]):
        super().__init__()
        self.adapters = nn.ModuleList(modules)




def disable_vision_checkpointing(model: nn.Module) -> None:
    vision_model = _require_intern_vision_model(model)
    encoder = getattr(vision_model, "encoder", None)
    if encoder is None:
        raise AttributeError("[Adapter][InternVL] vision_model missing encoder.")
    if getattr(encoder, "gradient_checkpointing", False):
        encoder.gradient_checkpointing = False
        print("[Adapter] Disabled vision encoder gradient checkpointing to allow adapter grads.")


def _infer_projector_dims(model) -> Tuple[int, int]:
    _, _, mlp = _locate_mlp1_owner(model)

    if isinstance(mlp, nn.Linear):
        return int(mlp.in_features), int(mlp.out_features)

    linears: List[nn.Linear] = []
    for module in mlp.modules():
        if module is mlp:
            continue
        if isinstance(module, nn.Linear):
            linears.append(module)

    if not linears:
        raise TypeError("[Vision-Adapter] Could not locate Linear layers inside mlp1.")

    first = linears[0]
    last = linears[-1]
    return int(first.in_features), int(last.out_features)


def _resolve_projector_dims(
    model,
    hidden_override: Optional[int] = None,
    out_override: Optional[int] = None,
) -> Tuple[int, int]:
    inferred_in, inferred_out = _infer_projector_dims(model)
    if hidden_override is not None and hidden_override > 0 and hidden_override != inferred_in:
        raise ValueError(
            f"[Vision-Adapter] Provided hidden size {hidden_override} "
            f"does not match projector input dim {inferred_in}."
        )
    if out_override is not None and out_override > 0 and out_override != inferred_out:
        raise ValueError(
            f"[Vision-Adapter] Provided output size {out_override} "
            f"does not match projector output dim {inferred_out}."
        )
    return inferred_in, inferred_out


def _wrap_projector_adapter(
    model: nn.Module,
    preprocess: Callable[[torch.Tensor], torch.Tensor],
    after: bool,
    output_dim: Optional[int] = None,
) -> None:
    owner, attr, base_module = _locate_mlp1_owner(model)
    wrapped = (
        PreprocessorWrapperAfter(preprocess, base_module, expected_output_dim=output_dim)
        if after
        else PreprocessorWrapper(preprocess, base_module)
    )
    setattr(owner, attr, wrapped)


def _make_attention_module(
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
    kwargs = dict(
        hidden_size=hidden_dim,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        debug=debug,
        use_dora=use_dora,
    )
    if out_size is not None:
        kwargs["out_size"] = out_size
    if issubclass(attention_cls, STMultiHeadAttentionLoRAInline) and num_heads is not None:
        kwargs["num_heads"] = num_heads
    return attention_cls(**kwargs)


def _describe_attention_cls(attention_cls: Type[nn.Module]) -> str:
    if attention_cls is STSelfAttentionLoRAInline:
        return "ST-SelfAttn"
    if attention_cls is STMultiHeadAttentionLoRAInline:
        return "ST-MHAttn"
    return attention_cls.__name__


def _select_attention_cls(adapter_name: str) -> Type[nn.Module]:
    adapter_name = adapter_name.lower()
    if "mhatt" in adapter_name:
        return STMultiHeadAttentionLoRAInline
    if "selfatt" in adapter_name:
        return STSelfAttentionLoRAInline
    return STMultiHeadAttentionLoRAInline


def install_intern_timeshift_lora_encoder(
    model,
    rank=32,
    alpha=32.0,
    n_div=3,
    dropout=0.1,
    wrap_proj: bool = True,
    num_frames: int = 8,
    use_dora: bool = False,
):
    vision_model = _require_intern_vision_model(model)
    vision_encoder = vision_model.encoder
    vision_prefix = _qualified_name(model, vision_model)
    prefix_root = _join_module_path(vision_prefix, "encoder")
    prefix_root = _join_module_path(vision_prefix, "encoder")
    prefix_root = _join_module_path(vision_prefix, "encoder")
    prefix_root = _join_module_path(vision_prefix, "encoder")
    prefix_root = _join_module_path(vision_prefix, "encoder")
    prefix_root = _join_module_path(vision_prefix, "encoder")
    vision_prefix = _qualified_name(model, vision_model)
    vision_prefix = _qualified_name(model, vision_model)
    vision_prefix = _qualified_name(model, vision_model)
    vision_prefix = _qualified_name(model, vision_model)
    count_qkv = 0
    count_proj = 0
    modules_to_save: List[str] = []

    for i, layer in enumerate(vision_encoder.layers):
        if not hasattr(layer, "attn"):
            raise AttributeError(f"Layer {i} missing attn module.")

        original_qkv = layer.attn.qkv
        wrapped_qkv = TimeShiftedVisionLinear(
            original_qkv,
            rank=rank,
            alpha=alpha,
            n_div=n_div,
            dropout=dropout,
            num_frames=num_frames,
            use_dora=use_dora,
        ).to(original_qkv.weight.device, dtype=original_qkv.weight.dtype)
        wrapped_qkv.layer_type = f"Layer{i}_QKV_TimeShiftLoRA"

        layer.attn.qkv = wrapped_qkv
        modules_to_save.append(f"{prefix_root}.layers.{i}.attn.qkv.adapter")
        count_qkv += 1

        if wrap_proj:
            original_proj = layer.attn.proj
            wrapped_proj = TimeShiftedVisionLinear(
                original_proj,
                rank=rank,
                alpha=alpha,
                n_div=n_div,
                dropout=dropout,
                num_frames=num_frames,
                use_dora=use_dora,
            ).to(original_proj.weight.device, dtype=original_proj.weight.dtype)
            wrapped_proj.layer_type = f"Layer{i}_Proj_TimeShiftLoRA"

            layer.attn.proj = wrapped_proj
            modules_to_save.append(f"{prefix_root}.layers.{i}.attn.proj.adapter")
            count_proj += 1

    model._modules_to_save_paths = modules_to_save

    print(
        f"[TimeShift-{'DoRA' if use_dora else 'LoRA'}-Enc] Injected TimeShift adapter into {count_qkv} vision QKV layers "
        f"and {count_proj} vision proj layers."
    )


def install_intern_st_lora_encoder(
    model,
    rank=32,
    alpha=128.0,
    k=3,
    dropout=0.1,
    debug: bool = False,
    wrap_proj: bool = True,
    use_dora: bool = False,
):
    vision_model = _require_intern_vision_model(model)
    vision_encoder = vision_model.encoder
    vision_prefix = _qualified_name(model, vision_model)
    prefix_root = _join_module_path(vision_prefix, "encoder")
    count_qkv = 0
    count_proj = 0
    modules_to_save: List[str] = []

    for i, layer in enumerate(vision_encoder.layers):
        if not hasattr(layer, "attn"):
            raise AttributeError(f"Layer {i} missing attn module.")

        original_qkv = layer.attn.qkv
        wrapped_qkv = STLoRAVisionLinear(
            original_qkv,
            rank=rank,
            alpha=alpha,
            k=k,
            dropout=dropout,
            debug=debug,
            use_dora=use_dora,
        ).to(original_qkv.weight.device, dtype=original_qkv.weight.dtype)
        wrapped_qkv.layer_type = f"Layer{i}_QKV_STLoRA"

        layer.attn.qkv = wrapped_qkv
        modules_to_save.append(f"{prefix_root}.layers.{i}.attn.qkv.stlora")
        count_qkv += 1

        if wrap_proj:
            original_proj = layer.attn.proj
            wrapped_proj = STLoRAVisionLinear(
                original_proj,
                rank=rank,
                alpha=alpha,
                k=k,
                dropout=dropout,
                debug=debug,
                use_dora=use_dora,
            ).to(original_proj.weight.device, dtype=original_proj.weight.dtype)
            wrapped_proj.layer_type = f"Layer{i}_Proj_STLoRA"

            layer.attn.proj = wrapped_proj
            modules_to_save.append(f"{prefix_root}.layers.{i}.attn.proj.stlora")
            count_proj += 1

    model._modules_to_save_paths = modules_to_save

    tag = "ST-DoRA" if use_dora else "ST-LoRA"
    print(
        f"[{tag}-Enc] Injected {tag} into {count_qkv} vision QKV layers "
        f"and {count_proj} vision proj layers."
    )
    print(f"[{tag}-Enc] Params: rank={rank}, alpha={alpha}, k={k}, dropout={dropout}")


def install_intern_stlstm_lora_encoder(
    model,
    rank: int = 32,
    alpha: float = 128.0,
    dropout: float = 0.1,
    num_layers: int = 1,
    bidirectional: bool = False,
    debug: bool = False,
    wrap_proj: bool = True,
    use_dora: bool = False,
):
    vision_model = _require_intern_vision_model(model)
    vision_encoder = vision_model.encoder
    vision_prefix = _qualified_name(model, vision_model)
    prefix_root = _join_module_path(vision_prefix, "encoder")
    count_qkv = 0
    count_proj = 0
    modules_to_save: List[str] = []

    for i, layer in enumerate(vision_encoder.layers):
        if not hasattr(layer, "attn"):
            raise AttributeError(f"[ST-LSTM-Enc] Encoder layer {i} missing attn module.")

        original_qkv = layer.attn.qkv
        if not isinstance(original_qkv, nn.Linear):
            raise TypeError(
                f"[ST-LSTM-Enc] Expected Linear for attn.qkv in layer {i}, got {type(original_qkv)}."
            )

        wrapped_qkv = STLSTMLoRAVisionLinear(
            original_qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            num_layers=num_layers,
            bidirectional=bidirectional,
            debug=debug,
            use_dora=use_dora,
        ).to(original_qkv.weight.device, dtype=original_qkv.weight.dtype)
        wrapped_qkv.layer_type = f"Layer{i}_QKV_STLSTM"

        layer.attn.qkv = wrapped_qkv
        modules_to_save.append(f"{prefix_root}.layers.{i}.attn.qkv.adapter")
        count_qkv += 1

        if wrap_proj:
            original_proj = layer.attn.proj
            if not isinstance(original_proj, nn.Linear):
                raise TypeError(
                    f"[ST-LSTM-Enc] Expected Linear for attn.proj in layer {i}, got {type(original_proj)}."
                )
            wrapped_proj = STLSTMLoRAVisionLinear(
                original_proj,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                num_layers=num_layers,
                bidirectional=bidirectional,
                debug=debug,
                use_dora=use_dora,
            ).to(original_proj.weight.device, dtype=original_proj.weight.dtype)
            wrapped_proj.layer_type = f"Layer{i}_Proj_STLSTM"

            layer.attn.proj = wrapped_proj
            modules_to_save.append(f"{prefix_root}.layers.{i}.attn.proj.adapter")
            count_proj += 1

    model._modules_to_save_paths = modules_to_save

    tag = "STLSTM-DoRA" if use_dora else "STLSTM-LoRA"
    print(
        f"[ST-LSTM-Enc] Injected {tag} into {count_qkv} vision QKV layers "
        f"and {count_proj} vision proj layers."
    )


def install_intern_st_attention_encoder(
    model,
    rank: int = 32,
    alpha: float = 128.0,
    dropout: float = 0.1,
    num_heads: int = 4,
    debug: bool = False,
    attention_cls: Type[nn.Module] = STMultiHeadAttentionLoRAInline,
    adapter_name: Optional[str] = None,
    wrap_proj: bool = True,
    use_dora: bool = False,
):
    vision_model = _require_intern_vision_model(model)
    vision_encoder = vision_model.encoder
    vision_prefix = _qualified_name(model, vision_model)
    prefix_root = _join_module_path(vision_prefix, "encoder")
    count_qkv = 0
    count_proj = 0
    modules_to_save: List[str] = []

    for i, layer in enumerate(vision_encoder.layers):
        if not hasattr(layer, "attn"):
            raise AttributeError(f"[ST-Attn-Enc] Encoder layer {i} missing attn module.")

        original_qkv = layer.attn.qkv
        if not isinstance(original_qkv, nn.Linear):
            raise TypeError(
                f"[ST-Attn-Enc] Expected Linear for attn.qkv in layer {i}, got {type(original_qkv)}."
            )

        wrapped_qkv = STAttentionLoRAVisionLinear(
            original_qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            num_heads=num_heads,
            debug=debug,
            attention_cls=attention_cls,
            use_dora=use_dora,
        ).to(device=original_qkv.weight.device, dtype=original_qkv.weight.dtype)
        wrapped_qkv.layer_type = f"Layer{i}_QKV_STAttention"

        layer.attn.qkv = wrapped_qkv
        modules_to_save.append(f"{prefix_root}.layers.{i}.attn.qkv.adapter")
        count_qkv += 1

        if wrap_proj:
            original_proj = layer.attn.proj
            if not isinstance(original_proj, nn.Linear):
                raise TypeError(
                    f"[ST-Attn-Enc] Expected Linear for attn.proj in layer {i}, got {type(original_proj)}."
                )
            wrapped_proj = STAttentionLoRAVisionLinear(
                original_proj,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                num_heads=num_heads,
                debug=debug,
                attention_cls=attention_cls,
                use_dora=use_dora,
            ).to(device=original_proj.weight.device, dtype=original_proj.weight.dtype)
            wrapped_proj.layer_type = f"Layer{i}_Proj_STAttention"

            layer.attn.proj = wrapped_proj
            modules_to_save.append(f"{prefix_root}.layers.{i}.attn.proj.adapter")
            count_proj += 1

    model._modules_to_save_paths = modules_to_save

    head_info = (
        f", heads={num_heads}"
        if issubclass(attention_cls, STMultiHeadAttentionLoRAInline)
        else ""
    )
    tag = ("STDora-" if use_dora else "") + _describe_attention_cls(attention_cls)
    print(
        f"[{tag}-Enc] Injected adapters into {count_qkv} vision QKV layers "
        f"and {count_proj} vision proj layers."
        f"(rank={rank}, alpha={alpha}, dropout={dropout}{head_info})"
    )


def install_intern_st_mamba_encoder(
    model,
    rank: int = 32,
    alpha: float = 128.0,
    dropout: float = 0.1,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    debug: bool = False,
    wrap_proj: bool = True,
    use_dora: bool = False,
):
    vision_model = _require_intern_vision_model(model)
    vision_encoder = vision_model.encoder
    vision_prefix = _qualified_name(model, vision_model)
    prefix_root = _join_module_path(vision_prefix, "encoder")
    count_qkv = 0
    count_proj = 0
    modules_to_save: List[str] = []

    for i, layer in enumerate(vision_encoder.layers):
        if not hasattr(layer, "attn"):
            raise AttributeError(f"[ST-Mamba-Enc] Encoder layer {i} missing attn module.")

        original_qkv = layer.attn.qkv
        if not isinstance(original_qkv, nn.Linear):
            raise TypeError(
                f"[ST-Mamba-Enc] Expected Linear for attn.qkv in layer {i}, got {type(original_qkv)}."
            )

        wrapped_qkv = STMambaVisionLinear(
            original_qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            debug=debug,
            use_dora=use_dora,
        ).to(original_qkv.weight.device, dtype=original_qkv.weight.dtype)
        wrapped_qkv.layer_type = f"Layer{i}_QKV_STMamba"

        layer.attn.qkv = wrapped_qkv
        modules_to_save.append(f"{prefix_root}.layers.{i}.attn.qkv.adapter")
        count_qkv += 1

        if wrap_proj:
            original_proj = layer.attn.proj
            if not isinstance(original_proj, nn.Linear):
                raise TypeError(
                    f"[ST-Mamba-Enc] Expected Linear for attn.proj in layer {i}, got {type(original_proj)}."
                )

            wrapped_proj = STMambaVisionLinear(
                original_proj,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                debug=debug,
                use_dora=use_dora,
            ).to(original_proj.weight.device, dtype=original_proj.weight.dtype)
            wrapped_proj.layer_type = f"Layer{i}_Proj_STMamba"

            layer.attn.proj = wrapped_proj
            modules_to_save.append(f"{prefix_root}.layers.{i}.attn.proj.adapter")
            count_proj += 1

    model._modules_to_save_paths = modules_to_save

    tag = "STMamba-DoRA" if use_dora else "STMamba"
    print(
        f"[ST-Mamba-Enc] Injected {tag} into {count_qkv} vision QKV layers "
        f"and {count_proj} vision proj layers."
        f"(rank={rank}, alpha={alpha}, dropout={dropout})"
    )


def _projector_preprocess_shape_check(x: torch.Tensor, hidden_dim: int, num_frames: int, tag: str) -> int:
    if not num_frames:
        raise ValueError(f"[{tag}] num_frames has not been configured.")
    if x.dim() != 3:
        raise ValueError(f"[{tag}] Expected (B, L, C); got {tuple(x.shape)}")
    if x.size(-1) != hidden_dim:
        raise ValueError(
            f"[{tag}] Expected projector input dim {hidden_dim}, got {x.size(-1)}."
        )
    if x.size(1) % num_frames != 0:
        raise ValueError(
            f"[{tag}] Token length {x.size(1)} not divisible by frames {num_frames}"
        )
    return x.size(1) // num_frames


def install_intern_st_adapter(
    model,
    hidden_size: Optional[int] = None,
    adapter_channels=384,
    kernel: Tuple[int, int, int] = (3, 1, 1),
    disable_cudnn: bool = False,
    num_frames: Optional[int] = None,
):
    hidden_dim, _ = _resolve_projector_dims(model, hidden_override=hidden_size)

    ref_param = next(model.parameters())
    adapter_module = SpatioTemporalAdapterModule(
        hidden_size=hidden_dim,
        adapter_channels=adapter_channels,
        kernel_size=kernel,
        disable_cudnn=disable_cudnn,
    ).to(device=ref_param.device, dtype=ref_param.dtype)
    for p in adapter_module.parameters():
        p.requires_grad = True

    def preprocess(x: torch.Tensor) -> torch.Tensor:
        num_frames = model.stadapter_num_frames
        tokens_per_frame = _projector_preprocess_shape_check(x, hidden_dim, num_frames, "ST-Adapter")
        x_bt = x.reshape(x.size(0), num_frames, tokens_per_frame, hidden_dim)
        x_bt = adapter_module(x_bt, num_frames=num_frames)
        x_bt = x_bt.reshape(x.size(0), x.size(1), hidden_dim)
        if x_bt.shape != x.shape:
            raise ValueError(
                f"[ST-Adapter] Residual shape {tuple(x_bt.shape)} does not match projector input {tuple(x.shape)}."
            )
        return x_bt

    _wrap_projector_adapter(model, preprocess, after=False)
    model.stadapter_module = adapter_module
    model.stadapter_num_frames = int(num_frames) if num_frames is not None else 8

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[ST-Adapter] Installed before mlp1. Trainable={trainable:,} / Total={total:,} "
        f"(hidden={hidden_dim}, channels={adapter_channels}, kernel={kernel})"
    )


def install_intern_gated_attention(model, gate_type: str = "headwise", wrap_encoder: bool = False) -> None:
    if gate_type == "headwise":
        gate_cls = HeadwiseGate
    elif gate_type == "layerwise":
        gate_cls = LayerwiseGate
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")

    language_model = _require_intern_language_model(model)
    layers = language_model.model.layers
    wrapped = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise AttributeError("[GatedAttn] Missing self_attn in decoder layer.")
        attn.q_proj = GatedAttentionWrapper(attn.q_proj, gate_cls())
        attn.k_proj = GatedAttentionWrapper(attn.k_proj, gate_cls())
        attn.v_proj = GatedAttentionWrapper(attn.v_proj, gate_cls())
        attn.o_proj = GatedAttentionWrapper(attn.o_proj, gate_cls())
        wrapped += 1

    if wrap_encoder:
        vision_model = _require_intern_vision_model(model)
        for layer in vision_model.encoder.layers:
            attn = getattr(layer, "attn", None)
            if attn is None:
                raise AttributeError("[GatedAttn] Missing attn in vision encoder layer.")
            attn.qkv = GatedAttentionWrapper(attn.qkv, gate_cls())
            attn.proj = GatedAttentionWrapper(attn.proj, gate_cls())

    print(f"[GatedAttn] Installed {gate_type} gates on {wrapped} decoder layers.")


def _build_decoder_adapter_module(
    base_layer: nn.Linear,
    adapter_name: str,
    args,
) -> nn.Module:
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
        attention_cls = _select_attention_cls(adapter_lower)
        num_heads = max(4, rank // 4) if issubclass(attention_cls, STMultiHeadAttentionLoRAInline) else 1
        module = _make_attention_module(
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


def install_decoder_shared_adapter(model, args, adapter_name: str):
    language_model = _require_intern_language_model(model)
    if not hasattr(language_model, "model"):
        raise AttributeError("[Decoder-Adapter] Model is missing language_model.model.")
    lm_inner = language_model.model
    layers = getattr(lm_inner, "layers", None)
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
                raise TypeError(
                    f"[Decoder-Adapter] Expected nn.Linear for {attr}, got {type(base_layer)}."
                )
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


def configure_adapter_clip_length(model, clip_length: int) -> None:
    if clip_length <= 0:
        raise ValueError("clip_length must be positive.")
    if hasattr(model, "stlora_module"):
        model.stlora_num_frames = clip_length
    if hasattr(model, "stadapter_module"):
        model.stadapter_num_frames = clip_length
    if hasattr(model, "timeshift_module"):
        model.timeshift_num_frames = clip_length

    vision_model = _require_intern_vision_model(model)
    layers = getattr(vision_model.encoder, "layers", None)
    if layers is None:
        raise AttributeError("[Adapter][InternVL] vision_model.encoder has no layers.")
    for layer in layers:
        if not hasattr(layer, "attn"):
            raise AttributeError("[Adapter][InternVL] Missing attn in vision encoder layer.")
        for attr in ("qkv", "proj"):
            module = getattr(layer.attn, attr, None)
            if isinstance(
                module,
                (
                    STLoRAVisionLinear,
                    STLSTMLoRAVisionLinear,
                    STMambaVisionLinear,
                    STAttentionLoRAVisionLinear,
                    TimeShiftedVisionLinear,
                ),
            ):
                module.num_frames = clip_length


def _decoder_lora_target_modules() -> List[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def _modules_to_save_for_peft_enc(peft_enc: str) -> Optional[List[str]]:
    return None


def _collect_intern_encoder_modules_to_save(model: nn.Module) -> List[str]:
    """
    Collect encoder adapter module paths so PEFT save_pretrained includes them.
    """
    vision_model = _require_intern_vision_model(model)
    vision_prefix = _qualified_name(model, vision_model)
    prefix_root = _join_module_path(vision_prefix, "encoder")
    layers = getattr(vision_model.encoder, "layers", None)
    if layers is None:
        raise AttributeError("[Adapter][InternVL] vision_model.encoder has no layers.")
    if len(layers) == 0:
        raise RuntimeError("[Adapter][InternVL] vision_model.encoder has no layers.")

    def _collect_wrapper_paths(wrapper: nn.Module, prefix: str) -> List[str]:
        paths: List[str] = []
        if isinstance(wrapper, GatedAttentionWrapper):
            gate = getattr(wrapper, "gate", None)
            if isinstance(gate, nn.Module):
                paths.append(f"{prefix}.gate")
            base_attn = getattr(wrapper, "base_attn", None)
            if isinstance(base_attn, nn.Module):
                paths.extend(_collect_wrapper_paths(base_attn, f"{prefix}.base_attn"))
            return paths

        if isinstance(wrapper, ENCODER_WRAPPER_TYPES):
            for inner_attr in ENCODER_INNER_ATTR_CANDIDATES:
                inner = getattr(wrapper, inner_attr, None)
                if isinstance(inner, nn.Module):
                    paths.append(f"{prefix}.{inner_attr}")
        return paths

    paths: List[str] = []
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "attn", None)
        if attn is None:
            raise AttributeError(f"[Adapter][InternVL] Missing attn in vision layer {idx}.")
        for proj_attr in ("qkv", "proj"):
            wrapper = getattr(attn, proj_attr, None)
            if isinstance(wrapper, nn.Module):
                prefix = f"{prefix_root}.layers.{idx}.attn.{proj_attr}"
                paths.extend(_collect_wrapper_paths(wrapper, prefix))

    return sorted(set(paths))


def _collect_intern_decoder_modules_to_save(model: nn.Module) -> List[str]:
    """
    Collect decoder adapter/gating module paths so PEFT save_pretrained includes them.
    """
    language_model = _require_intern_language_model(model)
    lm_prefix = _qualified_name(model, language_model)
    text_model = getattr(language_model, "model", None)
    if text_model is None:
        raise AttributeError("[Adapter][InternVL] language_model missing .model.")
    layers = getattr(text_model, "layers", None)
    if layers is None:
        raise AttributeError("[Adapter][InternVL] language_model.model has no layers.")

    prefix_root = _join_module_path(lm_prefix, "model.layers")
    paths: List[str] = []
    for idx, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            raise AttributeError(f"[Adapter][InternVL] Missing self_attn in decoder layer {idx}.")
        for proj_attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = getattr(self_attn, proj_attr, None)
            if isinstance(module, GatedAttentionWrapper):
                paths.append(f"{prefix_root}.{idx}.self_attn.{proj_attr}.gate")
            if isinstance(module, DecoderAdapterLinear):
                paths.append(f"{prefix_root}.{idx}.self_attn.{proj_attr}.adapter")

    return sorted(set(paths))


def _maybe_refresh_modules_to_save_paths(model: nn.Module, peft_enc: str) -> List[str]:
    modules_to_save: List[str] = []
    base = _modules_to_save_for_peft_enc(peft_enc)
    if base:
        modules_to_save.extend(base)

    modules_to_save.extend(_collect_intern_encoder_modules_to_save(model))
    modules_to_save.extend(_collect_intern_decoder_modules_to_save(model))

    existing = getattr(model, "_modules_to_save_paths", None)
    if existing:
        modules_to_save.extend(existing)

    modules_to_save = sorted(set(modules_to_save))
    setattr(model, "_modules_to_save_paths", modules_to_save)
    return modules_to_save


def _mark_modules_trainable(model: nn.Module, modules_to_save: Optional[List[str]]) -> None:
    if not modules_to_save:
        return
    for module_name in modules_to_save:
        try:
            module = model.get_submodule(module_name)
        except AttributeError as exc:
            raise AttributeError(f"Missing module '{module_name}' in model") from exc
        modules_to_save_dict = getattr(module, "modules_to_save", None)
        original_module = getattr(module, "original_module", None)
        if isinstance(modules_to_save_dict, nn.ModuleDict):
            for param in modules_to_save_dict.parameters():
                param.requires_grad = True
            if original_module is not None:
                for param in original_module.parameters():
                    param.requires_grad = False
            continue
        for param in module.parameters():
            param.requires_grad = True


def _ensure_vision_encoder_adapters_trainable(model: nn.Module) -> bool:
    vision_model = _require_intern_vision_model(model)
    layers = getattr(vision_model.encoder, "layers", None)
    if layers is None:
        raise AttributeError("[Adapter][InternVL] vision_model.encoder has no layers.")
    if len(layers) == 0:
        raise RuntimeError("[Adapter][InternVL] vision_model.encoder has no layers.")

    updated = False
    for layer in layers:
        attn = getattr(layer, "attn", None)
        if attn is None:
            raise AttributeError("[Adapter][InternVL] Missing attn in vision encoder layer.")
        for attr in ("qkv", "proj"):
            module = getattr(attn, attr, None)
            if not isinstance(
                module,
                (
                    STLoRAVisionLinear,
                    STLSTMLoRAVisionLinear,
                    STMambaVisionLinear,
                    STAttentionLoRAVisionLinear,
                    TimeShiftedVisionLinear,
                ),
            ):
                continue
            found_inner = False
            for name in ("adapter", "stlora"):
                inner = getattr(module, name, None)
                if inner is None:
                    continue
                found_inner = True
                modules_to_save_dict = getattr(inner, "modules_to_save", None)
                original_module = getattr(inner, "original_module", None)
                if isinstance(modules_to_save_dict, nn.ModuleDict):
                    for param in modules_to_save_dict.parameters():
                        if not param.requires_grad:
                            param.requires_grad = True
                            updated = True
                    if original_module is not None:
                        for param in original_module.parameters():
                            if param.requires_grad:
                                param.requires_grad = False
                                updated = True
                    continue
                for param in inner.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        updated = True
            if not found_inner:
                raise AttributeError(
                    f"[Adapter][InternVL] Encoder wrapper {type(module).__name__} missing adapter/stlora."
                )
    return updated


def _set_module_requires_grad(module: Optional[nn.Module], requires_grad: bool) -> None:
    if module is None:
        raise AttributeError("[Adapter][InternVL] Expected module for train-state update, got None.")
    for param in module.parameters():
        param.requires_grad = requires_grad


def _set_encoder_train_state(model: nn.Module, peft_enc: str) -> None:
    vision_model = _require_intern_vision_model(model)
    _, _, mlp1 = _locate_mlp1_owner(model)

    if peft_enc == "none":
        _set_module_requires_grad(vision_model, True)
        _set_module_requires_grad(mlp1, True)
    elif peft_enc == "frozen":
        _set_module_requires_grad(vision_model, False)
        _set_module_requires_grad(mlp1, False)


def _set_decoder_train_state(model: nn.Module, peft_dec: str) -> None:
    language_model = _require_intern_language_model(model)
    if peft_dec == "none":
        _set_module_requires_grad(language_model, True)
    elif peft_dec == "frozen":
        _set_module_requires_grad(language_model, False)


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
    target_modules = sorted(set(target_modules))
    if not target_modules:
        raise ValueError("[Adapter] No target modules provided for LoRA.")

    if modules_to_save is None:
        modules_to_save = getattr(model, "_modules_to_save_paths", None)

    
    lora_cfg = LoraConfig(
        use_dora=use_dora,
        r=args.adapter_rank,
        lora_alpha=args.adapter_alpha,
        lora_dropout=args.adapter_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=None,
        modules_to_save=modules_to_save,
    )

    model, adapter_name = _inject_peft_adapter(model, lora_cfg, adapter_name=adapter_name)

    _mark_modules_trainable(model, modules_to_save)
    _ensure_vision_encoder_adapters_trainable(model)

    if getattr(args, "debug", False):
        trainable_params_in_modules_to_save = 0
        print("[Adapter] Trainable layers:")
        
        for module_name in modules_to_save or []:
            try:
                module = model.get_submodule(module_name)
            except AttributeError as exc:
                raise AttributeError(f"Missing module '{module_name}' in model") from exc
            for param in module.parameters():
                if param.requires_grad:
                    trainable_params_in_modules_to_save += param.numel()
        print(
            f"[Adapter] trainable parameters in modules_to_save={trainable_params_in_modules_to_save}"
        )
        print(f"[Adapter] total parameters={sum(p.numel() for p in model.parameters())}")
        print(
            f"[Adapter] total trainable parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
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
    _ensure_vision_encoder_adapters_trainable(model)

    if getattr(args, "debug", False):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[Adapter][DEBUG] After VeRA: trainable={trainable:,} / total={total:,}")
    return model


def setup_adapter(model, args):
    need_adalora_setup = False
    peft_enc = args.peft_enc.lower()
    peft_dec = args.peft_dec.lower()
    enable_vision_grad = False
    adapter_frames = (
        int(args.frames)
        if getattr(args, "frames", 0) and args.frames > 0
        else int(getattr(args, "adapter_frames", 8))
    )

    for p in model.parameters():
        p.requires_grad = False

    hidden_override = args.st_hidden if args.st_hidden > 0 else None

    if peft_enc in {"gated_headwise", "gated_elementwise"}:
        install_intern_gated_attention(
            model,
            gate_type="headwise" if peft_enc == "gated_headwise" else "layerwise",
            wrap_encoder=True,
        )
        enable_vision_grad = True

    elif peft_enc in {"stlora_enc", "stdora_enc"}:
        install_intern_st_lora_encoder(
            model,
            rank=args.adapter_rank,
            alpha=args.adapter_alpha,
            k=args.st_kernel,
            dropout=args.adapter_dropout,
            wrap_proj=True,
            use_dora=peft_enc.startswith("stdora"),
        )
        enable_vision_grad = True

    elif peft_enc in {"stlora_mhatt_enc", "stlora_selfatt_enc", "stdora_mhatt_enc", "stdora_selfatt_enc"}:
        attention_cls = _select_attention_cls(peft_enc)
        num_heads = max(4, args.adapter_rank // 4)
        install_intern_st_attention_encoder(
            model,
            rank=args.adapter_rank,
            alpha=args.adapter_alpha,
            dropout=args.adapter_dropout,
            num_heads=num_heads,
            attention_cls=attention_cls,
            adapter_name=peft_enc,
            wrap_proj=True,
            use_dora=peft_enc.startswith("stdora"),
        )
        enable_vision_grad = True

    elif peft_enc in {"stlora_lstm_enc", "stdora_lstm_enc"}:
        install_intern_stlstm_lora_encoder(
            model,
            rank=args.adapter_rank,
            alpha=args.adapter_alpha,
            dropout=args.adapter_dropout,
            wrap_proj=True,
            use_dora=peft_enc.startswith("stdora"),
        )
        enable_vision_grad = True

    elif peft_enc in {"stmamba_enc", "stdora_mamba_enc"}:
        install_intern_st_mamba_encoder(
            model,
            rank=args.adapter_rank,
            alpha=args.adapter_alpha,
            dropout=args.adapter_dropout,
            wrap_proj=True,
            use_dora=peft_enc.startswith("stdora"),
        )
        enable_vision_grad = True

    elif peft_enc in {"timeshiftlora_enc", "timeshiftdora_enc"}:
        install_intern_timeshift_lora_encoder(
            model,
            rank=args.adapter_rank,
            alpha=args.adapter_alpha,
            n_div=args.tslora_n_div,
            dropout=args.adapter_dropout,
            wrap_proj=True,
            num_frames=adapter_frames,
            use_dora="dora" in peft_enc,
        )
        enable_vision_grad = True

    elif peft_enc not in {"none", "frozen", "lora_enc", "dora_enc", "vera_enc", "adalora_enc"}:
        raise ValueError(f"Unknown encoder PEFT type: {args.peft_enc}")

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

    modules_to_save = _maybe_refresh_modules_to_save_paths(model, peft_enc)

    lora_targets: List[str] = []
    enc_lora_targets: List[str] = []
    dec_lora_targets: List[str] = []
    vera_targets: List[str] = []
    if enc_lora or enc_dora or enc_adalora:
        enable_vision_grad = True
        if peft_enc == "adalora_enc":
            enc_lora_targets.extend(["qkv", "proj"])
        elif peft_enc.endswith("_enc"):
            enc_lora_targets.extend(["qkv"])
        else:
            projector_targets = _collect_projector_lora_targets(model)
            enc_lora_targets.extend(projector_targets)
    if enc_vera:
        enable_vision_grad = True
        if peft_enc.endswith("_enc"):
            vera_targets.extend(["qkv"])
        else:
            projector_targets = _collect_projector_lora_targets(model)
            vera_targets.extend(projector_targets)

    if peft_dec in {"lora", "dora", "adalora"}:
        dec_lora_targets.extend(_decoder_lora_target_modules())
    if dec_vera:
        vera_targets.extend(_decoder_lora_target_modules())

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
            task_type=None,
        )

    if peft_dec in DECODER_SHARED_ADAPTERS:
        model = install_decoder_shared_adapter(model, args, peft_dec)
    elif peft_dec not in {"none", "frozen", "lora", "dora", "adalora", "vera"}:
        raise ValueError(f"Unknown decoder PEFT type: {args.peft_dec}")

    _set_encoder_train_state(model, peft_enc)
    _set_decoder_train_state(model, peft_dec)
    _ensure_vision_encoder_adapters_trainable(model)

    if enable_vision_grad:
        # Keep vision checkpointing enabled to avoid extra memory use.
        setattr(model, "_force_vision_input_grads", True)

    model._vision_adapter_name = peft_enc
    return model, need_adalora_setup


def finalize_adalora(model, args, total_steps: int):
    tinit = int(args.adalora_tinit)
    tfinal = int(args.adalora_tfinal)
    delta_t = int(args.adalora_deltaT)

    if total_steps <= tinit + tfinal:
        tinit = min(int(0.1 * total_steps), max(0, total_steps - 2))
        tfinal = min(int(0.1 * total_steps), max(0, total_steps - tinit - 1))
        if total_steps <= tinit + tfinal:
            tinit = max(0, total_steps // 4)
            tfinal = max(0, total_steps // 10)
            if total_steps <= tinit + tfinal:
                tfinal = max(0, total_steps - tinit - 1)

    reduction_steps = max(0, total_steps - tinit - tfinal)
    if reduction_steps <= 0:
        if tfinal > 0:
            tfinal = max(0, tfinal - 1)
        elif tinit > 0:
            tinit = max(0, tinit - 1)
        reduction_steps = max(1, total_steps - tinit - tfinal)

    if delta_t > reduction_steps:
        delta_t = max(1, reduction_steps)

    print(
        f"[AdaLoRA] schedule resolved: total_steps={total_steps}, "
        f"tinit={tinit}, tfinal={tfinal}, deltaT={delta_t}, reduction_steps={reduction_steps}"
    )

    target_modules = getattr(args, "_lora_target_modules", None)
    if not target_modules:
        target_modules = _decoder_lora_target_modules()

    peft_cfg = AdaLoraConfig(
        init_r=args.adapter_rank,
        target_r=args.adapter_rank,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=delta_t,
        beta1=0.85,
        beta2=0.85,
        orth_reg_weight=0.5,
        total_step=total_steps,
        lora_alpha=args.adapter_alpha,
        lora_dropout=args.adapter_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=None,
    )
    model = get_peft_model(model, peft_cfg)
    _ensure_vision_encoder_adapters_trainable(model)
    return model


DECODER_SHARED_ADAPTERS: Set[str] = {
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
}
