import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

_QWEN_LAST_GRID_THW: Optional[List[List[int]]] = None
_QWEN_LAST_SPATIAL_MERGE: Optional[int] = None


def set_qwen_visual_context(
    grid_thw: Optional[torch.Tensor],
    spatial_merge_size: Optional[int] = None,
) -> None:
    global _QWEN_LAST_GRID_THW, _QWEN_LAST_SPATIAL_MERGE
    if grid_thw is None:
        _QWEN_LAST_GRID_THW = None
        _QWEN_LAST_SPATIAL_MERGE = spatial_merge_size
        return
    if isinstance(grid_thw, torch.Tensor):
        _QWEN_LAST_GRID_THW = grid_thw.detach().cpu().tolist()
    else:
        _QWEN_LAST_GRID_THW = grid_thw
    _QWEN_LAST_SPATIAL_MERGE = spatial_merge_size


def _get_qwen_visual_context() -> Tuple[Optional[List[List[int]]], Optional[int]]:
    return _QWEN_LAST_GRID_THW, _QWEN_LAST_SPATIAL_MERGE


def _reshape_qwen_2d_tokens(
    x: torch.Tensor,
    num_frames: int,
    tag: str,
) -> Tuple[torch.Tensor, int, int, int]:
    grid_thw, _ = _get_qwen_visual_context()
    if grid_thw:
        first = grid_thw[0]
        if len(first) != 3:
            raise ValueError(f"[{tag}] grid_thw entry must have 3 elements; got {first}")
        t0, h0, w0 = [int(v) for v in first]
        if any(
            len(entry) != 3 or entry[0] != t0 or entry[1] != h0 or entry[2] != w0
            for entry in grid_thw
        ):
            raise ValueError(f"[{tag}] grid_thw entries are not uniform across the batch.")
        p = h0 * w0
        b = len(grid_thw)
        total_tokens = b * t0 * p
        if x.size(0) != total_tokens:
            raise ValueError(
                f"[{tag}] 2D input tokens {x.size(0)} do not match "
                f"B*T*P={total_tokens} (B={b}, T={t0}, P={p})."
            )
        hidden = x.size(1)
        x_b = x.view(b, t0 * p, hidden)
        x_btpc = x_b.view(b, t0, p, hidden)
        return x_btpc, b, t0, p

    t = num_frames
    if t <= 0:
        raise ValueError(f"[{tag}] num_frames must be > 0 for 2D input.")
    seq_len = x.size(0)
    if seq_len % t != 0:
        raise ValueError(
            f"[{tag}] 2D input length {seq_len} not divisible by num_frames={t}."
        )
    p = seq_len // t
    hidden = x.size(1)
    x_btpc = x.view(1, t, p, hidden)
    return x_btpc, 1, t, p


class GatedAttentionWrapper(nn.Module):
    """
    Wraps a base attention module and applies a gate to its output.
    Works both for modules that return a tensor or (tensor, ...).
    """
    def __init__(self, base_attn: nn.Module, gate_module: nn.Module):
        super().__init__()
        self.base_attn = base_attn
        self.gate = gate_module
        

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """
        hidden_states: [B, L, C] input to attention (used as residual/context).
        """
        out = self.base_attn(hidden_states, *args, **kwargs)

        if isinstance(out, tuple):
            attn_out, *rest = out
            gated = self.gate(attn_out, residual=hidden_states)
            return (gated, *rest)

        attn_out = out
        gated = self.gate(attn_out, residual=hidden_states)
        return gated


class HeadwiseGate(nn.Module):
    """
    Content-dependent per-head gate.
    Uses a pooled token representation to produce one gate per head.
    """
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, num_heads)

        # initialize to open gate
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)


    def forward(self, attn_out: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        """
        attn_out: [B, L, C]
        residual: [B, L, C] (input to attention), optional
        """
        B, L, C = attn_out.shape
        H = self.num_heads
        D = self.head_dim

        # Use residual (input) as context if provided, otherwise use attn_out
        context_src = residual if residual is not None else attn_out
        # Simple context: mean over sequence
        context = context_src.mean(dim=1)              # [B, C]

        gates = torch.sigmoid(self.proj(self.norm(context)))  # [B, H]

        attn_out = attn_out.view(B, L, H, D)           # [B, L, H, D]
        gates = gates.view(B, 1, H, 1)                 # [B, 1, H, 1]
        attn_out = attn_out * gates                    # gated

        return attn_out.view(B, L, C)


class LayerwiseGate(nn.Module):
    """
    Content-dependent scalar gate per layer (shared across heads and positions).
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, 1)
        # initialize to open gate
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)


    def forward(self, attn_out: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        """
        attn_out: [B, L, C]
        residual: [B, L, C] (input to attention), optional
        """
        B, L, C = attn_out.shape

        context_src = residual if residual is not None else attn_out
        context = context_src.mean(dim=1)              # [B, C]

        gate = torch.sigmoid(self.proj(self.norm(context)))   # [B, 1]
        gate = gate.view(B, 1, 1)                      # [B, 1, 1]

        return attn_out * gate


class DoRALinear(nn.Linear):
    """
    Linear layer with DoRA-style reparameterization:
      weight = normalize(W) * magnitude
    Magnitude is learnable; initializing it to zeros mirrors LoRA's zero init for the
    up-projection while keeping gradients well-defined (direction comes from W init).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, eps: float = 1e-6):
        super().__init__(in_features, out_features, bias=bias)
        self.eps = eps
        # Standard init for direction; zero magnitude for zero initial update
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.magnitude = nn.Parameter(torch.zeros(out_features, dtype=self.weight.dtype, device=self.weight.device))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight_dir = F.normalize(self.weight, dim=1, eps=self.eps)
        scaled_weight = weight_dir * self.magnitude.unsqueeze(1)
        return F.linear(input, scaled_weight, self.bias)


class MinimalSSM(nn.Module):
    """Minimal state-space modeling block inspired by simplified Mamba behavior."""

    def __init__(self, d_model: int, d_state: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        self.x_proj = nn.Linear(d_model, d_model + d_state * 2)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        h = torch.zeros(batch, dim, self.d_state, dtype=x.dtype, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            projected = self.x_proj(x_t)
            delta, B, C = torch.split(projected, [dim, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(delta)
            A = -torch.exp(self.A_log)
            A_bar = torch.exp(delta.unsqueeze(-1) * A)
            B_bar = delta.unsqueeze(-1) * B.unsqueeze(1)
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = (h * C.unsqueeze(1)).sum(dim=-1) + self.D * x_t
            outputs.append(y_t)

        out = torch.stack(outputs, dim=1)
        return self.out_proj(out)

class STMambaLoRAInline(nn.Module):
    """
    LoRA-down (C -> r) -> temporal Mamba over T in rank space -> LoRA-up (r -> C).

    Input:  hidden_states of shape (B, T, P, C),
            where P = tokens per frame (e.g., CLS + patches per frame).

    - Mamba runs per token position across frames (no frame-average pooling).

    Caller is expected to do: hidden_states = hidden_states + delta
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.1,
        # Mamba-specific hyperparameters (use sensible defaults)
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        debug: bool = False,
        out_size: Optional[int] = None,
        use_dora: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.drop = nn.Dropout(dropout)
        self.debug = bool(debug)
        self.out_size = out_size if out_size is not None else self.hidden_size

        # LoRA down: C -> r
        self.down = nn.Linear(self.hidden_size, self.rank, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))

        # Minimal SSM acting as STMamba
        self.mamba = MinimalSSM(d_model=self.rank, d_state=d_state)
        self.d_conv = d_conv
        self.expand = expand

        # LoRA up: r -> C
        self.use_dora = bool(use_dora)
        if self.use_dora:
            self.up = DoRALinear(self.rank, self.out_size if out_size is not None else self.hidden_size, bias=False)
            nn.init.zeros_(self.up.magnitude)
        else:
            self.up = nn.Linear(self.rank, self.out_size if out_size is not None else self.hidden_size, bias=False)
            nn.init.zeros_(self.up.weight)

        # LoRA scaling (no gate)
        self.scaling = self.alpha / float(self.rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: int,
        frame_mask: Optional[torch.Tensor] = None,
        grid_hw: Optional[Tuple[int, int]] = None,  # kept for API compatibility, unused
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, P, C)
            num_frames:    T (temporal length)
            frame_mask:    optional, (T,), (B, T), or (B, T, ...) indicating valid frames
            grid_hw:       unused (for interface compatibility with conv-based ST-LoRA)
        Returns:
            delta: (B, T, P, C) residual to be added to hidden_states
        """
        if hidden_states.dim() != 4:
            raise ValueError(
                f"[ST-Mamba-LoRA] Expected (B, T, P, C); got {tuple(hidden_states.shape)}"
            )

        B, T, P, C = hidden_states.shape
        if T != num_frames:
            raise ValueError(
                f"[ST-Mamba-LoRA] num_frames={num_frames} but tensor has T={T}"
            )
        if C != self.hidden_size:
            raise ValueError(
                f"[ST-Mamba-LoRA] Hidden mismatch: {C} != {self.hidden_size}"
            )
        if P <= 0:
            raise ValueError(
                "[ST-Mamba-LoRA] Need at least one token per frame (P > 0)."
            )

        x = self.drop(hidden_states)  # (B, T, P, C)

        if frame_mask is not None:
            fm = frame_mask
            if fm.dim() == 1:
                fm = fm.unsqueeze(0)
            if fm.size(0) == 1 and B > 1:
                fm = fm.expand(B, T)
            frame_mask_bool = fm.to(dtype=torch.bool, device=x.device)
        else:
            frame_mask_bool = torch.ones(B, T, dtype=torch.bool, device=x.device)

        mask_float = frame_mask_bool.to(dtype=x.dtype)

        # Tokenwise temporal Mamba: run over T for each token position.
        x_btpc = x.view(B * T, P, C)
        red = self.down(x_btpc).view(B, T, P, self.rank)  # (B, T, P, r)
        red = red * mask_float[:, :, None, None]

        red_seq = red.permute(0, 2, 1, 3).contiguous()  # (B, P, T, r)
        red_seq = red_seq.view(B * P, T, self.rank)      # (B*P, T, r)
        red_seq = self.mamba(red_seq)                    # (B*P, T, r)
        red_seq = red_seq.view(B, P, T, self.rank)
        red_seq = red_seq.permute(0, 2, 1, 3).contiguous()  # (B, T, P, r)
        red_seq = red_seq * mask_float[:, :, None, None]

        red_up = red_seq.view(B * T, P, self.rank)
        delta = self.up(red_up) * self.scaling
        delta = delta.view(B, T, P, self.out_size)

        if self.debug:
            with torch.no_grad():
                print(
                    f"[ST-Mamba-LoRA] delta mean={float(delta.abs().mean()):.6f}",
                    flush=True,
                )

        # Only residual; caller does h = h + delta
        return delta


class _STBaseAttentionLoRAInline(nn.Module):
    """
    Base helper providing common logic for temporal attention LoRA variants.
    Subclasses only implement `_attend_rank_space`.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.1,
        debug: bool = False,
        out_size: Optional[int] = None,
        use_dora: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.drop = nn.Dropout(dropout)
        self.debug = bool(debug)
        self.out_size = out_size if out_size is not None else self.hidden_size
        self.use_dora = bool(use_dora)

        self.down = nn.Linear(self.hidden_size, self.rank, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))

        if self.use_dora:
            self.up = DoRALinear(self.rank, self.out_size, bias=False)
            nn.init.zeros_(self.up.magnitude)
        else:
            self.up = nn.Linear(self.rank, self.out_size, bias=False)
            nn.init.zeros_(self.up.weight)
        self.scaling = self.alpha / float(self.rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: int,
        grid_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if hidden_states.dim() != 4:
            raise ValueError(
                f"[ST-Attn-LoRA] Expected (B, T, P, C); got {tuple(hidden_states.shape)}"
            )

        B, T, P, C = hidden_states.shape
        if T != num_frames:
            raise ValueError(
                f"[ST-Attn-LoRA] num_frames={num_frames} but tensor has T={T}"
            )
        if C != self.hidden_size:
            raise ValueError(
                f"[ST-Attn-LoRA] Hidden mismatch: {C} != {self.hidden_size}"
            )
        if P <= 0:
            raise ValueError(
                "[ST-Attn-LoRA] Need at least one token per frame (P > 0)."
            )

        x = self.drop(hidden_states)

        x_btpc = x.view(B * T, P, C)
        red = self.down(x_btpc).view(B, T, P, self.rank)

        red_seq = red.permute(0, 2, 1, 3).contiguous().view(B * P, T, self.rank)
        att_seq = self._attend_rank_space(red_seq, key_padding_mask=None)
        att_seq = (
            att_seq.view(B, P, T, self.rank)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        red_up = att_seq.view(B * T, P, self.rank)
        delta = self.up(red_up) * self.scaling
        delta = delta.reshape(B, T, P, self.out_size)

        if self.debug:
            with torch.no_grad():
                print(
                    f"[ST-Attn-LoRA] delta mean={float(delta.abs().mean()):.6f}",
                    flush=True,
                )

        return delta

    def _attend_rank_space(
        self,
        seq: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError


class STMultiHeadAttentionLoRAInline(_STBaseAttentionLoRAInline):
    """
    Temporal multi-head attention operating on the per-token frame sequence.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.1,
        num_heads: int = 4,
        debug: bool = False,
        out_size: Optional[int] = None,
        use_dora: bool = False,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            debug=debug,
            out_size=out_size,
            use_dora=use_dora,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=self.rank,
            num_heads=num_heads,
            batch_first=True,
        )

    def _attend_rank_space(
        self,
        seq: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out, _ = self.attn(
            seq,
            seq,
            seq,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return out


class STSelfAttentionLoRAInline(_STBaseAttentionLoRAInline):
    """
    Single-head self-attention tailored for temporal fusion across frames.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.1,
        debug: bool = False,
        out_size: Optional[int] = None,
        use_dora: bool = False,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            debug=debug,
            out_size=out_size,
            use_dora=use_dora,
        )
        self.self_q = nn.Linear(self.rank, self.rank, bias=False)
        self.self_k = nn.Linear(self.rank, self.rank, bias=False)
        self.self_v = nn.Linear(self.rank, self.rank, bias=False)

    def _attend_rank_space(
        self,
        seq: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q = self.self_q(seq)
        k = self.self_k(seq)
        v = self.self_v(seq)

        scale = 1.0 / math.sqrt(self.rank)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).expand(-1, seq.size(1), -1)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v)

        if key_padding_mask is not None:
            valid = (~key_padding_mask).unsqueeze(-1).to(dtype=out.dtype)
            out = out * valid

        return out




class STLSTMLoRAInline(nn.Module):
    """
    LoRA-down (C -> r) -> temporal LSTM over T in rank space -> LoRA-up (r -> C).

    Input:  hidden_states of shape (B, T, P, C),
            where P = tokens per frame (e.g., CLS + patches per frame).

    All P tokens per frame are treated uniformly.
    Caller is expected to do: hidden_states = hidden_states + delta
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.1,
        num_layers: int = 1,
        bidirectional: bool = False,
        debug: bool = False,
        out_size: Optional[int] = None, 
        use_dora: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.drop = nn.Dropout(dropout)
        self.debug = bool(debug)
        self.out_size = out_size if out_size is not None else hidden_size
        self.use_dora = bool(use_dora)

        # LoRA down: C -> r
        self.down = nn.Linear(self.hidden_size, self.rank, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))

        # Temporal LSTM in rank space
        self.lstm = nn.LSTM(
            input_size=self.rank,
            hidden_size=self.rank,
            num_layers=num_layers,
            batch_first=True,   # (N, T, r)
            bidirectional=bidirectional,
        )

        lstm_out_dim = self.rank * (2 if bidirectional else 1)
        if lstm_out_dim != self.rank:
            self.proj_after_lstm = nn.Linear(lstm_out_dim, self.rank)
        else:
            self.proj_after_lstm = nn.Identity()

        # LoRA up: r -> C
        if self.use_dora:
            self.up = DoRALinear(self.rank, self.out_size, bias=False)
            nn.init.zeros_(self.up.magnitude)
        else:
            self.up = nn.Linear(self.rank, self.out_size, bias=False)
            nn.init.zeros_(self.up.weight)

        # LoRA scaling (no gate)
        self.scaling = self.alpha / float(self.rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: int,
        frame_mask: Optional[torch.Tensor] = None,
        grid_hw: Optional[Tuple[int, int]] = None,  # kept for API compatibility, unused
    ) -> torch.Tensor:
        if hidden_states.dim() != 4:
            raise ValueError(f"[ST-LSTM-LoRA] Expected (B, T, P, C); got {tuple(hidden_states.shape)}")

        B, T, P, C = hidden_states.shape
        if T != num_frames:
            raise ValueError(f"[ST-LSTM-LoRA] num_frames={num_frames} but tensor has T={T}")
        if C != self.hidden_size:
            raise ValueError(f"[ST-LSTM-LoRA] Hidden mismatch: {C} != {self.hidden_size}")
        if P <= 0:
            raise ValueError("[ST-LSTM-LoRA] Need at least one token per frame (P > 0).")

        x = self.drop(hidden_states)  # (B, T, P, C)

        # (B, T, P, C) -> (B*T, P, C)
        x_btpc = x.view(B * T, P, C)

        # LoRA down: (B*T, P, C) -> (B*T, P, r)
        red = self.down(x_btpc)
        red = red.view(B, T, P, self.rank)        # (B, T, P, r)

    

        # Optional frame mask: broadcast over tokens & rank
        fm = None
        if frame_mask is not None:
            fm = frame_mask
            if fm.dim() == 1:
                fm = fm.unsqueeze(0)              # (1, T)
            if fm.size(0) == 1 and B > 1:
                fm = fm.expand(B, T)             # (B, T)
            fm = fm.to(dtype=red.dtype, device=red.device)   # (B, T)
            red = red * fm[:, :, None, None]     # (B, T, P, r)

        # We want to run LSTM along T for each (B, token_index)
        # (B, T, P, r) -> (B, P, T, r) -> (B*P, T, r)
        red_seq = red.permute(0, 2, 1, 3).contiguous()       # (B, P, T, r)
        # Each token's temporal stream is flattened so the LSTM processes T steps per stream.
        red_seq = red_seq.view(B * P, T, self.rank)          # (B*P, T, r)

        # Temporal LSTM
        red_seq, _ = self.lstm(red_seq)                      # (B*P, T, r or 2r)
        red_seq = self.proj_after_lstm(red_seq)              # (B*P, T, r)

        # Back to (B, T, P, r)
        red_seq = red_seq.view(B, P, T, self.rank)           # (B, P, T, r)
        red_seq = red_seq.permute(0, 2, 1, 3).contiguous()   # (B, T, P, r)

        # Re-apply mask after LSTM (keep residual zeroed for masked frames)
        if fm is not None:
            red_seq = red_seq * fm[:, :, None, None]

        # (B, T, P, r) -> (B*T, P, r)
        red_up = red_seq.view(B * T, P, self.rank)

        # LoRA up + scaling
        delta = self.up(red_up)                  # (B*T, P, C)
        delta = delta * self.scaling
        delta = delta.reshape(B, T, P, self.out_size)
        if self.debug:
            with torch.no_grad():
                print(
                    f"[ST-LSTM-LoRA] delta mean={float(delta.abs().mean()):.6f}",
                    flush=True,
                )

        return delta



class STLoRA3DInline(nn.Module):
    """
    LoRA-down (C -> r) -> 3D depthwise conv over (T, H_p, W_p) in rank space -> LoRA-up (r -> C).

    Input:  hidden_states of shape (B, T, P, C),
            where P = tokens per frame (e.g., CLS + patches per frame).

    All P tokens per frame are treated uniformly by the adapter.
    Caller is expected to do: hidden_states = hidden_states + delta
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 4,
        alpha: float = 16.0,
        k_t: int = 3,
        k_s: int = 3,
        dropout: float = 0.1,
        debug: bool = False,
        out_size: Optional[int] = None,
        use_dora: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.k_t = int(k_t)
        self.k_s = int(k_s)
        self.drop = nn.Dropout(dropout)
        self.debug = bool(debug)
        self.out_size = out_size if out_size is not None else hidden_size

        # LoRA down: C -> r
        self.down = nn.Linear(self.hidden_size, self.rank, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))

        # 3D depthwise conv over (T, H_p, W_p) in rank space
        self.conv3d = nn.Conv3d(
            in_channels=self.rank,
            out_channels=self.rank,
            kernel_size=(self.k_t, self.k_s, self.k_s),
            stride=(1, 1, 1),
            padding=(self.k_t // 2, self.k_s // 2, self.k_s // 2),
            groups=self.rank,
            bias=True,
        )
        nn.init.constant_(self.conv3d.weight, 0.0)
        nn.init.constant_(self.conv3d.bias, 0.0)

        # LoRA up: r -> C
        self.use_dora = bool(use_dora)
        if self.use_dora:
            self.up = DoRALinear(self.rank, self.out_size, bias=False)
            nn.init.zeros_(self.up.magnitude)
        else:
            self.up = nn.Linear(self.rank, self.out_size, bias=False)
            nn.init.zeros_(self.up.weight)

        # LoRA scaling (no gate)
        self.scaling = self.alpha / float(self.rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: int,
        frame_mask: Optional[torch.Tensor] = None,
        grid_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if hidden_states.dim() != 4:
            raise ValueError(f"[ST-LoRA] Expected (B, T, P, C); got {tuple(hidden_states.shape)}")

        B, T, P, C = hidden_states.shape
        if T != num_frames:
            raise ValueError(f"[ST-LoRA] num_frames={num_frames} but tensor has T={T}")
        if C != self.hidden_size:
            raise ValueError(f"[ST-LoRA] Hidden mismatch: {C} != {self.hidden_size}")
        if P <= 0:
            raise ValueError("[ST-LoRA] Need at least one token per frame (P > 0).")

        x = self.drop(hidden_states)  # (B, T, P, C)

        # Infer or check spatial grid (H_p, W_p) over ALL P tokens
        if grid_hw is not None:
            H_p, W_p = grid_hw
            if H_p * W_p != P:
                raise ValueError(
                    f"[ST-LoRA] Provided grid {H_p}x{W_p} != tokens per frame {P}"
                )
        else:
            H_p = int(math.sqrt(P))
            W_p = P // max(1, H_p)
            while H_p * W_p != P and H_p > 1:
                H_p -= 1
                if P % H_p == 0:
                    W_p = P // H_p
                    break
            if H_p * W_p != P:
                raise ValueError(
                    f"[ST-LoRA] Cannot factor tokens-per-frame {P} into rectangular grid"
                )

        # ===== C -> r -> 3D conv -> r -> C on ALL tokens =====
        # LoRA down (per frame/token, no temporal mixing)
        red = self.down(x)  # (B, T, P, r)

        # To 3D conv layout: (B, r, T, H_p, W_p)
        # The rank dimension becomes the conv channel, and time+spatial grids form the depth/height/width axes.
        red_3d = red.view(B, T, H_p, W_p, self.rank).permute(
            0, 4, 1, 2, 3
        ).contiguous()

        # Optional frame masking before conv
        fm = None
        if frame_mask is not None:
            fm = frame_mask
            if fm.dim() == 1:
                fm = fm.unsqueeze(0)
            if fm.size(0) == 1 and B > 1:
                fm = fm.expand(B, T)
            fm = fm.to(dtype=red_3d.dtype, device=red_3d.device)
            fm = fm[:, None, :, None, None]  # (B, 1, T, 1, 1)
            red_3d = red_3d * fm

        # 3D depthwise conv (in rank space)
        red_3d = self.conv3d(red_3d)

        # Re-apply mask after conv
        if fm is not None:
            red_3d = red_3d * fm

        # Back to (B, T, P, r)
        red_up = red_3d.permute(0, 2, 3, 4, 1).contiguous().view(
            B, T, P, self.rank
        )

        # LoRA up + scaling
        delta = self.up(red_up)  # (B, T, P, C)
        delta = delta * self.scaling
        if self.debug:
            with torch.no_grad():
                print(
                    f"[ST-LoRA-uniform] "
                    f"delta mean={float(delta.abs().mean()):.6f}",
                    flush=True,
                )

        # Only residual; caller does h = h + delta
        return delta




class TimeShiftLoRA(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        rank: int = 32, 
        alpha: float = 32.0,
        n_div: int = 3, 
        dropout: float = 0.1,
        out_dim: Optional[int] = None,
        use_dora: bool = False,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.scaling = float(alpha) / self.rank
        self.n_div = n_div
        self.use_dora = bool(use_dora)
        
        out_dim = out_dim if out_dim is not None else hidden_size
        self.lora_down = nn.Linear(hidden_size, rank, bias=False)
        if self.use_dora:
            self.lora_up = DoRALinear(rank, out_dim, bias=False)
            nn.init.zeros_(self.lora_up.magnitude)
        else:
            self.lora_up = nn.Linear(rank, out_dim, bias=False)
            nn.init.zeros_(self.lora_up.weight)
        
        #self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor, num_frames: int, **kwargs):
        """
        Args:
            hidden_states: (Batch * Time, Tokens, Hidden) or (Batch, Time, Tokens, Hidden)
            num_frames: int (Time)
        """
        if hidden_states.dim() == 4:
            # Accept encoder/decoder callers that provide (B, T, P, C)
            b, t, p, c = hidden_states.shape
            hidden_states = hidden_states.view(b * t, p, c)
        elif hidden_states.dim() != 3:
            raise ValueError(
                f"TimeShiftLoRA expects 3D or 4D input; got {tuple(hidden_states.shape)}"
            )

        # 1. Down Project
        x = self.lora_down(hidden_states)
        #x = self.act(x)

        # 2. Temporal Shift
        # Ensure input is contiguous before reshaping
        BT, L, R = x.size()
        
        # Safety check for Batch > 1
        if BT % num_frames != 0:
            raise ValueError(f"Total frames {BT} not divisible by clip length {num_frames}")
            
        B = BT // num_frames
        
        # Fold: reshape to (Batch, Time, Tokens, Rank) so shifts operate along the temporal axis for each token/rank slice.
        x = x.view(B, num_frames, L, R)

        fold = R // self.n_div
        if fold > 0:
            out = torch.zeros_like(x)
            
            # Shift Past: x[:, :-1] means "All Batches, Time 0 to T-1"
            out[:, 1:, :, :fold] = x[:, :-1, :, :fold]
            
            # Shift Future: x[:, 1:] means "All Batches, Time 1 to T"
            out[:, :-1, :, fold:2*fold] = x[:, 1:, :, fold:2*fold]
            
            # Static
            out[:, :, :, 2*fold:] = x[:, :, :, 2*fold:]
            
            # Overwrite x with shifted version
            x = out

        # Unfold back to (Batch * Time, Tokens, Rank)
        x = x.view(BT, L, R)

        # 3. Up Project
        x = self.lora_up(x)
        
        return self.dropout(x) * self.scaling

class SpatioTemporalConvAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        adapter_channels: int,
        kernel_size: Tuple[int, int, int],
        disable_cudnn: bool,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels,
            adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(k // 2 for k in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)

        # Init conv to identity-like (no-op) and biases to 0
        nn.init.constant_(self.conv.weight, 0.0)
        nn.init.constant_(self.conv.bias, 0.0)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)

        # Non-linearity + optional norm in adapter space
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(adapter_channels)

        # Learnable gate on the residual path (start from 0 -> identity)
        self.gate = nn.Parameter(torch.zeros(1))

        self.disable_cudnn = disable_cudnn

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        x: (B*T, L, C) where L is the token count per frame.
        """
        if x.dim() != 3:
            raise ValueError("SpatioTemporalConvAdapter expects (B*T, L, C).")
        bt, token_count, channels = x.size()
        if num_frames <= 0 or bt % num_frames != 0:
            raise ValueError(
                f"num_frames ({num_frames}) must divide batch*time dimension ({bt})."
            )

        batch_size = bt // num_frames
        spatial_tokens = token_count
        if spatial_tokens <= 0:
            raise ValueError("Expected at least one token per frame.")

        spatial_height, spatial_width = self._infer_hw(spatial_tokens)

        residual = x
        # Linear down + non-linearity + norm
        x = self.fc1(x)  # (B*T, S, A)
        x = self.act(x)
        x = self.norm(x)

        adapter_channels = self.conv.in_channels

        # (B*T, S, A) -> (B, A, T, H, W)
        x = x.view(
            batch_size,
            num_frames,
            spatial_height,
            spatial_width,
            adapter_channels,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # 3D depthwise conv
        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and not self.disable_cudnn
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        # Non-linearity after conv
        x = self.act(x)

        # Back to (B*T, S, A)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(
            bt, spatial_tokens, adapter_channels
        )

        # Up projection and gated residual (no in-place write)
        x = self.fc2(x)  # (B*T, S, C)
        return residual + self.gate * x

    @staticmethod
    def _infer_hw(token_count: int) -> Tuple[int, int]:
        root = int(math.sqrt(token_count))
        for candidate in range(root, 0, -1):
            if token_count % candidate == 0:
                return candidate, token_count // candidate
        return 1, token_count


class SpatioTemporalAdapterModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        adapter_channels: int,
        kernel_size: Tuple[int, int, int],
        disable_cudnn: bool,
    ) -> None:
        super().__init__()
        self.adapter = SpatioTemporalConvAdapter(
            in_channels=hidden_size,
            adapter_channels=adapter_channels,
            kernel_size=kernel_size,
            disable_cudnn=disable_cudnn,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: int,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states.dim() != 4:
            raise ValueError(
                "SpatioTemporalAdapterModule expects hidden states shaped (B, T, L, C)."
            )

        batch_size, temporal_dim, token_count, channels = hidden_states.shape
        if temporal_dim != num_frames:
            raise ValueError(
                f"Expected temporal dimension {temporal_dim} to equal num_frames {num_frames}."
            )

        merged = hidden_states.view(batch_size * num_frames, token_count, channels)
        merged = self.adapter(merged, num_frames=num_frames)
        merged = merged.view(batch_size, num_frames, token_count, channels)

        if frame_mask is not None:
            mask = self._reshape_frame_mask(frame_mask, batch_size, num_frames).to(
                dtype=merged.dtype, device=merged.device
            )
            merged = merged * mask

        return merged

    @staticmethod
    def _reshape_frame_mask(
        frame_mask: torch.Tensor, batch_size: int, num_frames: int
    ) -> torch.Tensor:
        if frame_mask.dim() == 1 and frame_mask.numel() == num_frames:
            mask = frame_mask.view(1, num_frames, 1, 1).expand(batch_size, -1, -1, -1)
        elif frame_mask.dim() == 2 and frame_mask.shape == (batch_size, num_frames):
            mask = frame_mask.view(batch_size, num_frames, 1, 1)
        elif frame_mask.dim() == 4 and frame_mask.shape[:2] == (batch_size, num_frames):
            mask = frame_mask
        else:
            raise ValueError(
                "[ST-Adapter] frame_mask must be shape (num_frames,), (B, T), or (B, T, 1, 1); "
                f"got {tuple(frame_mask.shape)}"
            )
        return mask


class TimeShiftedVisionLinear(nn.Module):
    """
    Wraps a Linear layer (QKV or Proj) in the Vision Encoder.
    Adds a TimeShiftLoRA branch in parallel.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank=32,
        alpha=32.0,
        n_div=3,
        dropout=0.1,
        num_frames: int = 8,
        use_dora: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.use_dora = bool(use_dora)

        self.adapter = TimeShiftLoRA(
            hidden_size=self.in_features,
            rank=rank,
            alpha=alpha,
            n_div=n_div,
            dropout=dropout,
            out_dim=self.out_features,
            use_dora=self.use_dora,
        )
        self.layer_type = "linear"
        self.num_frames = int(num_frames)

    def forward(self, x):
        base_out = self.base_layer(x)
        if x.dim() == 2:
            x_btpc, bsz, t, p = _reshape_qwen_2d_tokens(x, self.num_frames, "TimeShift-Enc")
            hidden = x_btpc.size(-1)
            x_btpc_flat = x_btpc.view(bsz * t, p, hidden)
            delta = self.adapter(x_btpc_flat, num_frames=t)
            delta = delta.reshape(bsz, t * p, self.out_features).reshape(x.size(0), self.out_features)
            return base_out + delta
        if x.dim() != 3:
            raise ValueError(f"[TimeShift-Enc] Expected (B*T, P, C); got {tuple(x.shape)}")
        adapter_out = self.adapter(x, num_frames=self.num_frames)
        return base_out + adapter_out


class STLoRAVisionLinear(nn.Module):
    """
    Wraps a Linear layer (typically vision encoder QKV) and adds an ST-LoRA
    branch in parallel that operates over time and space.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 32,
        alpha: float = 128.0,
        k: int = 3,
        dropout: float = 0.1,
        debug: bool = False,
        use_dora: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.debug = True
        self.use_dora = bool(use_dora)

        self.stlora = STLoRA3DInline(
            hidden_size=self.in_features,
            rank=rank,
            alpha=alpha,
            k_t=k,
            k_s=k,
            dropout=dropout,
            debug=debug,
            out_size=self.out_features,
            use_dora=self.use_dora,
        )

        for p in self.stlora.parameters():
            p.requires_grad = True

        self.num_frames = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #if self.debug:
            #print(f"[ST-LoRA-Enc] qkv input shape: {tuple(x.shape)}")
            #print(f"[ST-LoRA-Enc] grad enabled: {torch.is_grad_enabled()}")
        base_out = self.base_layer(x)
        if x.dim() == 2:
            grid_thw, _ = _get_qwen_visual_context()
            if grid_thw:
                first = grid_thw[0]
                if len(first) != 3:
                    raise ValueError(
                        f"[ST-LoRA-Enc] grid_thw entry must have 3 elements; got {first}"
                    )
                t0, h0, w0 = [int(v) for v in first]
                if any(
                    len(entry) != 3 or entry[0] != t0 or entry[1] != h0 or entry[2] != w0
                    for entry in grid_thw
                ):
                    raise ValueError(
                        "[ST-LoRA-Enc] grid_thw entries are not uniform across the batch."
                    )
                p = h0 * w0
                b = len(grid_thw)
                total_tokens = b * t0 * p
                if x.size(0) == total_tokens:
                    hidden = x.size(1)
                    x_b = x.view(b, t0 * p, hidden)
                    x_btpc = x_b.view(b, t0, p, hidden)
                    delta = self.stlora(x_btpc, num_frames=t0)
                    delta = delta.view(b, t0 * p, self.out_features).reshape(
                        total_tokens, self.out_features
                    )
                    return base_out + delta

            t = self.num_frames
            if t <= 0:
                raise ValueError("[ST-LoRA-Enc] num_frames must be > 0 for 2D input.")
            seq_len = x.size(0)
            if seq_len % t != 0:
                raise ValueError(
                    f"[ST-LoRA-Enc] 2D input length {seq_len} not divisible by num_frames={t}."
                )
            p = seq_len // t
            hidden = x.size(1)
            x_btpc = x.view(1, t, p, hidden)
            delta = self.stlora(x_btpc, num_frames=t)
            delta = delta.view(seq_len, self.out_features)
            return base_out + delta
        if x.dim() != 3:
            raise ValueError(
                f"[ST-LoRA-Enc] Expected (B*T, P, C); got {tuple(x.shape)}"
            )

        bt, seq_len, hidden = x.shape
        t = self.num_frames
        if t <= 0 or bt % t != 0:
            raise ValueError(
                f"[ST-LoRA-Enc] Invalid num_frames={t} for input batch*time={bt}."
            )
        b = bt // t

        x_btpc = x.reshape(b, t, seq_len, hidden)
        delta = self.stlora(x_btpc, num_frames=t)
        delta = torch.reshape(delta, (bt, seq_len, self.out_features))
        return base_out + delta


class STLSTMLoRAVisionLinear(nn.Module):
    """
    Wraps a Linear layer (typically vision encoder QKV) and adds an STLSTM-LoRA
    branch that runs across time before projecting back to the output dimension.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 32,
        alpha: float = 128.0,
        dropout: float = 0.1,
        num_layers: int = 1,
        bidirectional: bool = False,
        debug: bool = False,
        use_dora: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.use_dora = bool(use_dora)

        self.adapter = STLSTMLoRAInline(
            hidden_size=self.in_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            num_layers=num_layers,
            bidirectional=bidirectional,
            debug=debug,
            out_size=self.out_features,
            use_dora=self.use_dora,
        )

        for p in self.adapter.parameters():
            p.requires_grad = True

        self.layer_type = "linear"
        self.num_frames = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if x.dim() == 2:
            x_btpc, bsz, t, p = _reshape_qwen_2d_tokens(x, self.num_frames, "ST-LSTM-Enc")
            delta = self.adapter(x_btpc, num_frames=t)
            delta = delta.reshape(bsz, t * p, self.out_features).reshape(x.size(0), self.out_features)
            return base_out + delta
        if x.dim() != 3:
            raise ValueError(
                f"[ST-LSTM-Enc] Expected (B*T, P, C); got {tuple(x.shape)}"
            )

        bt, seq_len, hidden = x.shape
        t = self.num_frames
        if t <= 0 or bt % t != 0:
            raise ValueError(
                f"[ST-LSTM-Enc] Invalid num_frames={t} for input batch*time={bt}."
            )
        b = bt // t

        x_btpc = x.reshape(b, t, seq_len, hidden)
        delta = self.adapter(x_btpc, num_frames=t)
        delta = torch.reshape(delta, (bt, seq_len, self.out_features))
        return base_out + delta


class STAttentionLoRAVisionLinear(nn.Module):
    __slots__ = ("base_layer", "adapter", "num_frames", "layer_type", "attention_cls")

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 32,
        alpha: float = 128.0,
        dropout: float = 0.1,
        num_heads: int = 4,
        debug: bool = False,
        attention_cls: Type[nn.Module] = STMultiHeadAttentionLoRAInline,
        use_dora: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.attention_cls = attention_cls
        self.use_dora = bool(use_dora)

        adapter_kwargs: Dict[str, Any] = dict(
            hidden_size=self.in_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            debug=debug,
            out_size=self.out_features,
            use_dora=self.use_dora,
        )
        if issubclass(attention_cls, STMultiHeadAttentionLoRAInline):
            adapter_kwargs["num_heads"] = num_heads

        self.adapter = attention_cls(**adapter_kwargs)

        for p in self.adapter.parameters():
            p.requires_grad = True

        self.layer_type = "linear"
        self.num_frames = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if x.dim() == 2:
            x_btpc, bsz, t, p = _reshape_qwen_2d_tokens(x, self.num_frames, "ST-Attn-Enc")
            delta = self.adapter(x_btpc, num_frames=t)
            delta = delta.reshape(bsz, t * p, self.out_features).reshape(x.size(0), self.out_features)
            return base_out + delta
        if x.dim() != 3:
            raise ValueError(
                f"[ST-Attn-Enc] Expected (B*T, P, C); got {tuple(x.shape)}"
            )

        bt, seq_len, hidden = x.shape
        t = self.num_frames
        if t <= 0 or bt % t != 0:
            raise ValueError(
                f"[ST-Attn-Enc] Invalid num_frames={t} for input batch*time={bt}."
            )
        b = bt // t

        x_btpc = x.reshape(b, t, seq_len, hidden)
        delta = self.adapter(x_btpc, num_frames=t)
        delta = torch.reshape(delta, (bt, seq_len, self.out_features))
        return base_out + delta


class STMambaVisionLinear(nn.Module):
    """
    Wraps a Linear layer and adds an STMamba-LoRA branch over the time dimension.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 32,
        alpha: float = 128.0,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        debug: bool = False,
        use_dora: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.use_dora = bool(use_dora)

        self.adapter = STMambaLoRAInline(
            hidden_size=self.in_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            debug=debug,
            out_size=self.out_features,
            use_dora=self.use_dora,
        )

        for p in self.adapter.parameters():
            p.requires_grad = True

        self.layer_type = "linear"
        self.num_frames = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if x.dim() == 2:
            x_btpc, bsz, t, p = _reshape_qwen_2d_tokens(x, self.num_frames, "ST-Mamba-Enc")
            delta = self.adapter(x_btpc, num_frames=t)
            delta = delta.reshape(bsz, t * p, self.out_features).reshape(x.size(0), self.out_features)
            return base_out + delta
        if x.dim() != 3:
            raise ValueError(
                f"[ST-Mamba-Enc] Expected (B*T, P, C); got {tuple(x.shape)}"
            )

        bt, seq_len, hidden = x.shape
        t = self.num_frames
        if t <= 0 or bt % t != 0:
            raise ValueError(
                f"[ST-Mamba-Enc] Invalid num_frames={t} for input batch*time={bt}."
            )
        b = bt // t

        x_btpc = x.reshape(b, t, seq_len, hidden)
        delta = self.adapter(x_btpc, num_frames=t)
        delta = torch.reshape(delta, (bt, seq_len, self.out_features))
        return base_out + delta


class PreprocessorWrapper(nn.Module):
    """Wrap a base module with a preprocessing hook that can modify hidden states."""

    def __init__(self, preprocess_fn: Callable[[torch.Tensor], torch.Tensor], base: nn.Module):
        super().__init__()
        self.preprocess_fn = preprocess_fn
        self.base = base

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.preprocess_fn(hidden_states)
        hidden_states = hidden_states + residual
        return self.base(hidden_states)


class PreprocessorWrapperAfter(nn.Module):
    """
    Wrap a base module with a preprocessing hook that injects a residual after the base.
    """

    def __init__(
        self,
        preprocess_fn: Callable[[torch.Tensor], torch.Tensor],
        base: nn.Module,
        expected_output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.preprocess_fn = preprocess_fn
        self.base = base
        self.expected_output_dim = expected_output_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.preprocess_fn(hidden_states)
        base_output = self.base(hidden_states)
        if self.expected_output_dim is not None and base_output.size(-1) != self.expected_output_dim:
            raise ValueError(
                f"[Vision-Adapter] Expected mlp output dim {self.expected_output_dim}, "
                f"got {base_output.size(-1)}."
            )
        if residual.shape != base_output.shape:
            raise ValueError(
                f"[Vision-Adapter] Residual shape {tuple(residual.shape)} "
                f"does not match projector output {tuple(base_output.shape)}."
            )
        return base_output + residual


class DecoderAdapterLinear(nn.Module):
    """
    Wraps a decoder linear projection with a spatio-temporal adapter that mirrors
    the vision-side module.
    """

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

    def _apply_adapter(
        self,
        seq: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if seq.numel() == 0:
            return seq.new_zeros(seq.size(0), seq.size(1), self.out_features)
        b, seq_len, _ = seq.shape
        x_btpc = seq.view(b, 1, seq_len, self.in_features)
        kwargs: Dict[str, Any] = {"num_frames": 1}
        if frame_mask is not None:
            raise ValueError(
                "[Decoder-Adapter] frame_mask is not supported for text adapters; "
                "provide real frame tokens or disable the mask."
            )
        delta = self.adapter(x_btpc, **kwargs)
        return delta.reshape(b, seq_len, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if x.dim() != 3:
            return base_out

        delta = self._apply_adapter(x)
        return base_out + delta


__all__ = [
    "DoRALinear",
    "STLoRA3DInline",
    "SpatioTemporalConvAdapter",
    "SpatioTemporalAdapterModule",
    "TimeShiftedVisionLinear",
    "STLoRAVisionLinear",
    "STLSTMLoRAVisionLinear",
    "STAttentionLoRAVisionLinear",
    "STMambaVisionLinear",
    "PreprocessorWrapper",
    "PreprocessorWrapperAfter",
    "DecoderAdapterLinear",
    "GatedAttentionWrapper",
    "HeadwiseGate",
    "LayerwiseGate",
    "STMultiHeadAttentionLoRAInline",
    "STSelfAttentionLoRAInline",
    "STMambaLoRAInline",
    "STLSTMLoRAInline",
    "TimeShiftLoRA",
    "set_qwen_visual_context",
]
