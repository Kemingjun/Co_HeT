from __future__ import annotations

from typing import Any, Callable


SUPPORTED_PRECISIONS = ("fp32", "amp_fp16")


def validate_precision(precision: str) -> str:
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"precision must be one of {SUPPORTED_PRECISIONS}")
    return precision


def amp_enabled(precision: str) -> bool:
    return validate_precision(precision) == "amp_fp16"


def autocast_context(torch_module: Any, precision: str):
    return torch_module.autocast(
        device_type="cuda",
        dtype=torch_module.float16,
        enabled=amp_enabled(precision),
    )


def create_grad_scaler(
    torch_module: Any, precision: str, init_scale: float = 16.0
):
    enabled = amp_enabled(precision)
    modern_scaler = getattr(getattr(torch_module, "amp", None), "GradScaler", None)
    if modern_scaler is not None:
        return modern_scaler("cuda", enabled=enabled, init_scale=init_scale)
    return torch_module.cuda.amp.GradScaler(
        enabled=enabled, init_scale=init_scale
    )


def scaled_optimizer_step(
    *,
    loss: Any,
    optimizer: Any,
    scaler: Any,
    clip_callback: Callable[[], Any],
) -> dict[str, Any]:
    scale_before = float(scaler.get_scale())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norms = clip_callback()
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    return {
        "grad_norms": grad_norms,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "step_skipped": scale_after < scale_before,
    }


def require_finite_tensors(**named_tensors: Any) -> None:
    import torch

    invalid = [
        name
        for name, tensor in named_tensors.items()
        if tensor is not None and not bool(torch.isfinite(tensor).all().item())
    ]
    if invalid:
        raise FloatingPointError(f"non-finite tensors: {', '.join(invalid)}")
