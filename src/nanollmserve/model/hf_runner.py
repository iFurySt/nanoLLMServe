"""Hugging Face model loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class LoadedModel:
    model: object
    tokenizer: object
    device: str
    dtype: str


def resolve_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str = "auto", *, device: str = "auto"):
    import torch

    if dtype == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        supported = ", ".join(sorted(mapping | {"auto": torch.float32}))
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of {supported}") from exc


def load_model_and_tokenizer(
    model_path: str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
) -> LoadedModel:
    """Load a causal LM and tokenizer from Hugging Face or a local directory."""

    # Some shared GPU environments expose broken optional sklearn/scipy or
    # torchvision installations. Transformers only needs those for unrelated
    # helper paths, so keep the v0.0 text-only causal-LM path independent.
    disable_optional_deps = os.environ.get(
        "NANOLLMSERVE_DISABLE_OPTIONAL_TRANSFORMERS_DEPS",
        os.environ.get("NANOLLMSERVE_DISABLE_SKLEARN", "1"),
    )
    if disable_optional_deps != "0":
        try:
            from transformers.utils import import_utils

            import_utils._scipy_available = False
            import_utils._sklearn_available = False
            import_utils._torchvision_available = False
        except Exception:
            pass

    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, device=resolved_device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=resolved_dtype,
            local_files_only=local_files_only,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=resolved_dtype,
            local_files_only=local_files_only,
        )
    model.to(resolved_device)
    model.eval()

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        dtype=str(resolved_dtype).removeprefix("torch."),
    )
