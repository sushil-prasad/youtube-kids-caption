from __future__ import annotations

import os
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    asr_model: str = "pasketti_first"
    device: str = "auto"
    asr_model_path: str | None = None
    forced_aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    enable_forced_aligner: bool = True
    language: str | None = "English"
    enable_audio_normalization: bool = False
    confidence_high: float = 0.90
    confidence_medium: float = 0.70
    max_new_tokens: int = 512
    max_inference_batch_size: int = 1


def load_settings() -> Settings:
    language = os.getenv("LANGUAGE", "English").strip()
    model_path = os.getenv("ASR_MODEL_PATH", "").strip() or None
    return Settings(
        asr_model=os.getenv("ASR_MODEL", "pasketti_first").strip() or "pasketti_first",
        device=os.getenv("DEVICE", "auto").strip() or "auto",
        asr_model_path=model_path,
        forced_aligner=os.getenv("FORCED_ALIGNER", "Qwen/Qwen3-ForcedAligner-0.6B").strip(),
        enable_forced_aligner=_env_bool("ENABLE_FORCED_ALIGNER", True),
        language=language or None,
        enable_audio_normalization=_env_bool("ENABLE_AUDIO_NORMALIZATION", False),
        confidence_high=_env_float("CONFIDENCE_HIGH", 0.90),
        confidence_medium=_env_float("CONFIDENCE_MEDIUM", 0.70),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "512")),
        max_inference_batch_size=int(os.getenv("MAX_INFERENCE_BATCH_SIZE", "1")),
    )


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def mps_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        return False


def resolve_device(requested: str = "auto") -> str:
    """Pick cuda, mps, or cpu. Never require CUDA."""
    choice = (requested or "auto").strip().lower()
    cuda = cuda_available()
    mps = mps_available()

    if choice in {"cuda", "mps", "cpu"}:
        if choice == "cuda" and not cuda:
            warnings.warn("WARNING: CUDA was requested but is unavailable. Falling back to CPU.", stacklevel=2)
            return "cpu"
        if choice == "mps" and not mps:
            warnings.warn("WARNING: MPS was requested but is unavailable. Falling back to CPU.", stacklevel=2)
            return "cpu"
        return choice

    if cuda:
        return "cuda"
    if mps:
        return "mps"
    return "cpu"


def torch_dtype_for(device: str):
    import torch

    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def device_label(device: str) -> str:
    if is_apple_silicon():
        return "Apple Silicon"
    if device == "cuda":
        return "NVIDIA GPU"
    return f"{platform.system()} {platform.machine()}"


def asr_model_label(settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    if settings.asr_model == "pasketti_first":
        return "Pasketti 1st Place / Qwen3-ASR-1.7B"
    return settings.asr_model


def print_device_banner(device: str | None = None, settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    backend = device or resolve_device(settings.device)
    lines = [
        f"Device: {device_label(backend)}",
        f"Backend: {backend.upper()}",
        f"ASR model: {asr_model_label(settings)}",
    ]
    if not _torch_available():
        lines.append("WARNING: torch is not installed. Inference cannot run until dependencies are installed.")
    elif backend == "cpu" and is_apple_silicon() and not mps_available():
        lines.append("WARNING: MPS is not available. Falling back to CPU.")
    text = "\n".join(lines)
    print(text)
    return text


def resolve_checkpoint(settings: Settings | None = None) -> str:
    """Prefer a local merged Pasketti checkpoint; otherwise the Qwen3-ASR-1.7B base."""
    settings = settings or load_settings()
    if settings.asr_model_path:
        return settings.asr_model_path
    local = ROOT / "models" / "pasketti_first"
    if (local / "config.json").is_file():
        return str(local)
    return "Qwen/Qwen3-ASR-1.7B"
