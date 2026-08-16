from __future__ import annotations

import warnings
from pathlib import Path

from app.config import Settings, load_settings, print_device_banner, resolve_checkpoint, resolve_device, torch_dtype_for
from app.models.base import ASRModel
from app.pipeline.timestamps import words_from_aligner, words_from_text
from app.transcript import Transcript

_LOADED: dict[tuple[str, str, bool], object] = {}


class PaskettiFirstModel(ASRModel):
    """1st-place Pasketti Word Track architecture: Qwen3-ASR-1.7B (+ optional forced aligner).

    The competition LoRA ensemble was not published as a single Hub repo. Place a merged
    checkpoint under models/pasketti_first or set ASR_MODEL_PATH. Otherwise the released
    Qwen/Qwen3-ASR-1.7B weights are used for inference.
    """

    name = "pasketti_first"

    def __init__(self, settings: Settings | None = None, device: str | None = None) -> None:
        self.settings = settings or load_settings()
        self.device = resolve_device(device or self.settings.device)
        self.checkpoint = resolve_checkpoint(self.settings)
        self._backend = None
        self._aligner_enabled = False

    def transcribe(self, audio_path: str | Path) -> Transcript:
        print_device_banner(self.device, self.settings)
        audio_path = Path(audio_path)
        backend = self._load()
        kwargs = {
            "audio": str(audio_path),
            "language": self.settings.language,
            "return_time_stamps": self._aligner_enabled,
        }
        results = backend.transcribe(**kwargs)
        result = results[0] if isinstance(results, list) else results
        text = (getattr(result, "text", None) or "").strip()
        language = getattr(result, "language", None) or self.settings.language
        duration = _wav_duration(audio_path)
        stamps = getattr(result, "time_stamps", None) if self._aligner_enabled else None
        words = words_from_aligner(stamps) if stamps is not None else []
        if not words:
            if self._aligner_enabled:
                warnings.warn(
                    "WARNING: forced aligner returned no word timestamps. Falling back to even spacing.",
                    stacklevel=2,
                )
            words = words_from_text(text, duration)
        return Transcript(
            text=text,
            words=words,
            language=language,
            model=self.name,
            device=self.device,
        )

    def _load(self):
        key = (self.checkpoint, self.device, self.settings.enable_forced_aligner)
        cached = _LOADED.get(key)
        if cached is not None:
            self._backend, self._aligner_enabled = cached
            return self._backend

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "qwen-asr is required for the Pasketti backend. Install with: pip install -r requirements.txt"
            ) from exc

        dtype = torch_dtype_for(self.device)
        device_map = "cuda:0" if self.device == "cuda" else self.device
        aligner_ok = False
        load_kwargs: dict = {
            "dtype": dtype,
            "device_map": device_map,
            "max_inference_batch_size": self.settings.max_inference_batch_size,
            "max_new_tokens": self.settings.max_new_tokens,
        }
        if self.settings.enable_forced_aligner:
            try:
                load_kwargs["forced_aligner"] = self.settings.forced_aligner
                load_kwargs["forced_aligner_kwargs"] = {
                    "dtype": dtype,
                    "device_map": device_map,
                }
                backend = Qwen3ASRModel.from_pretrained(self.checkpoint, **load_kwargs)
                aligner_ok = True
            except Exception as exc:
                warnings.warn(
                    f"WARNING: forced aligner failed on {self.device} ({exc}). "
                    "Falling back to ASR without the aligner.",
                    stacklevel=2,
                )
                load_kwargs.pop("forced_aligner", None)
                load_kwargs.pop("forced_aligner_kwargs", None)
                backend = Qwen3ASRModel.from_pretrained(self.checkpoint, **load_kwargs)
        else:
            backend = Qwen3ASRModel.from_pretrained(self.checkpoint, **load_kwargs)

        self._backend = backend
        self._aligner_enabled = aligner_ok
        _LOADED[key] = (backend, aligner_ok)
        return backend


def _wav_duration(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate() or 1
        return frames / float(rate)
