from __future__ import annotations

from app.config import Settings, load_settings, resolve_device
from app.models.base import ASRModel
from app.models.pasketti import PaskettiFirstModel

MODEL_REGISTRY: dict[str, type[ASRModel]] = {
    "pasketti_first": PaskettiFirstModel,
}

_INSTANCES: dict[tuple[str, str], ASRModel] = {}


def get_asr_model(name: str | None = None, device: str | None = None, settings: Settings | None = None) -> ASRModel:
    settings = settings or load_settings()
    model_name = name or settings.asr_model
    cls = MODEL_REGISTRY.get(model_name)
    if cls is None:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown ASR_MODEL={model_name!r}. Registered: {known}")
    resolved = resolve_device(device or settings.device)
    cache_key = (model_name, resolved)
    if cache_key not in _INSTANCES:
        _INSTANCES[cache_key] = cls(settings=settings, device=resolved)
    return _INSTANCES[cache_key]
