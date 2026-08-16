from __future__ import annotations

from app.config import is_apple_silicon, resolve_device
from app.models.registry import MODEL_REGISTRY, get_asr_model
from app.pipeline.orchestrator import build_parser


def test_pasketti_is_registered() -> None:
    assert "pasketti_first" in MODEL_REGISTRY
    model = get_asr_model("pasketti_first", device="cpu")
    assert model.name == "pasketti_first"


def test_unknown_model_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown ASR_MODEL"):
        get_asr_model("not_a_model")


def test_cli_help() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "video" in help_text
    assert "--output" in help_text
    assert "--device" in help_text
    assert "--skip-asr" in help_text


def test_resolve_device_never_requires_cuda() -> None:
    device = resolve_device("auto")
    assert device in {"cpu", "mps", "cuda"}
    assert resolve_device("cpu") == "cpu"
    if not is_apple_silicon():
        assert resolve_device("auto") in {"cpu", "cuda", "mps"}
