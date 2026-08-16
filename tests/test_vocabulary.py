from __future__ import annotations

from pathlib import Path

from app.pipeline.vocabulary import Vocabulary, apply_vocabulary, load_vocabulary
from app.transcript import Word


def test_default_vocabulary_includes_readme_examples() -> None:
    vocab = load_vocabulary()
    terms = {term.lower() for term in vocab.terms()}
    assert "bluey" in terms
    assert "fluffernoodle" in terms
    assert "rainbow castle" in terms


def test_add_and_group_terms() -> None:
    vocab = Vocabulary()
    vocab.add("Bluey", "character_names")
    vocab.add("Rainbow Castle", "fictional")
    grouped = vocab.grouped()
    assert "Bluey" in grouped["character_names"]
    assert "Rainbow Castle" in grouped["fictional"]


def test_apply_vocabulary_canonicalizes_near_misses() -> None:
    vocab = Vocabulary()
    vocab.add("Bluey", "character_names")
    vocab.add("Fluffernoodle", "fictional")
    words = [
        Word("bluey", 0.0, 0.3, 0.6),
        Word("found", 0.3, 0.5, 0.9),
        Word("fluffernoddle", 0.5, 1.0, 0.55),
    ]
    result = apply_vocabulary(words, vocab)
    assert result[0].word == "Bluey"
    assert result[1].word == "found"
    assert result[2].word == "Fluffernoodle"


def test_apply_vocabulary_does_not_rewrite_unrelated_words() -> None:
    vocab = load_vocabulary()
    words = [Word("Hello", 0.0, 0.3, 1.0), Word("world", 0.3, 0.7, 1.0)]
    result = apply_vocabulary(words, vocab)
    assert [word.word for word in result] == ["Hello", "world"]


def test_job_vocabulary_merges_creator_list(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "vocabulary.txt").write_text("Liam\tpeople\n", encoding="utf-8")
    vocab = load_vocabulary(job)
    terms = {term.lower() for term in vocab.terms()}
    assert "liam" in terms
    assert "bluey" in terms
