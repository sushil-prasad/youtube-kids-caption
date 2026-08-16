from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.config import ROOT, Settings, load_settings
from app.pipeline.audio import write_json
from app.pipeline.profanity import normalize_phrase, normalize_token
from app.transcript import Word

CATEGORIES = (
    "character_names",
    "people",
    "locations",
    "games",
    "toys",
    "fictional",
    "brands",
    "phrases",
    "other",
)

DEFAULT_VOCAB_PATH = ROOT / "app" / "safety" / "vocabulary.txt"
CREATOR_VOCAB_PATH = ROOT / "data" / "creator_vocabulary.json"


@dataclass
class VocabEntry:
    term: str
    category: str = "other"

    def to_dict(self) -> dict[str, str]:
        return {"term": self.term, "category": self.category}


@dataclass
class Vocabulary:
    entries: list[VocabEntry] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def terms(self) -> list[str]:
        seen: list[str] = []
        found: set[str] = set()
        for entry in self.entries:
            key = normalize_phrase(entry.term)
            if key and key not in found:
                found.add(key)
                seen.append(entry.term)
        return seen

    def asr_hints(self) -> list[str]:
        return self.terms()[:40]

    def removed_keys(self) -> set[str]:
        return {normalize_phrase(term) for term in self.removed if normalize_phrase(term)}

    def add(self, term: str, category: str = "other") -> None:
        cleaned = " ".join((term or "").split())
        if not cleaned:
            return
        category = category if category in CATEGORIES else "other"
        key = normalize_phrase(cleaned)
        self.removed = [item for item in self.removed if normalize_phrase(item) != key]
        for existing in self.entries:
            if normalize_phrase(existing.term) == key:
                existing.term = cleaned
                existing.category = category
                return
        self.entries.append(VocabEntry(cleaned, category))

    def remove(self, term: str) -> bool:
        cleaned = " ".join((term or "").split())
        key = normalize_phrase(cleaned)
        if not key:
            return False
        self.entries = [entry for entry in self.entries if normalize_phrase(entry.term) != key]
        if key not in self.removed_keys():
            self.removed.append(cleaned)
        return True

    def grouped(self) -> dict[str, list[str]]:
        groups = {name: [] for name in CATEGORIES}
        for entry in self.entries:
            groups.setdefault(entry.category, []).append(entry.term)
        return groups

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "terms": self.terms(),
            "grouped": self.grouped(),
            "removed": list(self.removed),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vocabulary:
        vocab = cls()
        for item in data.get("entries") or []:
            if isinstance(item, str):
                vocab.add(item)
            else:
                vocab.add(str(item.get("term") or ""), str(item.get("category") or "other"))
        for term in data.get("terms") or []:
            vocab.add(str(term))
        grouped = data.get("grouped") or {}
        if isinstance(grouped, dict):
            for category, terms in grouped.items():
                for term in terms or []:
                    vocab.add(str(term), str(category))
        for term in data.get("removed") or []:
            cleaned = " ".join(str(term).split())
            if cleaned and normalize_phrase(cleaned) not in vocab.removed_keys():
                vocab.removed.append(cleaned)
        return vocab


def _parse_vocab_file(path: Path) -> list[VocabEntry]:
    if not path.is_file():
        return []
    entries: list[VocabEntry] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "\t" in line:
            term, category = line.split("\t", 1)
        else:
            term, category = line, "other"
        term = " ".join(term.split())
        category = category.strip() or "other"
        if category not in CATEGORIES:
            category = "other"
        if term:
            entries.append(VocabEntry(term, category))
    return entries


def load_vocabulary(
    job_dir: str | Path | None = None,
    settings: Settings | None = None,
    extra: Iterable[str] = (),
) -> Vocabulary:
    settings = settings or load_settings()
    vocab = Vocabulary()
    default_path = Path(settings.vocabulary_path) if settings.vocabulary_path else DEFAULT_VOCAB_PATH
    for entry in _parse_vocab_file(default_path):
        vocab.add(entry.term, entry.category)
    creator_path = CREATOR_VOCAB_PATH
    creator_removed: set[str] = set()
    if creator_path.is_file():
        try:
            loaded = Vocabulary.from_dict(json.loads(creator_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            loaded = Vocabulary()
        creator_removed = loaded.removed_keys()
        for entry in loaded.entries:
            vocab.add(entry.term, entry.category)
    if job_dir:
        job = Path(job_dir)
        for entry in _parse_vocab_file(job / "vocabulary.txt"):
            vocab.add(entry.term, entry.category)
        json_path = job / "vocabulary.json"
        if json_path.is_file():
            try:
                loaded = Vocabulary.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                loaded = Vocabulary()
            for entry in loaded.entries:
                vocab.add(entry.term, entry.category)
    for term in extra:
        vocab.add(term)
    if creator_removed:
        vocab.entries = [
            entry for entry in vocab.entries if normalize_phrase(entry.term) not in creator_removed
        ]
        vocab.removed = [
            term for term in vocab.removed if normalize_phrase(term) in creator_removed
        ]
        for term in sorted(creator_removed):
            if term not in vocab.removed_keys():
                vocab.removed.append(term)
    return vocab


def _similarity(left: str, right: str) -> float:
    a = normalize_token(left)
    b = normalize_token(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if max(len(a), len(b)) <= 2:
        return 1.0 if a == b else 0.0
    # Simple edit-distance ratio without importing the correction module.
    prev = list(range(len(b) + 1))
    for i, char in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(
                min(current[j - 1] + 1, prev[j] + 1, prev[j - 1] + (char != other))
            )
        prev = current
    return 1.0 - prev[-1] / max(len(a), len(b))


def _canonical(term: str, sample: str) -> str:
    if sample.isupper():
        return term.upper()
    if sample[:1].isupper():
        return term[0].upper() + term[1:] if term else term
    # Proper names in the list keep their stored casing.
    return term


def apply_vocabulary(words: list[Word], vocab: Vocabulary, threshold: float = 0.84) -> list[Word]:
    """Replace near-miss ASR tokens with creator vocabulary. Does not rewrite unrelated words."""
    if not words or not vocab.entries:
        return list(words)
    singles = [entry for entry in vocab.entries if " " not in entry.term.strip()]
    phrases = [entry for entry in vocab.entries if " " in entry.term.strip()]
    phrases.sort(key=lambda item: len(item.term.split()), reverse=True)
    used = [False] * len(words)
    result: list[Word | None] = [None] * len(words)

    for entry in phrases:
        parts = entry.term.split()
        n = len(parts)
        for start in range(0, len(words) - n + 1):
            if any(used[start:start + n]):
                continue
            window = words[start:start + n]
            score = min(_similarity(window[i].word, parts[i]) for i in range(n))
            if score < threshold:
                continue
            result[start] = Word(
                word=_canonical(entry.term, window[0].word),
                start=window[0].start,
                end=window[-1].end,
                confidence=window[0].confidence,
                speaker=window[0].speaker,
            )
            for index in range(start, start + n):
                used[index] = True
            for index in range(start + 1, start + n):
                result[index] = None

    for index, word in enumerate(words):
        if used[index]:
            continue
        token = word.word
        best: tuple[float, VocabEntry] | None = None
        for entry in singles:
            score = _similarity(token, entry.term)
            if best is None or score > best[0]:
                best = (score, entry)
        if best and (best[0] >= 0.999 or (best[0] >= threshold and (word.confidence is None or word.confidence < 0.97))):
            result[index] = Word(
                word=_canonical(best[1].term, token),
                start=word.start,
                end=word.end,
                confidence=word.confidence,
                speaker=word.speaker,
            )
        else:
            result[index] = word
        used[index] = True

    return [item for item in result if item is not None]


def load_creator_vocabulary() -> Vocabulary:
    if not CREATOR_VOCAB_PATH.is_file():
        return Vocabulary()
    try:
        return Vocabulary.from_dict(json.loads(CREATOR_VOCAB_PATH.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return Vocabulary()


def save_creator_vocabulary(vocab: Vocabulary) -> Path:
    dest = CREATOR_VOCAB_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, vocab.to_dict())
    return dest


def vocabulary_from_job(job_dir: str | Path | None = None) -> Path | Vocabulary:
    vocab = load_vocabulary(job_dir)
    if job_dir:
        dest = Path(job_dir) / "vocabulary.json"
        write_json(dest, vocab.to_dict())
        return dest
    return vocab


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Load and write creator vocabulary lists.")
    parser.add_argument("--job-dir", default=None)
    parser.add_argument("--add", action="append", default=[], help="Add a term (repeatable)")
    parser.add_argument("--category", default="other")
    args = parser.parse_args(argv)
    vocab = load_vocabulary(args.job_dir)
    for term in args.add:
        vocab.add(term, args.category)
    if args.job_dir:
        dest = Path(args.job_dir) / "vocabulary.json"
        write_json(dest, vocab.to_dict())
        print(dest)
    grouped = vocab.grouped()
    for category in CATEGORIES:
        terms = grouped.get(category) or []
        if terms:
            print(f"{category}: {', '.join(terms)}")
    if not vocab.entries:
        print("Vocabulary is empty. Add terms with --add or vocabulary.txt.")


if __name__ == "__main__":
    main()
