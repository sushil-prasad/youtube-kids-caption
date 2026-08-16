from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.config import ROOT, Settings, load_settings
from app.pipeline.audio import write_json
from app.pipeline.timestamps import job_transcript_path, load_transcript
from app.transcript import Word

SAFETY_DIR = ROOT / "app" / "safety"
DISCLAIMER = "Captions are not guaranteed safe. Creator review is required."
_TOKEN_RE = re.compile(r"[^a-z0-9']+")
_SUFFIXES = ("ings", "ing", "in'", "ers", "er", "ies", "ied", "ed", "es", "s")


@dataclass
class SafetyLexicon:
    words: dict[str, str] = field(default_factory=dict)
    phrases: dict[str, str] = field(default_factory=dict)
    allowlist: set[str] = field(default_factory=set)
    confusions: dict[str, list[str]] = field(default_factory=dict)
    collocations: dict[str, list[str]] = field(default_factory=dict)
    vocabulary: set[str] = field(default_factory=set)

    def max_phrase_len(self) -> int:
        if not self.phrases:
            return 0
        return max(len(phrase.split()) for phrase in self.phrases)


@dataclass
class SafetyHit:
    index: int
    end_index: int
    word: str
    start: float
    end: float
    match_type: str
    matched: str
    severity: str
    allowlisted: bool
    asr_confidence: float | None
    context_before: list[str]
    context_after: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_token(token: str) -> str:
    lowered = (token or "").lower().replace("’", "'")
    cleaned = _TOKEN_RE.sub("", lowered)
    return cleaned.strip("'")


def normalize_phrase(phrase: str) -> str:
    return " ".join(part for part in (normalize_token(token) for token in phrase.split()) if part)


def _parse_list_file(path: Path, *, phrases: bool = False) -> list[tuple[str, str]]:
    if not path.is_file():
        return []
    rows: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "\t" in line:
            token, severity = line.split("\t", 1)
        else:
            token, severity = line, "strong"
        token = normalize_phrase(token) if phrases else normalize_token(token)
        severity = (severity or "strong").strip().lower() or "strong"
        if severity not in {"strong", "mild"}:
            severity = "strong"
        if token:
            rows.append((token, severity))
    return rows


def _default_path(configured: str | None, filename: str) -> Path:
    if configured:
        return Path(configured)
    return SAFETY_DIR / filename


def load_allowlist(
    settings: Settings | None = None,
    extra: Iterable[str] = (),
    job_dir: str | Path | None = None,
) -> list[str]:
    settings = settings or load_settings()
    path = _default_path(settings.allowlist_path, "allowlist.txt")
    names: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        normalized = normalize_token(token)
        if normalized and normalized not in seen:
            seen.add(normalized)
            names.append(normalized)

    for token, _severity in _parse_list_file(path):
        add(token)
    for token in settings.allowlist:
        add(token)
    if job_dir:
        add_path = Path(job_dir) / "allowlist.txt"
        for token, _severity in _parse_list_file(add_path):
            add(token)
    for token in extra:
        add(token)
    return names


def load_lexicon(
    settings: Settings | None = None,
    extra_allowlist: Iterable[str] = (),
    extra_vocabulary: Iterable[str] = (),
    job_dir: str | Path | None = None,
) -> SafetyLexicon:
    settings = settings or load_settings()
    words = {token: severity for token, severity in _parse_list_file(_default_path(settings.profanity_words_path, "profanity_words.txt"))}
    phrases = {
        token: severity
        for token, severity in _parse_list_file(
            _default_path(settings.profanity_phrases_path, "profanity_phrases.txt"),
            phrases=True,
        )
    }
    allowlist = set(load_allowlist(settings, extra=extra_allowlist, job_dir=job_dir))
    confusions: dict[str, list[str]] = {}
    collocations: dict[str, list[str]] = {}
    corrections_path = _default_path(settings.corrections_path, "corrections.json")
    if corrections_path.is_file():
        data = json.loads(corrections_path.read_text(encoding="utf-8"))
        confusions = {
            normalize_token(key): [normalize_token(item) for item in values if normalize_token(item)]
            for key, values in (data.get("confusions") or {}).items()
        }
        collocations = {
            key.strip().lower(): [normalize_token(item) for item in values if normalize_token(item)]
            for key, values in (data.get("collocations") or {}).items()
        }
    vocabulary = {normalize_token(item) for item in extra_vocabulary if normalize_token(item)}
    vocab_path = settings.vocabulary_path
    if vocab_path:
        for token, _severity in _parse_list_file(Path(vocab_path)):
            vocabulary.add(token)
    return SafetyLexicon(
        words=words,
        phrases=phrases,
        allowlist=allowlist,
        confusions=confusions,
        collocations=collocations,
        vocabulary=vocabulary,
    )


def _stems(token: str) -> list[str]:
    found = [token]
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            found.append(token[: -len(suffix)])
    return found


def _severity_for_token(token: str, lexicon: SafetyLexicon) -> str | None:
    if not token:
        return None
    for stem in _stems(token):
        if stem in lexicon.words:
            return lexicon.words[stem]
    return None


def _matched_stem(token: str, lexicon: SafetyLexicon) -> str | None:
    for stem in _stems(token):
        if stem in lexicon.words:
            return stem
    return None


def detect_profanity(
    words: list[Word],
    lexicon: SafetyLexicon | None = None,
    context_window: int = 5,
) -> list[SafetyHit]:
    """Exact word and phrase matches only. Never substring-search inside a token."""
    lexicon = lexicon or load_lexicon()
    window = max(1, context_window)
    tokens = [normalize_token(word.word) for word in words]
    covered = [False] * len(words)
    hits: list[SafetyHit] = []

    max_n = lexicon.max_phrase_len()
    for length in range(max_n, 1, -1):
        for start in range(0, len(tokens) - length + 1):
            if any(covered[start:start + length]):
                continue
            phrase = " ".join(tokens[start:start + length])
            severity = lexicon.phrases.get(phrase)
            if not severity:
                continue
            allowlisted = any(tokens[i] in lexicon.allowlist for i in range(start, start + length) if tokens[i])
            for i in range(start, start + length):
                covered[i] = True
            last = start + length - 1
            confidences = [words[i].confidence for i in range(start, last + 1) if words[i].confidence is not None]
            mean_conf = (sum(confidences) / len(confidences)) if confidences else None
            hits.append(
                SafetyHit(
                    index=start,
                    end_index=last + 1,
                    word=" ".join(word.word for word in words[start:last + 1]),
                    start=words[start].start,
                    end=words[last].end,
                    match_type="phrase",
                    matched=phrase,
                    severity=severity,
                    allowlisted=allowlisted,
                    asr_confidence=mean_conf,
                    context_before=[words[i].word for i in range(max(0, start - window), start)],
                    context_after=[words[i].word for i in range(last + 1, min(len(words), last + 1 + window))],
                )
            )

    for index, token in enumerate(tokens):
        if covered[index] or not token:
            continue
        severity = _severity_for_token(token, lexicon)
        if not severity:
            continue
        matched = _matched_stem(token, lexicon) or token
        allowlisted = token in lexicon.allowlist or matched in lexicon.allowlist
        hits.append(
            SafetyHit(
                index=index,
                end_index=index + 1,
                word=words[index].word,
                start=words[index].start,
                end=words[index].end,
                match_type="word",
                matched=matched,
                severity=severity,
                allowlisted=allowlisted,
                asr_confidence=words[index].confidence,
                context_before=[words[i].word for i in range(max(0, index - window), index)],
                context_after=[words[i].word for i in range(index + 1, min(len(words), index + 1 + window))],
            )
        )
    hits.sort(key=lambda hit: hit.index)
    return hits


def analysis_payload(
    hits: list[SafetyHit],
    lexicon: SafetyLexicon,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    return {
        "mode": settings.safety_mode,
        "unknown_profanity": settings.unknown_profanity,
        "language": settings.language,
        "allowlist": sorted(lexicon.allowlist),
        "disclaimer": DISCLAIMER,
        "hits": [hit.to_dict() for hit in hits],
        "summary": {
            "hit_count": len(hits),
            "word_hits": sum(1 for hit in hits if hit.match_type == "word"),
            "phrase_hits": sum(1 for hit in hits if hit.match_type == "phrase"),
            "allowlisted": sum(1 for hit in hits if hit.allowlisted),
            "strong": sum(1 for hit in hits if hit.severity == "strong" and not hit.allowlisted),
            "mild": sum(1 for hit in hits if hit.severity == "mild" and not hit.allowlisted),
        },
    }


def profanity_from_job(job_dir: str | Path, settings: Settings | None = None) -> Path:
    settings = settings or load_settings()
    job_dir = Path(job_dir)
    transcript = load_transcript(job_transcript_path(job_dir, stage="safety"))
    lexicon = load_lexicon(settings, job_dir=job_dir)
    hits = detect_profanity(transcript.words, lexicon, context_window=settings.context_window)
    dest = job_dir / "safety_analysis.json"
    write_json(dest, analysis_payload(hits, lexicon, settings))
    return dest


def format_allowlist(lexicon: SafetyLexicon | None = None) -> str:
    lexicon = lexicon or load_lexicon()
    names = ", ".join(sorted(lexicon.allowlist)) or "(empty)"
    return f"Allowlist ({len(lexicon.allowlist)} words): {names}"


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Detect potentially unsafe words and phrases.")
    parser.add_argument("--job-dir", default=None)
    parser.add_argument("--allowlist", action="store_true", help="Print the merged creator allowlist")
    args = parser.parse_args(argv)
    settings = load_settings()
    lexicon = load_lexicon(settings, job_dir=args.job_dir)
    if args.allowlist:
        print(format_allowlist(lexicon))
        if not args.job_dir:
            return
    if not args.job_dir:
        parser.error("--job-dir is required unless --allowlist is set")
    path = profanity_from_job(args.job_dir, settings=settings)
    print(path)


if __name__ == "__main__":
    main()
