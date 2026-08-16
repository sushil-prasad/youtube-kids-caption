from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from app.config import Settings, load_settings, normalize_safety_mode, normalize_unknown_profanity
from app.pipeline.audio import write_json
from app.pipeline.profanity import (
    DISCLAIMER,
    SafetyHit,
    SafetyLexicon,
    analysis_payload,
    detect_profanity,
    load_lexicon,
    normalize_token,
)
from app.pipeline.timestamps import job_transcript_path, load_transcript
from app.transcript import Transcript, Word

REVIEW_MODES = {"review-only", "literal", "review"}
CENSOR_MARK = "_"


@dataclass
class SafetyDecision:
    index: int
    end_index: int
    original: str
    replacement: str
    reason: str
    confidence: float
    case: str
    action: str
    needs_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (char != other)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def phonetic_similarity(left: str, right: str) -> float:
    a = normalize_token(left)
    b = normalize_token(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    distance = levenshtein(a, b)
    ratio = 1.0 - distance / max(len(a), len(b))
    soundex_bonus = 0.15 if _soundex(a) == _soundex(b) else 0.0
    prefix = 0.1 if a[0] == b[0] else 0.0
    return max(0.0, min(1.0, ratio + soundex_bonus + prefix))


def _soundex(word: str) -> str:
    if not word:
        return ""
    letters = [ch for ch in word.upper() if ch.isalpha()]
    if not letters:
        return ""
    mapping = {
        **dict.fromkeys("BFPV", "1"),
        **dict.fromkeys("CGJKQSXZ", "2"),
        **dict.fromkeys("DT", "3"),
        **dict.fromkeys("L", "4"),
        **dict.fromkeys("MN", "5"),
        **dict.fromkeys("R", "6"),
    }
    first = letters[0]
    digits: list[str] = []
    previous = mapping.get(first, "0")
    for char in letters[1:]:
        digit = mapping.get(char)
        if not digit or digit == previous:
            if char not in "HW":
                previous = digit or previous
            continue
        digits.append(digit)
        previous = digit
        if len(digits) == 3:
            break
    return (first + "".join(digits) + "000")[:4]


def context_likelihood(prev: list[str], candidate: str, nxt: list[str], lexicon: SafetyLexicon) -> float:
    left = normalize_token(prev[-1]) if prev else ""
    right = normalize_token(nxt[0]) if nxt else ""
    cand = normalize_token(candidate)
    keys = [f"{left}|{right}", f"{left}|", f"|{right}"]
    for key in keys:
        options = lexicon.collocations.get(key) or []
        if cand in options:
            if "|" not in key.strip("|"):
                return 0.7
            if key == f"{left}|{right}":
                return 0.95
            return 0.8
    if cand in lexicon.vocabulary or cand in lexicon.allowlist:
        return 0.55
    return 0.25


def score_alternative(
    original: str,
    candidate: str,
    prev: list[str],
    nxt: list[str],
    asr_confidence: float | None,
    lexicon: SafetyLexicon,
) -> float:
    phon = phonetic_similarity(original, candidate)
    ctx = context_likelihood(prev, candidate, nxt, lexicon)
    vocab_boost = 0.1 if normalize_token(candidate) in lexicon.vocabulary | lexicon.allowlist else 0.0
    low_asr = 0.0 if asr_confidence is None else (1.0 - asr_confidence) * 0.15
    return max(0.0, min(1.0, 0.30 * phon + 0.55 * ctx + vocab_boost + low_asr))


def generate_alternatives(hit: SafetyHit, lexicon: SafetyLexicon) -> list[str]:
    token = normalize_token(hit.matched)
    prev = [normalize_token(part) for part in hit.context_before]
    nxt = [normalize_token(part) for part in hit.context_after]
    ordered: list[str] = []
    seen: set[str] = set()

    def add(word: str) -> None:
        normalized = normalize_token(word)
        if not normalized or normalized == token or normalized in seen:
            return
        if normalized in lexicon.words or normalized in lexicon.phrases:
            return
        seen.add(normalized)
        ordered.append(normalized)

    for item in lexicon.confusions.get(token, []):
        add(item)
    left = prev[-1] if prev else ""
    right = nxt[0] if nxt else ""
    for key in (f"{left}|{right}", f"{left}|", f"|{right}"):
        for item in lexicon.collocations.get(key, []):
            add(item)
    for item in list(lexicon.allowlist) + list(lexicon.vocabulary):
        if phonetic_similarity(token, item) >= 0.72:
            add(item)
    return ordered


def best_alternative(
    hit: SafetyHit,
    lexicon: SafetyLexicon,
) -> tuple[str | None, float]:
    prev = hit.context_before
    nxt = hit.context_after
    best_word: str | None = None
    best_score = 0.0
    for candidate in generate_alternatives(hit, lexicon):
        score = score_alternative(hit.matched, candidate, prev, nxt, hit.asr_confidence, lexicon)
        if score > best_score:
            best_word, best_score = candidate, score
    return best_word, best_score


def decide_safety(
    hit: SafetyHit,
    alternative: str | None,
    alt_score: float,
    settings: Settings,
) -> SafetyDecision:
    """Cases A keep / B replace / C censor with _ / D flag. Never silent when uncertain."""
    mode = normalize_safety_mode(settings.safety_mode)
    unknown = normalize_unknown_profanity(settings.unknown_profanity)
    original = hit.word
    asr = hit.asr_confidence

    if hit.allowlisted:
        return SafetyDecision(
            hit.index, hit.end_index, original, original, "allowlist", 1.0, "A", "keep", False
        )

    strong_alt = bool(alternative) and alt_score >= settings.correction_threshold
    if strong_alt and alternative:
        asr_is_high = asr is not None and asr >= settings.confidence_high
        if asr_is_high and alt_score < 0.85:
            return SafetyDecision(
                hit.index,
                hit.end_index,
                original,
                original,
                "uncertain_high_asr_confidence",
                alt_score,
                "D",
                "flag",
                True,
            )
        return SafetyDecision(
            hit.index,
            hit.end_index,
            original,
            _preserve_shape(original, alternative),
            "contextual_correction",
            round(alt_score, 4),
            "B",
            "replace",
            False,
        )

    if mode in REVIEW_MODES:
        return SafetyDecision(
            hit.index, hit.end_index, original, original, "review_mode", 0.4, "D", "flag", True
        )

    if hit.severity == "strong":
        if unknown == "keep":
            return SafetyDecision(
                hit.index, hit.end_index, original, original, "unknown_keep", 0.35, "D", "flag", True
            )
        return SafetyDecision(
            hit.index, hit.end_index, original, CENSOR_MARK, "unresolved_profanity", 0.35, "C", "censor", True
        )

    # Mild / uncertain whether profane.
    if unknown == "censor" and mode == "strict":
        return SafetyDecision(
            hit.index, hit.end_index, original, CENSOR_MARK, "strict_mild_censor", 0.3, "C", "censor", True
        )
    if unknown == "keep":
        return SafetyDecision(
            hit.index, hit.end_index, original, original, "uncertain_keep", 0.3, "D", "flag", True
        )
    return SafetyDecision(
        hit.index, hit.end_index, original, original, "uncertain_flag", 0.3, "D", "flag", True
    )


def _preserve_shape(original: str, replacement: str) -> str:
    prefix = ""
    suffix = ""
    body = original
    while body and body[0] in "\"'“”‘’(":
        prefix += body[0]
        body = body[1:]
    while body and body[-1] in "\"'“”‘’),:;!?.":
        suffix = body[-1] + suffix
        body = body[:-1]
    text = replacement
    if body[:1].isupper() and text:
        text = text[0].upper() + text[1:]
    return prefix + text + suffix


def apply_decisions(words: list[Word], decisions: list[SafetyDecision]) -> list[Word]:
    by_index = {decision.index: decision for decision in decisions if decision.action in {"replace", "censor"}}
    result: list[Word] = []
    skip_until = 0
    for index, word in enumerate(words):
        if index < skip_until:
            continue
        decision = by_index.get(index)
        if not decision:
            result.append(word)
            continue
        start = word.start
        end = words[decision.end_index - 1].end if decision.end_index - 1 < len(words) else word.end
        result.append(
            Word(
                word=decision.replacement,
                start=start,
                end=end,
                confidence=word.confidence,
                speaker=word.speaker,
            )
        )
        skip_until = decision.end_index
    return result


def correct_words(
    words: list[Word],
    settings: Settings | None = None,
    lexicon: SafetyLexicon | None = None,
    hits: list[SafetyHit] | None = None,
) -> tuple[list[Word], list[SafetyDecision], list[SafetyHit]]:
    settings = settings or load_settings()
    lexicon = lexicon or load_lexicon(settings)
    hits = hits if hits is not None else detect_profanity(words, lexicon, context_window=settings.context_window)
    decisions: list[SafetyDecision] = []
    for hit in hits:
        alternative, score = best_alternative(hit, lexicon)
        decisions.append(decide_safety(hit, alternative, score, settings))
    return apply_decisions(words, decisions), decisions, hits


def correction_from_job(
    job_dir: str | Path,
    settings: Settings | None = None,
    safety_mode: str | None = None,
    apply_policy: bool = True,
) -> Path:
    settings = settings or load_settings()
    if safety_mode:
        settings = replace(settings, safety_mode=normalize_safety_mode(safety_mode, settings.safety_mode))
    job_dir = Path(job_dir)
    source = job_transcript_path(job_dir, stage="safety")
    original = load_transcript(source)
    from app.pipeline.vocabulary import apply_vocabulary, load_vocabulary

    vocab = load_vocabulary(job_dir, settings)
    lexicon = load_lexicon(settings, job_dir=job_dir, extra_vocabulary=vocab.terms())
    prepared = apply_vocabulary(original.words, vocab)
    words, decisions, hits = correct_words(prepared, settings=settings, lexicon=lexicon)
    write_json(job_dir / "safety_analysis.json", analysis_payload(hits, lexicon, settings))
    log = {
        "disclaimer": DISCLAIMER,
        "mode": settings.safety_mode,
        "unknown_profanity": settings.unknown_profanity,
        "apply_policy": apply_policy,
        "decisions": [decision.to_dict() for decision in decisions],
        "summary": {
            "kept": sum(1 for item in decisions if item.action == "keep"),
            "replaced": sum(1 for item in decisions if item.action == "replace"),
            "censored": sum(1 for item in decisions if item.action == "censor"),
            "flagged": sum(1 for item in decisions if item.action == "flag" or item.needs_review),
        },
    }
    write_json(job_dir / "correction_log.json", log)
    if apply_policy:
        corrected_words = words
    else:
        corrected_words = list(original.words)
    corrected = Transcript(
        text=" ".join(word.word for word in corrected_words),
        words=corrected_words,
        language=original.language,
        model=original.model,
        device=original.device,
    )
    dest = job_dir / "corrected_transcript.json"
    payload = corrected.to_dict()
    payload["corrections"] = [decision.to_dict() for decision in decisions]
    payload["disclaimer"] = DISCLAIMER
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Context-aware correction and safety policy (cases A–D).")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--apply-policy", action="store_true", help="Apply keep/replace/censor/flag decisions")
    parser.add_argument("--safety-mode", default=None)
    args = parser.parse_args(argv)
    path = correction_from_job(args.job_dir, safety_mode=args.safety_mode, apply_policy=True)
    print(path)
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
