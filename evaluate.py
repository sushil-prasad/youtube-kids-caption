"""Compare caption pipelines on a shared evaluation set.

Does not assume a competition leaderboard ranking equals YouTube children's-content quality.
Captions are not guaranteed safe. Fixture evaluation never loads the ASR model.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app import __version__
from app.config import ROOT, Settings, load_settings
from app.pipeline.audio import write_json
from app.pipeline.confidence import classify_confidence, estimate_confidence
from app.pipeline.correction import CENSOR_MARK, SafetyDecision, apply_decisions, correct_words
from app.pipeline.profanity import DISCLAIMER, detect_profanity, load_lexicon, normalize_token
from app.pipeline.punctuation import punctuate_words
from app.pipeline.segmentation import build_captions
from app.pipeline.vocabulary import apply_vocabulary, load_vocabulary
from app.transcript import Caption, Word

NOTE = (
    "Competition ranking is not the same as YouTube children's-content quality. "
    "Automatic captions require creator review."
)

_WORD_RE = re.compile(r"[a-z0-9']+")
TIMESTAMP_TOLERANCE_S = 0.20

ABLATION_CONFIGS = {
    "asr_only": {
        "punctuation": False,
        "vocabulary": False,
        "profanity": False,
        "correction": False,
        "segmentation": True,
    },
    "asr_punctuation": {
        "punctuation": True,
        "vocabulary": False,
        "profanity": False,
        "correction": False,
        "segmentation": True,
    },
    "asr_profanity": {
        "punctuation": False,
        "vocabulary": False,
        "profanity": True,
        "correction": False,
        "segmentation": True,
    },
    "asr_correction": {
        "punctuation": False,
        "vocabulary": True,
        "profanity": False,
        "correction": True,
        "segmentation": True,
    },
    "asr_profanity_correction": {
        "punctuation": False,
        "vocabulary": True,
        "profanity": True,
        "correction": True,
        "segmentation": True,
    },
    "full_pipeline": {
        "punctuation": True,
        "vocabulary": True,
        "profanity": True,
        "correction": True,
        "segmentation": True,
    },
}


@dataclass
class Clip:
    clip_id: str
    reference_text: str
    reference_words: list[Word]
    hypothesis_words: list[Word]
    gold_profane: list[str] = field(default_factory=list)
    gold_clean: list[str] = field(default_factory=list)
    gold_corrections: list[dict[str, str]] = field(default_factory=list)


def levenshtein(left: list[str] | str, right: list[str] | str) -> int:
    a = list(left)
    b = list(right)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, token in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(
                min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (token != other))
            )
        previous = current
    return previous[-1]


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower().replace("’", "'"))


def word_error_rate(reference: str, hypothesis: str) -> dict[str, float | int]:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    distance = levenshtein(ref, hyp)
    denom = max(len(ref), 1)
    return {
        "wer": round(distance / denom, 4),
        "distance": distance,
        "reference_words": len(ref),
        "hypothesis_words": len(hyp),
    }


def character_error_rate(reference: str, hypothesis: str) -> dict[str, float | int]:
    ref = "".join(tokenize(reference))
    hyp = "".join(tokenize(hypothesis))
    distance = levenshtein(ref, hyp)
    denom = max(len(ref), 1)
    return {
        "cer": round(distance / denom, 4),
        "distance": distance,
        "reference_chars": len(ref),
        "hypothesis_chars": len(hyp),
    }


def timestamp_accuracy(
    reference: list[Word],
    hypothesis: list[Word],
    tolerance: float = TIMESTAMP_TOLERANCE_S,
) -> dict[str, float | int]:
    """Onset/offset error for identically aligned tokens (Levenshtein traceback)."""
    ref = [(normalize_token(word.word), word.start, word.end) for word in reference]
    hyp = [(normalize_token(word.word), word.start, word.end) for word in hypothesis]
    aligned = _align_tokens(ref, hyp)
    onset: list[float] = []
    offset: list[float] = []
    within = 0
    compared = 0
    for left, right in aligned:
        if left is None or right is None or left[0] != right[0] or not left[0]:
            continue
        compared += 1
        start_err = abs(left[1] - right[1])
        end_err = abs(left[2] - right[2])
        onset.append(start_err)
        offset.append(end_err)
        if start_err <= tolerance and end_err <= tolerance:
            within += 1
    return {
        "pairs_compared": compared,
        "mean_onset_error_s": round(sum(onset) / len(onset), 4) if onset else 0.0,
        "mean_offset_error_s": round(sum(offset) / len(offset), 4) if offset else 0.0,
        "within_200ms": round(within / compared, 4) if compared else 1.0,
        "tolerance_s": tolerance,
    }


def _align_tokens(
    reference: list[tuple[str, float, float]],
    hypothesis: list[tuple[str, float, float]],
) -> list[tuple[tuple[str, float, float] | None, tuple[str, float, float] | None]]:
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1][0] == hypothesis[j - 1][0] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    i, j = n, m
    pairs: list[tuple[tuple[str, float, float] | None, tuple[str, float, float] | None]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
            0 if reference[i - 1][0] == hypothesis[j - 1][0] else 1
        ):
            pairs.append((reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((reference[i - 1], None))
            i -= 1
        else:
            pairs.append((None, hypothesis[j - 1] if j > 0 else None))
            j -= 1
    pairs.reverse()
    return pairs


def low_confidence_rate(words: list[Word], settings: Settings | None = None) -> dict[str, float | int]:
    settings = settings or load_settings()
    if not words:
        return {"low_confidence_rate": 0.0, "low": 0, "total": 0}
    low = 0
    for word in words:
        score = settings.confidence_unknown if word.confidence is None else float(word.confidence)
        if classify_confidence(score, settings.confidence_high, settings.confidence_medium) == "low":
            low += 1
    return {
        "low_confidence_rate": round(low / len(words), 4),
        "low": low,
        "total": len(words),
    }


def safety_rates(
    hits_matched: list[str],
    gold_profane: list[str],
    gold_clean: list[str],
) -> dict[str, float | int]:
    flagged = {normalize_token(item) for item in hits_matched if normalize_token(item)}
    profane = {normalize_token(item) for item in gold_profane if normalize_token(item)}
    clean = {normalize_token(item) for item in gold_clean if normalize_token(item)}
    true_pos = flagged & profane
    false_pos = flagged & clean
    false_neg = profane - flagged
    fp_denom = max(len(clean), 1)
    fn_denom = max(len(profane), 1)
    return {
        "profanity_false_positive_rate": round(len(false_pos) / fp_denom, 4) if clean else 0.0,
        "profanity_false_negative_rate": round(len(false_neg) / fn_denom, 4) if profane else 0.0,
        "true_positives": len(true_pos),
        "false_positives": len(false_pos),
        "false_negatives": len(false_neg),
        "gold_profane": len(profane),
        "gold_clean": len(clean),
        "flagged": len(flagged),
    }


def correction_accuracy(
    words: list[Word],
    gold_corrections: list[dict[str, str]],
) -> dict[str, float | int]:
    if not gold_corrections:
        return {"correction_accuracy": 1.0, "correct": 0, "total": 0}
    present = {normalize_token(word.word) for word in words}
    correct = 0
    for item in gold_corrections:
        target = normalize_token(item.get("to") or "")
        source = normalize_token(item.get("from") or "")
        if target and target in present:
            correct += 1
        elif source and source not in present and not target:
            correct += 1
    return {
        "correction_accuracy": round(correct / len(gold_corrections), 4),
        "correct": correct,
        "total": len(gold_corrections),
    }


def censorship_rate(words: list[Word]) -> dict[str, float | int]:
    if not words:
        return {"censorship_rate": 0.0, "censored": 0, "total": 0}
    censored = sum(1 for word in words if word.word.strip() == CENSOR_MARK)
    return {
        "censorship_rate": round(censored / len(words), 4),
        "censored": censored,
        "total": len(words),
    }


def reading_speed_metrics(captions: list[Caption]) -> dict[str, float | int]:
    if not captions:
        return {"mean_cps": 0.0, "mean_wps": 0.0, "reading_speed_warnings": 0, "caption_count": 0}
    warnings = sum(1 for caption in captions if caption.reading_status != "OK" or "reading_speed" in caption.flags)
    return {
        "mean_cps": round(sum(caption.cps for caption in captions) / len(captions), 4),
        "mean_wps": round(sum(caption.wps for caption in captions) / len(captions), 4),
        "reading_speed_warnings": warnings,
        "caption_count": len(captions),
    }


def _words_from_payload(items: list[dict[str, Any]] | None) -> list[Word]:
    words: list[Word] = []
    for item in items or []:
        words.append(
            Word(
                word=str(item.get("word") or ""),
                start=float(item.get("start") or 0.0),
                end=float(item.get("end") or 0.0),
                confidence=item.get("confidence"),
                speaker=item.get("speaker"),
            )
        )
    return words


def load_dataset(path: str | Path) -> list[Clip]:
    root = Path(path)
    if root.is_file() and root.suffix.lower() == ".json" and root.name != "manifest.json":
        return [_clip_from_dict(json.loads(root.read_text(encoding="utf-8")), root.stem)]
    manifest_path = root / "manifest.json" if root.is_dir() else root
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No evaluation dataset at {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    clips: list[Clip] = []
    for entry in manifest.get("clips") or []:
        if isinstance(entry, str):
            clip_path = (base / entry).resolve()
            data = json.loads(clip_path.read_text(encoding="utf-8"))
            clips.append(_clip_from_dict(data, clip_path.stem))
        else:
            clips.append(_clip_from_dict(entry, str(entry.get("id") or f"clip_{len(clips)+1}")))
    if not clips:
        raise ValueError(f"Dataset {path} has no clips")
    return clips


def _clip_from_dict(data: dict[str, Any], fallback_id: str) -> Clip:
    reference = data.get("reference") or {}
    hypothesis = data.get("hypothesis") or {}
    safety = data.get("safety") or {}
    ref_text = str(reference.get("text") or data.get("reference_text") or "")
    ref_words = _words_from_payload(reference.get("words"))
    hyp_words = _words_from_payload(hypothesis.get("words") or data.get("words"))
    if not hyp_words and hypothesis.get("text"):
        hyp_words = _words_from_payload(
            [{"word": token, "start": i * 0.3, "end": i * 0.3 + 0.25, "confidence": 0.9} for i, token in enumerate(str(hypothesis["text"]).split())]
        )
    return Clip(
        clip_id=str(data.get("id") or fallback_id),
        reference_text=ref_text,
        reference_words=ref_words,
        hypothesis_words=hyp_words,
        gold_profane=list(safety.get("profane") or data.get("gold_profane") or []),
        gold_clean=list(safety.get("clean") or data.get("gold_clean") or []),
        gold_corrections=list(safety.get("corrections") or data.get("gold_corrections") or []),
    )


def apply_pipeline(
    words: list[Word],
    config: dict[str, bool],
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    working = list(words)
    if config.get("punctuation"):
        working = punctuate_words(working)
    vocab = load_vocabulary() if (config.get("vocabulary") or config.get("correction")) else None
    if config.get("vocabulary") and vocab is not None:
        working = apply_vocabulary(working, vocab)
    hits = []
    decisions: list[SafetyDecision] = []
    extra = vocab.terms() if vocab is not None else ()
    lexicon = load_lexicon(settings, extra_vocabulary=extra)
    if config.get("profanity") or config.get("correction"):
        hits = detect_profanity(working, lexicon, context_window=settings.context_window)
        if config.get("correction"):
            working, decisions, hits = correct_words(working, settings=settings, lexicon=lexicon, hits=hits)
        elif config.get("profanity"):
            for hit in hits:
                if hit.allowlisted:
                    continue
                decisions.append(
                    SafetyDecision(
                        hit.index,
                        hit.end_index,
                        hit.word,
                        CENSOR_MARK,
                        "profanity_filter",
                        0.0,
                        "C",
                        "censor",
                        True,
                    )
                )
            working = apply_decisions(working, decisions)
    confidence = estimate_confidence(working, settings)
    captions: list[Caption] = []
    if config.get("segmentation"):
        captions = build_captions(working, settings=settings, confidence_payload=confidence)
    text = " ".join(word.word for word in working)
    non_allowlisted = [hit.matched for hit in hits if not hit.allowlisted]
    return {
        "text": text,
        "words": working,
        "captions": captions,
        "hits": hits,
        "hit_terms": non_allowlisted,
        "decisions": decisions,
        "confidence": confidence,
    }


def score_clip(clip: Clip, produced: dict[str, Any], settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    hyp_text = produced["text"]
    wer = word_error_rate(clip.reference_text, hyp_text)
    cer = character_error_rate(clip.reference_text, hyp_text)
    metrics = {
        "wer": wer["wer"],
        "distance": wer["distance"],
        "reference_words": wer["reference_words"],
        "hypothesis_words": wer["hypothesis_words"],
        "cer": cer["cer"],
        "cer_distance": cer["distance"],
        "reference_chars": cer["reference_chars"],
        **timestamp_accuracy(clip.reference_words or clip.hypothesis_words, produced["words"]),
        **low_confidence_rate(produced["words"], settings),
        **safety_rates(produced["hit_terms"], clip.gold_profane, clip.gold_clean),
        **correction_accuracy(produced["words"], clip.gold_corrections),
        **censorship_rate(produced["words"]),
        **reading_speed_metrics(produced["captions"]),
    }
    return {
        "id": clip.clip_id,
        "hypothesis": hyp_text,
        "reference": clip.reference_text,
        "metrics": metrics,
    }


def aggregate(clip_results: list[dict[str, Any]]) -> dict[str, Any]:
    wer_dist = sum(int(item["metrics"]["distance"]) for item in clip_results)
    wer_ref = sum(int(item["metrics"]["reference_words"]) for item in clip_results)
    cer_dist = sum(int(item["metrics"].get("distance", 0)) for item in clip_results)
    # CER uses the same key "distance" as WER in each clip dict — they overwrite.
    # Store cer distance separately when scoring. Fix score_clip to nest wer/cer.
    return {
        "clips": len(clip_results),
        "wer": round(wer_dist / max(wer_ref, 1), 4),
        "reference_words": wer_ref,
        "mean_low_confidence_rate": _mean(clip_results, "low_confidence_rate"),
        "mean_profanity_false_positive_rate": _mean(clip_results, "profanity_false_positive_rate"),
        "mean_profanity_false_negative_rate": _mean(clip_results, "profanity_false_negative_rate"),
        "mean_correction_accuracy": _mean(clip_results, "correction_accuracy"),
        "mean_censorship_rate": _mean(clip_results, "censorship_rate"),
        "mean_cps": _mean(clip_results, "mean_cps"),
        "reading_speed_warnings": sum(int(item["metrics"]["reading_speed_warnings"]) for item in clip_results),
        "mean_timestamp_within_200ms": _mean(clip_results, "within_200ms"),
        "mean_onset_error_s": _mean(clip_results, "mean_onset_error_s"),
    }


def _mean(clip_results: list[dict[str, Any]], key: str) -> float:
    values = [float(item["metrics"][key]) for item in clip_results if key in item["metrics"]]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def evaluate_config(
    clips: list[Clip],
    config_name: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    config = ABLATION_CONFIGS[config_name]
    results = []
    cer_dist = 0
    cer_ref = 0
    for clip in clips:
        produced = apply_pipeline(clip.hypothesis_words, config, settings)
        scored = score_clip(clip, produced, settings)
        results.append(scored)
        cer_dist += int(scored["metrics"]["cer_distance"])
        cer_ref += int(scored["metrics"]["reference_chars"])
    summary = aggregate(results)
    summary["cer"] = round(cer_dist / max(cer_ref, 1), 4)
    summary["config"] = config_name
    return {
        "config": config_name,
        "summary": summary,
        "clips": results,
        "disclaimer": DISCLAIMER,
        "note": NOTE,
    }


def format_report(payload: dict[str, Any]) -> str:
    lines = [
        "Caption evaluation",
        f"Model: {payload.get('model', 'pasketti_first')}",
        f"Pipeline version: {payload.get('pipeline_version', __version__)}",
        NOTE,
        DISCLAIMER,
        "",
    ]
    reports = payload.get("reports") or [payload]
    for report in reports:
        summary = report.get("summary") or report
        lines.append(f"[{report.get('config') or summary.get('config') or 'evaluate'}]")
        lines.append(f"WER: {summary.get('wer', 0):.4f}")
        lines.append(f"CER: {summary.get('cer', 0):.4f}")
        lines.append(f"word timestamp accuracy (within 200ms): {summary.get('mean_timestamp_within_200ms', 0):.4f}")
        lines.append(f"low-confidence rate: {summary.get('mean_low_confidence_rate', 0):.4f}")
        lines.append(f"profanity false-positive rate: {summary.get('mean_profanity_false_positive_rate', 0):.4f}")
        lines.append(f"profanity false-negative rate: {summary.get('mean_profanity_false_negative_rate', 0):.4f}")
        lines.append(f"correction accuracy: {summary.get('mean_correction_accuracy', 0):.4f}")
        lines.append(f"censorship rate: {summary.get('mean_censorship_rate', 0):.4f}")
        lines.append(f"caption reading speed (mean CPS): {summary.get('mean_cps', 0):.2f}")
        lines.append(f"reading-speed warnings: {summary.get('reading_speed_warnings', 0)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_evaluation(
    dataset: str | Path,
    model: str = "pasketti_first",
    ablation: bool = False,
    output: str | Path | None = None,
) -> dict[str, Any]:
    clips = load_dataset(dataset)
    names = list(ABLATION_CONFIGS) if ablation else ["full_pipeline"]
    reports = [evaluate_config(clips, name) for name in names]
    payload = {
        "model": model,
        "dataset": str(dataset),
        "pipeline_version": __version__,
        "disclaimer": DISCLAIMER,
        "note": NOTE,
        "ablation": ablation,
        "reports": reports,
    }
    dest = Path(output) if output else ROOT / "outputs" / "evaluation" / ("ablation.json" if ablation else "metrics.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, payload)
    payload["output"] = str(dest)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python evaluate.py",
        description="Evaluate caption quality on a shared dataset. Does not load ASR unless you pass --run-asr (not used on the fixture set).",
    )
    parser.add_argument("--dataset", default="evaluation_data", help="Dataset directory or clip JSON")
    parser.add_argument("--model", default="pasketti_first", help="Label for the report (registry name)")
    parser.add_argument("--ablation", action="store_true", help="ASR-only through full pipeline")
    parser.add_argument("--output", default=None, help="Where to write the JSON report")
    parser.add_argument("--run-asr", action="store_true", help="Reserved; fixture evaluation never calls ASR")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run_asr:
        print("Fixture evaluation does not run ASR. Pass hypothesis words in the dataset JSON.")
        print("To transcribe a real clip: python -m app.pipeline VIDEO")
    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = (ROOT / dataset) if not dataset.exists() else dataset
    payload = run_evaluation(dataset, model=args.model, ablation=args.ablation, output=args.output)
    print(format_report(payload))
    print(payload["output"])


if __name__ == "__main__":
    main()
