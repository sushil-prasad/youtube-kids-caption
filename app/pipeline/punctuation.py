from __future__ import annotations

from pathlib import Path

from app.config import load_settings
from app.pipeline.timestamps import load_transcript, write_word_timestamps
from app.transcript import Transcript, Word

GREETING_PREFIXES = {
    ("hey", "guys"),
    ("hey", "everybody"),
    ("hey", "everyone"),
    ("hi", "guys"),
    ("hi", "everyone"),
    ("hi", "everybody"),
    ("hello", "everyone"),
    ("hello", "everybody"),
    ("hello", "friends"),
}

QUESTION_STARTERS = {
    "what",
    "what's",
    "whats",
    "why",
    "how",
    "who",
    "where",
    "when",
    "which",
    "whose",
    "whom",
    "is",
    "are",
    "am",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "will",
    "won't",
    "wont",
    "shall",
    "should",
    "isn't",
    "isnt",
    "aren't",
    "arent",
    "don't",
    "dont",
    "didn't",
    "didnt",
    "can't",
    "cant",
    "couldn't",
    "couldnt",
    "wouldn't",
    "wouldnt",
    "wasn't",
    "wasnt",
    "weren't",
    "werent",
    "was",
    "were",
    "has",
    "have",
    "had",
}

I_FORMS = {"i", "i'm", "i've", "i'd", "i'll", "im", "ive", "id", "ill"}
TERMINALS = ".!?"


def _norm(token: str) -> str:
    return token.lower().strip("\"“”‘’'()[]{}.,!?;:")


def _has_terminal(token: str) -> bool:
    stripped = token.rstrip("\"'”’")
    return bool(stripped) and stripped[-1] in TERMINALS


def _capitalize_token(token: str) -> str:
    chars = list(token)
    for index, char in enumerate(chars):
        if char.isalpha():
            chars[index] = char.upper()
            break
    return "".join(chars)


def _capitalize_i(token: str) -> str:
    core = token.strip("\"“”‘’()[]{}.,!?;:")
    if core.lower() in I_FORMS or core.lower() == "i":
        return _capitalize_token(token)
    return token


def _attach_terminal(token: str, mark: str) -> str:
    trailing_quote = ""
    body = token
    while body and body[-1] in "\"'”’)":
        trailing_quote = body[-1] + trailing_quote
        body = body[:-1]
    body = body.rstrip(",;:")
    if body and body[-1] in TERMINALS:
        return body + trailing_quote
    return body + mark + trailing_quote


def _clone_word(word: Word, text: str | None = None) -> Word:
    return Word(
        word=word.word if text is None else text,
        start=word.start,
        end=word.end,
        confidence=word.confidence,
        speaker=word.speaker,
    )


def _is_greeting_pair(first: Word, second: Word) -> bool:
    return (_norm(first.word), _norm(second.word)) in GREETING_PREFIXES


def _split_sentences(words: list[Word], pause_gap: float) -> list[list[Word]]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current:
            gap = word.start - current[-1].end
            if gap >= pause_gap or _has_terminal(current[-1].word):
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)

    expanded: list[list[Word]] = []
    for group in groups:
        if len(group) >= 3 and _is_greeting_pair(group[0], group[1]):
            expanded.append(group[:2])
            expanded.append(group[2:])
        else:
            expanded.append(group)
    return expanded


def _sentence_mark(group: list[Word]) -> str:
    if not group:
        return "."
    if len(group) <= 2 and (
        _norm(group[0].word) in {"hey", "hi", "hello", "wow", "yay"}
        or (len(group) == 2 and _is_greeting_pair(group[0], group[1]))
    ):
        return "!"
    first = _norm(group[0].word)
    if first in QUESTION_STARTERS:
        return "?"
    return "."


def punctuate_words(words: list[Word], pause_gap: float | None = None) -> list[Word]:
    """Add sentence punctuation and capitalization without rewriting word stems."""
    if not words:
        return []
    if pause_gap is None:
        pause_gap = load_settings().punctuation_pause
    result: list[Word] = []
    for group in _split_sentences(words, pause_gap):
        mark = _sentence_mark(group)
        for index, word in enumerate(group):
            text = word.word
            original_core = _norm(text)
            if index == 0:
                text = _capitalize_token(text)
            else:
                text = _capitalize_i(text)
            if index == len(group) - 1:
                text = _attach_terminal(text, mark)
            # Never change non-punctuation letters besides the first-letter / "I" cases above.
            stem = _norm(text)
            if stem and original_core and stem != original_core:
                # Apostrophe-insensitive I-forms and capitalization only.
                if original_core not in I_FORMS and original_core != "i":
                    text = word.word
                    if index == 0:
                        text = _capitalize_token(text)
                    if index == len(group) - 1:
                        text = _attach_terminal(text, mark)
            result.append(_clone_word(word, text))
    return result


def punctuate_transcript(transcript: Transcript, pause_gap: float | None = None) -> Transcript:
    words = punctuate_words(transcript.words, pause_gap=pause_gap)
    text = " ".join(word.word for word in words) if words else transcript.text
    return Transcript(
        text=text,
        words=words,
        language=transcript.language,
        model=transcript.model,
        device=transcript.device,
    )


def punctuate_job(job_dir: str | Path) -> Path:
    job_dir = Path(job_dir)
    timestamps_path = job_dir / "word_timestamps.json"
    if not timestamps_path.is_file():
        raise FileNotFoundError(f"Missing {timestamps_path}. Run timestamps first.")
    original = load_transcript(timestamps_path)
    punctuated = punctuate_transcript(original)
    dest = job_dir / "punctuated_transcript.json"
    write_word_timestamps(punctuated, dest)
    return dest


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Punctuate and capitalize a transcript. Raw ASR is left unchanged.")
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    path = punctuate_job(args.job_dir)
    print(path)


if __name__ == "__main__":
    main()
