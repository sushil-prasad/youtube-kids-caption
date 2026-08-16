from __future__ import annotations

from app.pipeline.profanity import SafetyLexicon, detect_profanity, load_allowlist, load_lexicon, normalize_token
from app.transcript import Word


def _words(*tokens: str, confidence: float = 0.9) -> list[Word]:
    words: list[Word] = []
    cursor = 0.0
    for token in tokens:
        words.append(Word(token, cursor, cursor + 0.3, confidence))
        cursor += 0.3
    return words


def test_normalize_strips_punctuation_and_case() -> None:
    assert normalize_token("Ass!") == "ass"
    assert normalize_token("SCUNTHORPE") == "scunthorpe"


def test_exact_word_match_is_case_and_punct_insensitive() -> None:
    lexicon = SafetyLexicon(words={"badword": "strong"})
    hits = detect_profanity(_words("What", "a", "Badword!"), lexicon)
    assert len(hits) == 1
    assert hits[0].match_type == "word"
    assert hits[0].matched == "badword"
    assert hits[0].severity == "strong"


def test_phrase_match() -> None:
    lexicon = SafetyLexicon(phrases={"what the heck": "mild"})
    hits = detect_profanity(_words("What", "the", "heck", "was", "that"), lexicon)
    assert len(hits) == 1
    assert hits[0].match_type == "phrase"
    assert hits[0].matched == "what the heck"
    assert hits[0].index == 0
    assert hits[0].end_index == 3


def test_substring_false_positives_are_not_flagged() -> None:
    lexicon = SafetyLexicon(words={"ass": "strong"})
    text = _words("The", "class", "bass", "assistant", "from", "Scunthorpe", "bought", "a", "cassette")
    hits = detect_profanity(text, lexicon)
    assert hits == []


def test_default_allowlist_includes_readme_examples() -> None:
    names = set(load_allowlist())
    for word in ("bass", "class", "scunthorpe", "assistant"):
        assert word in names


def test_allowlist_prevents_censor_decision() -> None:
    lexicon = SafetyLexicon(words={"badword": "strong"}, allowlist={"badword"})
    hits = detect_profanity(_words("a", "badword", "idea"), lexicon)
    assert len(hits) == 1
    assert hits[0].allowlisted is True


def test_inflection_does_not_create_substring_hits() -> None:
    lexicon = SafetyLexicon(words={"ass": "strong", "fuck": "strong"})
    hits = detect_profanity(_words("class", "passing", "assume", "fucking"), lexicon)
    matched = {hit.matched for hit in hits}
    assert "ass" not in matched
    assert "fuck" in matched


def test_default_lexicon_does_not_flag_class_or_bass() -> None:
    hits = detect_profanity(_words("The", "class", "played", "bass"), load_lexicon())
    assert hits == []


def test_load_lexicon_reads_files() -> None:
    lexicon = load_lexicon()
    assert lexicon.words
    assert "bass" in lexicon.allowlist
    assert any(" " in phrase for phrase in lexicon.phrases)
