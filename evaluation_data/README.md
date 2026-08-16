# Evaluation fixtures

JSON clips used by `python evaluate.py --dataset evaluation_data`.

Each clip has a **reference** (intended captions) and an ASR **hypothesis** (already tokenized). Fixture evaluation never downloads or runs the ASR model.

Do not treat these scores as a competition leaderboard. They exist to compare pipeline stages (ablation) and to keep `make phase6` fast and deterministic.

To score a real transcription, copy `word_timestamps.json` words into a clip file as `hypothesis.words` and put the human transcript in `reference.text`.
