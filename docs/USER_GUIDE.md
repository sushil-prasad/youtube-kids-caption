# Transcribe a video

Turn a children's/family video (or audio clip) into a UTF-8 `.srt` caption file. You can use the CLI alone, or review captions in the dashboard (`python -m app.main --serve`). The dashboard is optional.

The pipeline is complete through evaluation (Phase 6): audio extraction, ASR, timestamps, confidence, punctuation, optional speakers/sound events, vocabulary, safety, segmentation, validated SRT, creator review, export, and metrics.

## Prerequisites

- **Apple Silicon Mac** (MPS) or any machine with CPU. NVIDIA CUDA is not required.
- **Python 3.10–3.13** recommended (`python3 --version`)
- **FFmpeg** (and ffprobe) on your PATH

```bash
brew install ffmpeg
ffmpeg -version
```

Supported inputs include `.mp4`, `.mov`, `.mkv`, and audio such as `.wav`, `.mp3`, and `.m4a` (anything FFmpeg can decode). The original file is never modified.

## One-time setup

From the repository root:

```bash
make setup
make install
```

`make setup` creates `.venv`, `data/`, `outputs/`, `models/`, and a `.env` from `.env.example`.

`make install` installs PyTorch, `qwen-asr`, and other inference dependencies.

The first transcription downloads **Qwen/Qwen3-ASR-1.7B** (and the forced aligner). That can take several GB and several minutes.

Optional: copy a merged Pasketti checkpoint into `models/pasketti_first` or set `ASR_MODEL_PATH` in `.env`. If neither is set, the public Qwen3-ASR-1.7B weights are used.

## Transcribe

```bash
make pipeline VIDEO=/path/to/clip.mp4
```

Or:

```bash
.venv/bin/python -m app.pipeline /path/to/clip.mp4
```

You should see a device banner similar to:

```text
Device: Apple Silicon
Backend: MPS
ASR model: Pasketti 1st Place / Qwen3-ASR-1.7B
```

On success, captions are written next to the video as `clip.srt`.

### Choose the output path

```bash
make pipeline VIDEO=/path/to/clip.mp4 OUTPUT=/path/to/clip.srt
```

```bash
.venv/bin/python -m app.pipeline /path/to/clip.mp4 --output /path/to/clip.srt
```

### Choose device

`DEVICE=auto` in `.env` picks MPS on Apple Silicon, CUDA if present, otherwise CPU.

```bash
.venv/bin/python -m app.pipeline /path/to/clip.mp4 --device mps
.venv/bin/python -m app.pipeline /path/to/clip.mp4 --device cpu
```

## What you get

Each run creates a job directory:

```text
outputs/job_YYYYMMDD_NNN/
  job.json
  audio.wav                 # 16 kHz mono PCM
  raw_transcript.json       # original ASR (never overwritten by punctuation)
  word_timestamps.json
  confidence.json
  punctuated_transcript.json
  vocabulary.json
  speaker_segments.json         # if ENABLE_DIARIZATION=1
  sound_events.json             # if ENABLE_SOUND_EVENTS=1
  annotated_transcript.json     # speakers / [events] / vocab
  safety_analysis.json
  corrected_transcript.json
  correction_log.json
  final_captions.json
  timestamp_validation.json
  final.srt
  reviewed_captions.json        # after dashboard edits
  reviewed.srt
  raw.srt                       # original ASR as SRT (never overwrites raw_transcript.json)
  correction_report.md
```

| File | Purpose |
|---|---|
| `audio.wav` | Extracted audio used for ASR |
| `raw_transcript.json` | Original ASR transcript plus words |
| `word_timestamps.json` | `{word, start, end, confidence}` |
| `confidence.json` | Per-word 0–1 scores and high/medium/low bands |
| `punctuated_transcript.json` | Readable sentences; raw ASR is kept separately |
| `vocabulary.json` | Merged creator vocabulary |
| `speaker_segments.json` | Speaker 1/2/… turns (optional) |
| `sound_events.json` | `[laughter]`-style events above threshold (optional) |
| `annotated_transcript.json` | Transcript after optional speakers, events, and vocab |
| `safety_analysis.json` | Word- and phrase-level safety hits (not guaranteed complete) |
| `corrected_transcript.json` | After keep / replace / `_` / flag decisions |
| `correction_log.json` | Every safety decision, with a review disclaimer |
| `final_captions.json` | Segmented cues with reading-speed flags |
| `timestamp_validation.json` | Repaired or flagged timestamp issues |
| `final.srt` | UTF-8 subtitles for YouTube Studio |
| `reviewed.srt` | Captions after human edits (dashboard or export) |
| `raw.srt` | Original ASR words as SRT; the raw transcript stays recoverable |
| `correction_report.md` | Keep / replace / censor / flag decisions |

Upload `reviewed.srt` (or `final.srt` / the sidecar `clip.srt`) in YouTube Studio as captions. Automatic captions still need a human pass; this tool does not guarantee perfect accuracy or safety.

## Re-run without calling ASR again

Useful after changing segmentation or punctuation, or if transcription already finished:

```bash
make pipeline SKIP_ASR=1 JOB_DIR=outputs/job_20260816_001 OUTPUT=/tmp/clip.srt
```

```bash
.venv/bin/python -m app.pipeline --skip-asr --job-dir outputs/job_20260816_001 --output /tmp/clip.srt
```

## Stage by stage

```bash
make extract-audio VIDEO=/path/to/clip.mp4
make asr VIDEO=/path/to/clip.mp4
make timestamps JOB_DIR=outputs/job_YYYYMMDD_NNN
make confidence JOB_DIR=outputs/job_YYYYMMDD_NNN
make punctuation JOB_DIR=outputs/job_YYYYMMDD_NNN
make diarization JOB_DIR=outputs/job_YYYYMMDD_NNN
make sound-events JOB_DIR=outputs/job_YYYYMMDD_NNN
make vocabulary JOB_DIR=outputs/job_YYYYMMDD_NNN
make profanity JOB_DIR=outputs/job_YYYYMMDD_NNN
make correction JOB_DIR=outputs/job_YYYYMMDD_NNN
make segmentation JOB_DIR=outputs/job_YYYYMMDD_NNN
make srt JOB_DIR=outputs/job_YYYYMMDD_NNN OUTPUT=/tmp/clip.srt
```

```bash
.venv/bin/python -m app.pipeline /path/to/clip.mp4 --stop-after extract-audio
.venv/bin/python -m app.pipeline /path/to/clip.mp4 --stop-after asr
```

## Settings

Edit `.env` (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `ASR_MODEL` | `pasketti_first` | Model registry name |
| `DEVICE` | `auto` | `auto`, `mps`, `cpu`, or `cuda` |
| `LANGUAGE` | `English` | Forced ASR language; blank for auto-detect |
| `ENABLE_AUDIO_NORMALIZATION` | `0` | FFmpeg loudnorm before ASR |
| `ENABLE_FORCED_ALIGNER` | `1` | Word-level timestamps; falls back to even spacing if the aligner fails |
| `ASR_MODEL_PATH` | unset | Local merged checkpoint directory |
| `CONFIDENCE_HIGH` | `0.90` | High-confidence band lower bound |
| `CONFIDENCE_MEDIUM` | `0.70` | Medium-confidence band lower bound |
| `MAX_CHARS_PER_LINE` | `42` | Caption line length |
| `MAX_CPS` | `17.0` | Reading-speed limit (characters per second) |
| `SAFETY_MODE` | `strict` | `strict` / `standard` (safe) or `review-only` (keep original, flag) |
| `UNKNOWN_PROFANITY` | `censor` | Unresolved hits: `censor` (`_`), `flag`, or `keep` |
| `ALLOWLIST` | unset | Extra comma-separated words that must not be censored |
| `ENABLE_DIARIZATION` | `0` | Optional speaker-change detection |
| `ENABLE_SOUND_EVENTS` | `0` | Optional `[sound event]` captions (only above threshold) |
| `SOUND_EVENT_THRESHOLD` | `0.80` | Minimum confidence to insert a sound event |

```bash
.venv/bin/python -m app.pipeline --help
make help
```

## Review in the dashboard

The dashboard is a frontend over the same pipeline. You do not need it to produce an SRT.

```bash
make install   # once, so FastAPI is available
.venv/bin/python -m app.main --serve
```

Then open http://127.0.0.1:8000

- Upload a `.mp4`, `.mov`, `.mkv`, or audio (`.wav`, `.mp3`, `.m4a`). A copy is stored in the job folder; the original file is not modified. Audio is wrapped in a black-frame mp4 so the player can run.
- The Mac prototype processes **one video at a time**.
- When status is `READY FOR REVIEW`, play the video, click captions to seek, edit text, and handle safety flags (Accept / Keep censored / Edit).
- Confidence uses teal (high), amber (medium), and terracotta (low).
- Add vocabulary and change safety settings from the other tabs. Settings apply to the next job.
- Export **reviewed SRT**, **raw ASR SRT**, and a **correction report**. Raw ASR is always recoverable.

`make api` / `make dashboard` / `make export` check that these pieces exist without starting a long-running server. To serve: `make api SERVE=1`.

```bash
.venv/bin/python -m app.api.captions --export --job-dir outputs/job_YYYYMMDD_NNN
```

## Try it with a new clip

The first real transcription downloads the ASR weights (several GB) and can take a few minutes. Later runs reuse the cache.

### Fastest: CLI

Video:

```bash
make pipeline VIDEO=/path/to/your_clip.mp4
```

Audio (`.wav`, `.mp3`, `.m4a`, …) — same command; FFmpeg extracts the track. The original file is not modified:

```bash
make pipeline VIDEO=/path/to/your_clip.wav
```

Or:

```bash
.venv/bin/python -m app.pipeline /path/to/your_clip.wav --output /tmp/your_clip.srt
```

You should see a device banner, then a sidecar `your_clip.srt` next to the input (unless you passed `--output`). Job files land in `outputs/job_YYYYMMDD_NNN/`, including `final.srt`, `raw_transcript.json`, and `final_captions.json`.

Use a **short** clip the first time (10–30 seconds of kids'/family speech with a clear voice).

### Dashboard (review the same pipeline)

```bash
.venv/bin/python -m app.main --serve
```

Open http://127.0.0.1:8000 and upload the file. Audio uploads are wrapped as a still video so you can play along with captions.

If you already have audio and want an mp4 yourself:

```bash
make wrap-audio AUDIO=/path/to/your_clip.wav OUTPUT=/tmp/your_clip.mp4
```

Then upload `/tmp/your_clip.mp4` (or pass it to `make pipeline`).

Checklist once status is **READY FOR REVIEW**:

1. Play the media — captions should track the playhead.
2. Low-confidence cues use terracotta; click a cue to seek.
3. If a safety flag appears, try Accept / Keep censored / Edit.
4. Edit a line in the transcript box.
5. Add a vocabulary term (a character name) on the Vocabulary tab.
6. Export reviewed SRT, raw ASR SRT, and the correction report.

Automatic captions still need a human pass. They are not guaranteed safe or perfectly accurate.

### Metrics without a new clip

```bash
make phase6
```

This scores the JSON fixtures in `evaluation_data/` (WER, CER, timestamps, safety, reading speed, ablation). It does **not** download the ASR model.

```bash
.venv/bin/python evaluate.py --dataset evaluation_data --model pasketti_first
.venv/bin/python evaluate.py --dataset evaluation_data --model pasketti_first --ablation
```

## Evaluate

Reports are written to `outputs/evaluation/metrics.json` (and `ablation.json` with `--ablation`). Do not treat a competition leaderboard rank as YouTube children's-content quality.

## Troubleshooting

**`FFmpeg not found`**  
Install with `brew install ffmpeg` and confirm `ffmpeg` and `ffprobe` are on your PATH.

**Extraction tests skipped / pipeline fails on audio**  
The video must contain an audio track. The original file is left unchanged; check `outputs/job_*/audio.wav`.

**First run is slow or uses a lot of disk**  
The ASR weights download on first use. Later runs reuse the Hugging Face cache and keep the model loaded only for that process.

**`WARNING: … Falling back to CPU`**  
MPS or CUDA was requested but is unavailable. Transcription still runs, more slowly.

**No word timestamps**  
If the forced aligner fails to load, words are spaced evenly across the clip duration. SRT is still produced.

**Python 3.14 errors during `make install`**  
Use Python 3.10–3.13 for the venv, then `make install` again.

**Unit tests only (no model download)**  

```bash
make phase1
make phase2
make phase3
make phase4
make phase5
make phase6
make test
```
