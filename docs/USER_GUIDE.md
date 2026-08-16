# Transcribe a video

Turn a children's/family video into a UTF-8 `.srt` caption file. The dashboard is not required.

Phase 1 only: audio extraction, child-focused ASR, word timestamps, and basic SRT. Safety review, speaker labels, and the editor come in later phases.

## Prerequisites

- **Apple Silicon Mac** (MPS) or any machine with CPU. NVIDIA CUDA is not required.
- **Python 3.10–3.13** recommended (`python3 --version`)
- **FFmpeg** (and ffprobe) on your PATH

```bash
brew install ffmpeg
ffmpeg -version
```

Supported inputs include `.mp4`, `.mov`, `.mkv`, and other formats FFmpeg can decode. The original file is never modified.

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
  raw_transcript.json
  word_timestamps.json
  final.srt
```

| File | Purpose |
|---|---|
| `audio.wav` | Extracted audio used for ASR |
| `raw_transcript.json` | Full transcript plus words |
| `word_timestamps.json` | `{word, start, end, confidence}` |
| `final.srt` | UTF-8 subtitles for YouTube Studio |

Upload `final.srt` (or the sidecar `clip.srt`) in YouTube Studio as captions. Automatic captions still need a human pass; this tool does not guarantee perfect accuracy.

## Re-run without calling ASR again

Useful after changing SRT grouping, or if transcription already finished:

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

```bash
.venv/bin/python -m app.pipeline --help
make help
```

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
```
