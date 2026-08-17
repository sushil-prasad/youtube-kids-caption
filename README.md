# youtube-kids-caption

To transcribe a video or audio file to `.srt`, see **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)**. CLI: `make pipeline VIDEO=clip.mp4`. Dashboard (optional): `python -m app.main --serve` at http://127.0.0.1:8000. Evaluation: `make phase6`.

# Build a Child-Focused Automatic Captioning System for YouTube Creators

## 1. Project goal

Build an end-to-end application that takes a children's/family video as input and produces a high-quality, child-appropriate `.srt` subtitle file.

The system should be optimized for children's speech and should use the **1st-place solution from the DrivenData "On Top of Pasketti" Word Track challenge as the initial ASR backend**, specifically the released/fine-tuned Qwen3-ASR-1.7B-based model/checkpoint where available.

The application is intended for creators of children's/family content who need accurate automatic captions.

The system should prioritize:

1. Accuracy on children's voices
2. Robustness to background noise
3. Word-level timestamps
4. Safe/child-appropriate captions
5. Context-aware correction of potentially incorrect ASR words
6. Human review of uncertain captions
7. High-quality `.srt` generation
8. A simple creator-facing dashboard

The application must be designed so the ASR model can be replaced later without rewriting the rest of the system.

---

# 2. Target platform

Development is initially being done on **Apple Silicon Mac**.

Do NOT assume CUDA is available.

The initial system should support:

* Apple Silicon / MPS when supported
* CPU fallback
* Automatic device detection

The architecture should later allow deployment to an NVIDIA GPU server without changing the application logic.

Do not attempt to reproduce the Pasketti competition training pipeline locally. We are using the trained checkpoint/model for inference.

---

# 3. High-level architecture

Implement the following pipeline:

VIDEO
→ AUDIO EXTRACTION
→ AUDIO PREPROCESSING
→ CHILD-FOCUSED ASR
→ WORD-LEVEL TIMESTAMPS
→ CONFIDENCE ESTIMATION
→ PUNCTUATION/CAPITALIZATION
→ SPEAKER DETECTION
→ SOUND EVENT DETECTION
→ PROFANITY DETECTION
→ CONTEXTUAL CORRECTION
→ ALLOWLIST/CUSTOM VOCABULARY
→ SAFETY DECISION
→ CAPTION SEGMENTATION
→ SRT GENERATION
→ DASHBOARD REVIEW
→ FINAL SRT EXPORT

Each stage should be a separate Python module/service rather than one giant script.

---

# 4. Recommended project structure

Use a modular structure similar to:

```text
project/
│
├── app/
│   ├── main.py
│   ├── config.py
│   │
│   ├── pipeline/
│   │   ├── orchestrator.py
│   │   ├── audio.py
│   │   ├── asr.py
│   │   ├── timestamps.py
│   │   ├── confidence.py
│   │   ├── punctuation.py
│   │   ├── diarization.py
│   │   ├── sound_events.py
│   │   ├── profanity.py
│   │   ├── correction.py
│   │   ├── vocabulary.py
│   │   ├── segmentation.py
│   │   └── srt.py
│   │
│   ├── models/
│   │   ├── base.py
│   │   ├── pasketti.py
│   │   └── registry.py
│   │
│   ├── api/
│   │   ├── upload.py
│   │   ├── jobs.py
│   │   ├── captions.py
│   │   └── settings.py
│   │
│   └── database/
│       ├── models.py
│       └── database.py
│
├── dashboard/
│
├── models/
│
├── tests/
│
├── data/
│
├── outputs/
│
├── requirements.txt
├── README.md
└── .env.example
```

The exact framework can be chosen by the coding agent, but favor a simple Python backend such as FastAPI and a lightweight frontend such as React/Next.js if appropriate.

---

# 5. Stage 1 — Video ingestion

Allow the user to upload:

* `.mp4`
* `.mov`
* `.mkv`
* other common video formats supported by FFmpeg

Create a unique job ID for every upload.

Example:

```text
job_20260816_001
```

Store:

* original filename
* duration
* video resolution
* frame rate
* file size
* audio properties
* processing status

Do not modify the original uploaded file.

---

# 6. Stage 2 — Audio extraction

Use FFmpeg.

Extract:

```text
16 kHz
mono
PCM WAV
```

Example conceptual command:

```bash
ffmpeg -i input.mp4 -ac 1 -ar 16000 output.wav
```

Do not actually assume this exact command is sufficient for every file. Validate the output.

Also support optional audio normalization.

Preserve the original audio separately.

The pipeline should produce:

```text
original_video.mp4
audio.wav
```

---

# 7. Stage 3 — Child-focused ASR

Implement an ASR abstraction:

```python
class ASRModel:
    def transcribe(self, audio_path):
        ...
```

The initial implementation should use the **Pasketti 1st-place Word Track model**, based on **Qwen3-ASR-1.7B**.

Do not hard-code the rest of the application to Qwen.

The model backend should be configurable:

```text
ASR_MODEL=pasketti_first
DEVICE=auto
```

Device selection:

```text
Apple Silicon:
    MPS if supported

otherwise:
    CPU

NVIDIA:
    CUDA
```

The system should clearly report which device is being used.

---

# 8. Word-level transcription

The ASR result should be represented internally as structured words.

Example:

```json
{
  "word": "dinosaur",
  "start": 12.42,
  "end": 12.91,
  "confidence": 0.94
}
```

Do NOT immediately convert the ASR result to an `.srt`.

Keep a rich intermediate representation.

Example:

```json
{
  "text": "Look at the purple dinosaur!",
  "words": [
    {
      "word": "Look",
      "start": 10.10,
      "end": 10.42,
      "confidence": 0.97
    }
  ]
}
```

This representation will be used by the profanity filter, correction system, speaker system, and caption segmenter.

---

# 9. Confidence estimation

Every word/segment should have an estimated confidence score between 0 and 1 whenever possible.

Classify confidence:

```text
0.90–1.00 = high
0.70–0.89 = medium
0.00–0.69 = low
```

These thresholds must be configurable.

Do not assume ASR confidence is perfectly calibrated.

Treat it as one signal among several.

Store:

```text
ASR confidence
correction confidence
profanity risk
overall caption confidence
```

---

# 10. Punctuation and capitalization

Convert raw ASR output into readable captions.

Example:

Raw:

```text
hey guys today we're going to build a castle
```

Output:

```text
Hey guys! Today we're going to build a castle.
```

Use a lightweight punctuation/capitalization module.

Do not unnecessarily change words.

Keep the raw ASR output available for debugging.

---

# 11. Child-specific/custom vocabulary

Implement a customizable vocabulary system.

Creators should be able to add:

* character names
* people's names
* locations
* game names
* toy names
* fictional words
* recurring phrases
* brand names
* creator-specific terminology

Example:

```text
Bluey
Bingo
Muffin
Rainbow Castle
Fluffernoodle
Sparkleberry
```

The vocabulary should be available to the correction stage.

If the ASR backend supports decoding prompts or vocabulary boosting, use them.

Otherwise use the vocabulary during post-processing/context correction.

---

# 12. Profanity detection

Implement a dedicated profanity/safety module.

Do NOT simply perform:

```python
if word in profanity_list:
    replace(word, "_")
```

The profanity system must be context-aware.

It should consider:

* exact word
* phrase
* surrounding words
* word confidence
* phonetic similarity
* semantic context
* whether the word could be a legitimate non-profane word
* creator allowlist
* language

Support both:

### Word-level profanity

and

### Phrase-level profanity

The system should avoid false positives caused by substring matching.

For example, do not censor an innocent word simply because it contains characters associated with a profane word.

---

# 13. Profanity allowlist

Create an allowlist.

Words in the allowlist should not be censored unless the creator explicitly changes the setting.

Support creator-specific allowlists.

Example:

```text
bass
class
Scunthorpe
assistant
```

The exact default list should be configurable and should not be hard-coded throughout the application.

---

# 14. Context-aware profanity correction

This is one of the core features of the application.

If ASR produces a potentially inappropriate word, do NOT immediately censor it.

Instead, inspect the surrounding context.

Example:

```text
ASR:
"That was a really [SUSPICIOUS WORD] idea."
```

The correction system should inspect:

* previous 3–5 words
* following 3–5 words
* full sentence when available
* ASR confidence
* phonetic similarity
* custom vocabulary
* language-model likelihood

Attempt to determine whether the ASR output is a mistranscription.

Example conceptual flow:

```text
ASR prediction
       ↓
Profanity detector
       ↓
Potentially unsafe
       ↓
Generate plausible alternatives
       ↓
Compare alternatives using context
       ↓
 ┌──────────────┬──────────────────┬─────────────────┐
 │ confident    │ likely correction│ unresolved      │
 │ safe         │                  │                 │
 ├──────────────┼──────────────────┼─────────────────┤
 │ keep word    │ replace word     │ replace with _  │
 └──────────────┴──────────────────┴─────────────────┘
```

Every correction should be logged internally.

Example:

```json
{
  "original": "bad_word",
  "replacement": "bad_word_candidate",
  "reason": "contextual_correction",
  "confidence": 0.87
}
```

---

# 15. Safety decision policy

For each potentially profane word:

### Case A — clearly legitimate

Keep it.

### Case B — likely ASR error with strong contextual alternative

Replace with the alternative.

### Case C — profanity is probably real but correction is uncertain

Replace with:

```text
_
```

### Case D — uncertain whether the word is profane

Flag it for human review.

Never silently make aggressive corrections when confidence is low.

---

# 16. Safety modes

Implement at least two modes.

## Safe mode

Prioritize child-appropriate captions.

Potentially unsafe words are corrected or censored.

## Literal/review mode

Show the original ASR result and flag potentially unsafe words for creator review.

Do not delete the original transcription.

---

# 17. Speaker detection / diarization

Implement speaker detection as a modular feature.

The first version does not need perfect speaker identity.

At minimum detect speaker changes:

```text
Speaker 1
Speaker 2
Speaker 1
```

Allow the creator to rename speakers:

```text
Speaker 1 → Mom
Speaker 2 → Liam
Speaker 3 → Narrator
```

Speaker changes should influence caption segmentation.

Do not require speaker diarization for the basic pipeline to function.

---

# 18. Non-speech sound detection

Implement an optional sound-event recognition module.

Recognize useful caption events such as:

```text
[laughter]
[giggles]
[crying]
[screaming]
[applause]
[music]
[dog barking]
[doorbell]
[background noise]
```

Use square brackets.

Sound-event detection must be modular so the application can run without it on limited hardware.

Do not hallucinate sound events.

Only insert an event when confidence exceeds a configurable threshold.

---

# 19. Caption segmentation

Convert the word-level representation into readable subtitle segments.

Optimize for:

* natural linguistic boundaries
* sentence boundaries
* pauses
* speaker changes
* maximum characters per line
* maximum lines
* reasonable reading speed
* minimum caption duration
* maximum caption duration

Prefer:

```text
Look at this giant
purple dinosaur!
```

over:

```text
Look at this giant purple
dinosaur!
```

Do not split captions in the middle of a natural phrase unless necessary.

Make segmentation settings configurable.

---

# 20. Reading speed

Calculate approximate characters-per-second or words-per-second.

Flag captions that are too dense.

Example internal status:

```text
Reading speed: 18 CPS
Status: OK
```

If a caption is excessively dense:

1. attempt better segmentation
2. extend timing if possible
3. split the caption
4. flag for review if necessary

---

# 21. Timestamp validation

Before exporting:

Validate:

* timestamps are increasing
* no negative timestamps
* no malformed timestamps
* no overlapping captions unless explicitly supported
* captions fall within video duration
* no empty captions
* minimum duration
* maximum duration
* reasonable gaps

Automatically repair simple timestamp errors.

Flag serious errors.

---

# 22. SRT generation

Generate standard UTF-8 `.srt`.

Example:

```text
1
00:00:10,100 --> 00:00:12,900
Look at the purple dinosaur!

2
00:00:13,100 --> 00:00:15,200
Isn't he huge?
```

The SRT exporter should be independent from the dashboard.

A CLI command should also exist:

```bash
python -m app.pipeline input.mp4 --output output.srt
```

---

# 23. Caption quality report

After processing, generate a quality summary.

Example:

```text
Caption Quality

Overall confidence: 94%

Words: 1,842
Low-confidence words: 37
Potential profanity: 6
Automatically corrected: 4
Censored: 2
Speaker changes: 18
Sound events: 24

Reading-speed warnings: 3
```

This should appear in the dashboard.

---

# 24. Human review dashboard

The dashboard is implemented in `dashboard/` and served by `python -m app.main --serve` (http://127.0.0.1:8000). A page load starts blank; video and captions appear after **Upload file** or **Try sample**.

### Upload

- **Upload file** — pick a `.mp4`, `.mov`, `.mkv`, or audio (`.wav`, `.mp3`, `.m4a`). Preview shows the first frame (paused). Processing starts immediately. A copy is stored in the job folder; the original is not modified.
- **Cancel upload** — stop the in-progress job so another clip can be started.
- **Try sample** — runs the pipeline on `sample.mp4`.
- One video at a time. A leftover busy job from a server restart is marked interrupted so a new upload is not blocked.

### Video player

Play the uploaded clip in a 16:9 frame (vertical videos are pillarboxed). Captions seek the player; timestamps on cues are `MM:SS`.

### Caption timeline and editor

Caption blocks sit on a timeline aligned with the video. Click a cue to seek. Edit text in the transcript list. While a job is running, the caption list shows a skeleton loader.

### Confidence highlighting

The timeline and the left bar on each cue use the same colors:

```text
teal — high confidence
amber — medium confidence
terracotta — low confidence
```

### Safety warnings

Potentially unsafe captions should be clearly flagged.

Example:

```text
⚠ Potentially unsafe word

"That was a really ___ idea."

Suggested correction:
"That was a really bad idea."

[Accept] [Keep censored] [Edit]
```

### Quality report

After captions are ready, the processing card is followed by overall confidence, word counts, low-confidence words, safety stats, and reading-speed warnings. The empty quality card is hidden until those numbers exist.

### Timeline synchronization

Clicking a caption should seek the video to that timestamp.

Clicking a timestamp should select the corresponding caption.

---

# 25. Creator vocabulary interface

Dashboard should provide:

```text
Custom Vocabulary

[+ Add word]

Character names
Brands
Places
Made-up words
Other
```

Allow import/export of vocabulary lists.

---

# 26. Creator safety settings

Provide settings such as:

```text
Safety Mode:
  ● Strict
  ○ Standard
  ○ Review-only

Unknown profanity:
  ● Censor
  ○ Flag
  ○ Keep

Sound effects:
  ● Enable
  ○ Disable
```

Speaker labels are not shown in the dashboard. Use sensible defaults for children's content.

---

# 27. Export

Provide:

```text
Download .SRT
```

Also provide:

```text
Download reviewed .SRT
Download raw ASR .SRT
Download correction report
```

The creator should always be able to recover the original transcription.

---

# 28. Processing architecture

Use a job-based architecture.

Example:

```text
UPLOAD
  ↓
QUEUED
  ↓
EXTRACTING_AUDIO
  ↓
TRANSCRIBING
  ↓
ALIGNING
  ↓
PUNCTUATING
  ↓
DETECTING_SPEAKERS
  ↓
DETECTING_SOUNDS
  ↓
SAFETY_ANALYSIS
  ↓
CORRECTING
  ↓
SEGMENTING
  ↓
GENERATING_SRT
  ↓
READY_FOR_REVIEW
  ↓
REVIEWED
  ↓
EXPORTED
```

The dashboard should display the current processing state.

---

# 29. Logging and reproducibility

Every processing job should save:

```text
job configuration
ASR model version
model checksum if available
device used
processing time
pipeline version
safety configuration
custom vocabulary
corrections
censoring decisions
```

This is important for debugging and research.

Never overwrite raw outputs.

---

# 30. Intermediate artifacts

Store:

```text
audio.wav
raw_transcript.json
word_timestamps.json
confidence.json
speaker_segments.json
sound_events.json
safety_analysis.json
corrected_transcript.json
final_captions.json
final.srt
```

This allows individual stages to be rerun without retranscribing the entire video.

For example:

```bash
--skip-asr
```

should allow the developer to rerun profanity filtering/segmentation without paying the ASR cost again.

---

# 31. CLI

Provide a CLI for development.

Examples:

```bash
python -m app.pipeline input.mp4
```

```bash
python -m app.pipeline input.mp4 \
    --output output.srt \
    --device mps
```

```bash
python -m app.pipeline input.mp4 \
    --safety-mode strict \
    --enable-diarization \
    --enable-sound-events
```

Also provide:

```bash
python -m app.pipeline --help
```

---

# 32. Dashboard should NOT be required to run the pipeline

The backend pipeline must work independently.

A developer should be able to run:

```bash
python -m app.pipeline video.mp4
```

and get:

```text
video.srt
```

without starting the web application.

The dashboard is a frontend for the pipeline, not the pipeline itself.

---

# 33. Mac Silicon requirements

At startup, detect:

```text
Apple Silicon
MPS available
CPU available
```

Print something similar to:

```text
Device: Apple Silicon
Backend: MPS
ASR model: Pasketti 1st Place / Qwen3-ASR-1.7B
```

If a dependency does not support MPS:

```text
WARNING: <component> does not support MPS.
Falling back to CPU.
```

Do not crash merely because CUDA is unavailable.

Avoid dependencies that require NVIDIA-specific functionality for basic inference.

Do not make CUDA-specific assumptions anywhere in the core pipeline.

---

# 34. Performance

For the Mac prototype:

* process one video at a time
* avoid unnecessary model reloads
* cache loaded models
* cache intermediate artifacts
* provide processing-time estimates
* show progress in the dashboard

Do not optimize for batch inference initially.

Correctness is more important than throughput.

---

# 35. Model abstraction

Create a model registry:

```python
MODEL_REGISTRY = {
    "pasketti_first": PaskettiFirstModel,
}
```

Eventually support:

```text
pasketti_first
pasketti_second
future_child_asr
```

The rest of the system must not know which ASR model generated the transcription.

---

# 36. Testing

Create unit tests for:

### Audio

* extraction
* sample rate
* mono conversion

### Timestamps

* valid SRT timestamps
* overlapping captions
* invalid timestamps

### Profanity

* exact matches
* phrases
* false positives
* allowlist
* capitalization
* punctuation

### Correction

* safe words
* obvious ASR errors
* uncertain words
* profanity replacement

### Segmentation

* maximum line length
* sentence boundaries
* speaker changes
* reading speed

### SRT

* valid output
* UTF-8
* correct numbering
* correct timestamp formatting

---

# 37. Evaluation framework

Create an evaluation script so different ASR models can be compared on the same videos.

Example:

```bash
python evaluate.py \
    --dataset evaluation_data \
    --model pasketti_first
```

Report:

```text
WER
CER
word timestamp accuracy
low-confidence rate
profanity false-positive rate
profanity false-negative rate
correction accuracy
censorship rate
caption reading speed
```

This will eventually allow comparison of:

```text
Pasketti 1st
Pasketti 2nd
Pasketti 3rd
future child-focused model
```

Do not assume the competition leaderboard ranking necessarily translates directly to YouTube children's content.

---

# 38. Research-friendly design

The system should make it possible to perform ablation experiments.

At minimum support:

```text
ASR only

ASR
+ punctuation

ASR
+ profanity filtering

ASR
+ context correction

ASR
+ profanity filtering
+ context correction

Full pipeline
```

This will allow measurement of whether each feature actually improves the final captions.

---

# 39. Important safety principle

The system should never claim that its captions are guaranteed to be safe or perfectly accurate.

The dashboard should clearly communicate that automatic captions require creator review.

The purpose of the safety system is to **reduce the probability of inappropriate or erroneous captions**, not to guarantee perfect filtering.

---

# 40. Development priorities

Implement in this order:

## Phase 1 — Core ASR

1. Project setup
2. FFmpeg audio extraction
3. Pasketti first-place model integration
4. Mac MPS/CPU support
5. Raw transcription
6. Word timestamps
7. Basic SRT generation

At the end of Phase 1:

```text
video.mp4 → video.srt
```

must work.

## Phase 2 — Caption quality

8. Confidence
9. Punctuation
10. Capitalization
11. Caption segmentation
12. Reading-speed checks
13. Timestamp validation

## Phase 3 — Child safety

14. Profanity dictionary
15. Allowlist
16. Phrase-level detection
17. Contextual correction
18. Unresolved profanity → `_`
19. Safety confidence

## Phase 4 — Advanced captions

20. Speaker detection
21. Speaker labels
22. Sound-event recognition
23. Custom vocabulary

## Phase 5 — Dashboard

24. Upload
25. Processing progress
26. Video player
27. Caption editor
28. Confidence visualization
29. Safety review
30. Vocabulary management
31. Export

## Phase 6 — Evaluation

32. Evaluation dataset
33. WER/CER
34. Safety metrics
35. Model comparison
36. Ablation experiments

---

# 41. Do NOT over-engineer the first version

The first milestone should be:

```text
MP4
 ↓
FFmpeg
 ↓
WAV
 ↓
Pasketti 1st-place ASR
 ↓
word timestamps
 ↓
basic segmentation
 ↓
SRT
```

Get this working on Apple Silicon before implementing diarization, sound events, contextual correction, or the dashboard.

Then add each feature independently.

The final system should be modular enough that any individual component can be disabled without breaking the rest of the pipeline.

---

# 42. Definition of done

The project is successful when a user can:

1. Open the dashboard.
2. Upload a children's video (or Try sample).
3. Wait for processing.
4. See the video and generated captions synchronized.
5. See low-confidence captions (timeline and cue color bars).
6. See potentially unsafe words flagged.
7. Accept/reject contextual corrections.
8. Edit captions manually.
9. Add custom vocabulary.
10. Review detected sound events if enabled.
11. Export a valid UTF-8 `.srt`.
12. Upload that SRT to YouTube Studio.

The entire system should also work without the dashboard through the CLI.

Prioritize a reliable, testable pipeline over visual polish.