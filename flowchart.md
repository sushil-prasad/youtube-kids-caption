                    ┌─────────────────────┐
                    │     INPUT VIDEO      │
                    │       .mp4/.mov      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   FFmpeg extraction  │
                    │  16 kHz mono WAV     │
                    │  + optional video    │
                    │      metadata        │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌────────────────────────────┐
                 │   CHILD-FOCUSED ASR         │
                 │                            │
                 │ Pasketti 1st-place model   │
                 │ Qwen3-ASR 1.7B             │
                 └─────────────┬──────────────┘
                               │
                               ▼
                 ┌────────────────────────────┐
                 │ WORD-LEVEL TRANSCRIPTION   │
                 │                            │
                 │ words                     │
                 │ timestamps                │
                 │ confidence                 │
                 └─────────────┬──────────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
       punctuation       speaker detection   sound events
       capitalization       / diarization      [laughter]
             │                 │                [music]
             └─────────────────┼────────────────┘
                               ▼
                 ┌────────────────────────────┐
                 │   SAFETY / CORRECTION     │
                 │                            │
                 │ profanity detection       │
                 │ context correction         │
                 │ allowlist                  │
                 │ custom vocabulary          │
                 │ confidence scoring         │
                 └─────────────┬──────────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
             confidently safe       unresolved/unsafe
                    │                      │
                    │                      ▼
                    │                    "_"
                    │
                    └──────────┬───────────┘
                               ▼
                 ┌────────────────────────────┐
                 │   CAPTION SEGMENTATION     │
                 │                            │
                 │ line length                │
                 │ reading speed              │
                 │ pauses                     │
                 │ punctuation                │
                 │ speaker changes             │
                 └─────────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       .SRT          │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌────────────────────────────┐
                 │        DASHBOARD           │
                 │                            │
                 │ video/audio preview        │
                 │ transcript editor          │
                 │ confidence warnings        │
                 │ profanity warnings         │
                 │ timestamps                 │
                 │ speaker labels             │
                 │ custom dictionary          │
                 │ export .srt                │
                 └────────────────────────────┘