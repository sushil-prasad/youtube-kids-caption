# youtube-kids-caption — see docs/IMPLEMENTATION.md
# Phases 1–6 are implemented. `make help` lists targets.

PYTHON ?= .venv/bin/python
VIDEO ?=
OUTPUT ?=
JOB_DIR ?=
SKIP_ASR ?= 0
SAFETY_MODE ?=
ENABLE_DIARIZATION ?=
ENABLE_SOUND_EVENTS ?=
SERVE ?= 0
AUDIO ?=

.DEFAULT_GOAL := help

.PHONY: help setup install check-device \
	extract-audio asr timestamps srt cli phase1 test-phase1 \
	confidence punctuation segmentation reading-speed validate-timestamps phase2 test-phase2 \
	profanity allowlist correction safety-policy phase3 test-phase3 \
	diarization sound-events vocabulary phase4 test-phase4 \
	api dashboard export phase5 test-phase5 \
	evaluate ablation phase6 test-phase6 \
	pipeline wrap-audio test all

define require_module
	echo "==> $(1)"; \
	echo "    expected artifacts: $(2)"; \
	echo "    see docs/IMPLEMENTATION.md"; \
	if [ ! -f "$(3)" ]; then \
		echo "Not implemented yet: missing $(3)"; \
		exit 1; \
	fi
endef

help: ## List targets (see docs/IMPLEMENTATION.md)
	@echo "youtube-kids-caption"
	@echo "Plan: docs/IMPLEMENTATION.md"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make phase1"
	@echo "  make pipeline VIDEO=path.mp4"
	@echo "  make pipeline VIDEO=path.mp4 SKIP_ASR=1"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

setup: check-device ## Project dirs, FFmpeg check, device print
	@echo "==> Project setup"
	@mkdir -p data outputs models
	@FFMPEG_BIN="$$(command -v ffmpeg 2>/dev/null || true)"; \
	if [ -z "$$FFMPEG_BIN" ]; then \
		for p in /opt/homebrew/bin/ffmpeg /usr/local/bin/ffmpeg; do \
			if [ -x "$$p" ]; then FFMPEG_BIN="$$p"; break; fi; \
		done; \
	fi; \
	if [ -z "$$FFMPEG_BIN" ]; then \
		echo "    warning: FFmpeg not found. Install it before extract-audio."; \
	else \
		echo "    FFmpeg: $$($$FFMPEG_BIN -version | head -n 1)"; \
	fi
	@echo "    created: data/ outputs/ models/"
	@if [ ! -d .venv ]; then \
		python3 -m venv .venv; \
		echo "    created: .venv"; \
	fi
	@.venv/bin/pip install -q pytest python-dotenv
	@if [ ! -f .env ] && [ -f .env.example ]; then \
		cp .env.example .env; \
		echo "    created: .env from .env.example"; \
	fi
	@echo "    ASR deps: make install"
	@echo "    run captions: make pipeline VIDEO=path.mp4"

install: ## Create venv and pip install -r requirements.txt
	@mkdir -p data outputs models
	@if [ ! -d .venv ]; then python3 -m venv .venv; fi
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt

check-device: ## Print Apple Silicon / MPS / CPU (no CUDA assumption)
	@echo "==> Device detection"
	@echo "    OS: $$(uname -s)  arch: $$(uname -m)"
	@if [ "$$(uname -s)" = "Darwin" ] && [ "$$(uname -m)" = "arm64" ]; then \
		echo "    Device: Apple Silicon"; \
		echo "    Backend: MPS if supported, else CPU"; \
	else \
		echo "    Device: $$(uname -s)/$$(uname -m)"; \
		echo "    Backend: CPU (CUDA later if available; not required)"; \
	fi
	@echo "    ASR model: Pasketti 1st Place / Qwen3-ASR-1.7B"
	@echo "    Do not crash if CUDA is unavailable."

# ---------------------------------------------------------------------------
# Phase 1 — Core ASR
# ---------------------------------------------------------------------------

extract-audio: ## FFmpeg: 16 kHz mono PCM WAV
	@$(call require_module,FFmpeg audio extraction,audio.wav,app/pipeline/audio.py)
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make extract-audio VIDEO=path.mp4"; exit 1; fi
	$(PYTHON) -m app.pipeline "$(VIDEO)" --stop-after extract-audio $(if $(JOB_DIR),--job-dir "$(JOB_DIR)",)

asr: ## Pasketti 1st-place / Qwen3-ASR-1.7B
	@$(call require_module,Child-focused ASR,raw_transcript.json,app/models/pasketti.py)
	@if [ -n "$(JOB_DIR)" ]; then \
		$(PYTHON) -m app.pipeline.asr --job-dir "$(JOB_DIR)"; \
	else \
		if [ -z "$(VIDEO)" ]; then echo "Usage: make asr VIDEO=path.mp4"; exit 1; fi; \
		$(PYTHON) -m app.pipeline "$(VIDEO)" --stop-after asr; \
	fi

timestamps: ## Word-level {word, start, end, confidence}
	@$(call require_module,Word-level timestamps,word_timestamps.json,app/pipeline/timestamps.py)
	@if [ -z "$(JOB_DIR)" ]; then echo "Usage: make timestamps JOB_DIR=outputs/job_..."; exit 1; fi
	$(PYTHON) -m app.pipeline.timestamps --job-dir "$(JOB_DIR)"

srt: ## Basic UTF-8 SRT from words
	@$(call require_module,Basic SRT generation,final.srt,app/pipeline/srt.py)
	@if [ -z "$(JOB_DIR)" ]; then echo "Usage: make srt JOB_DIR=outputs/job_..."; exit 1; fi
	$(PYTHON) -m app.pipeline.srt --job-dir "$(JOB_DIR)" $(if $(OUTPUT),--output "$(OUTPUT)",)

cli: ## python -m app.pipeline (dashboard not required)
	@$(call require_module,CLI pipeline,final.srt,app/pipeline/orchestrator.py)
	$(PYTHON) -m app.pipeline --help

phase1: setup test-phase1 cli ## Core ASR: mp4 → srt (tests + CLI; run pipeline with VIDEO=)

test-phase1: setup ## Unit tests: audio extraction + SRT
	@$(call require_module,Phase 1 tests,pytest,tests/test_audio.py)
	$(PYTHON) -m pytest tests/test_audio.py tests/test_srt.py tests/test_timestamps.py tests/test_config.py -q

# ---------------------------------------------------------------------------
# Phase 2 — Caption quality
# ---------------------------------------------------------------------------

confidence: ## Per-word confidence + high/medium/low bands
	@$(call require_module,Confidence estimation,confidence.json,app/pipeline/confidence.py)
	@if [ -n "$(JOB_DIR)" ]; then \
		$(PYTHON) -m app.pipeline.confidence --job-dir "$(JOB_DIR)"; \
	else \
		$(PYTHON) -c "from app.pipeline.confidence import classify_confidence"; \
		echo "    confidence module ready (JOB_DIR=outputs/job_... to run)"; \
	fi

punctuation: ## Punctuation and capitalization (keep raw ASR)
	@$(call require_module,Punctuation and capitalization,punctuated transcript,app/pipeline/punctuation.py)
	@if [ -n "$(JOB_DIR)" ]; then \
		$(PYTHON) -m app.pipeline.punctuation --job-dir "$(JOB_DIR)"; \
	else \
		$(PYTHON) -c "from app.pipeline.punctuation import punctuate_words"; \
		echo "    punctuation module ready (JOB_DIR=outputs/job_... to run)"; \
	fi

segmentation: ## Caption blocks: lines, pauses, speaker changes
	@$(call require_module,Caption segmentation,final_captions.json,app/pipeline/segmentation.py)
	@if [ -n "$(JOB_DIR)" ]; then \
		$(PYTHON) -m app.pipeline.segmentation --job-dir "$(JOB_DIR)"; \
	else \
		$(PYTHON) -c "from app.pipeline.segmentation import build_captions"; \
		echo "    segmentation module ready (JOB_DIR=outputs/job_... to run)"; \
	fi

reading-speed: ## CPS/WPS checks; split or flag dense cues
	@$(call require_module,Reading-speed checks,reading-speed flags,app/pipeline/segmentation.py)
	@if [ -n "$(JOB_DIR)" ]; then \
		$(PYTHON) -m app.pipeline.segmentation --job-dir "$(JOB_DIR)" --check-reading-speed; \
	else \
		$(PYTHON) -c "from app.pipeline.segmentation import apply_reading_speed"; \
		echo "    reading-speed checks ready (JOB_DIR=outputs/job_... to run)"; \
	fi

validate-timestamps: ## Repair/flag invalid or overlapping cues
	@$(call require_module,Timestamp validation,validated SRT,app/pipeline/srt.py)
	@if [ -n "$(JOB_DIR)" ]; then \
		$(PYTHON) -m app.pipeline.srt --job-dir "$(JOB_DIR)" --validate; \
	else \
		$(PYTHON) -c "from app.pipeline.srt import validate_captions"; \
		echo "    timestamp validation ready (JOB_DIR=outputs/job_... to run)"; \
	fi

phase2: confidence punctuation segmentation reading-speed validate-timestamps test-phase2 ## Caption quality

test-phase2: ## Unit tests: timestamps + segmentation
	@$(call require_module,Phase 2 tests,pytest,tests/test_segmentation.py)
	$(PYTHON) -m pytest tests/test_timestamps.py tests/test_segmentation.py tests/test_confidence.py tests/test_punctuation.py tests/test_srt.py -q

# ---------------------------------------------------------------------------
# Phase 3 — Child safety
# ---------------------------------------------------------------------------

profanity: ## Context-aware word- and phrase-level detection
	@$(call require_module,Profanity detection,safety_analysis.json,app/pipeline/profanity.py)
	@if [ -n "$(JOB_DIR)" ]; then \
		$(PYTHON) -m app.pipeline.profanity --job-dir "$(JOB_DIR)"; \
	else \
		$(PYTHON) -c "from app.pipeline.profanity import detect_profanity"; \
		echo "    profanity module ready (JOB_DIR=outputs/job_... to run)"; \
	fi

allowlist: ## Creator allowlist (not hard-coded)
	@$(call require_module,Profanity allowlist,allowlist applied,app/pipeline/profanity.py)
	@if [ -n "$(JOB_DIR)" ]; then \
		$(PYTHON) -m app.pipeline.profanity --job-dir "$(JOB_DIR)" --allowlist; \
	else \
		$(PYTHON) -m app.pipeline.profanity --allowlist; \
	fi

correction: ## Contextual correction log (keep / replace / flag)
	@$(call require_module,Context-aware correction,corrected_transcript.json,app/pipeline/correction.py)
	@if [ -n "$(JOB_DIR)" ]; then \
		$(PYTHON) -m app.pipeline.correction --job-dir "$(JOB_DIR)"; \
	else \
		$(PYTHON) -c "from app.pipeline.correction import correct_words"; \
		echo "    correction module ready (JOB_DIR=outputs/job_... to run)"; \
	fi

safety-policy: ## Cases A–D; safe vs review mode
	@$(call require_module,Safety decision policy,safety decisions,app/config.py)
	@if [ -n "$(JOB_DIR)" ]; then \
		$(PYTHON) -m app.pipeline.correction --job-dir "$(JOB_DIR)" --apply-policy $(if $(SAFETY_MODE),--safety-mode $(SAFETY_MODE),); \
	else \
		$(PYTHON) -c "from app.pipeline.correction import decide_safety"; \
		echo "    safety policy ready (JOB_DIR=outputs/job_... to run)"; \
	fi

phase3: profanity allowlist correction safety-policy test-phase3 ## Child safety

test-phase3: ## Unit tests: profanity + correction
	@$(call require_module,Phase 3 tests,pytest,tests/test_profanity.py)
	$(PYTHON) -m pytest tests/test_profanity.py tests/test_correction.py -q

# ---------------------------------------------------------------------------
# Phase 4 — Advanced captions (optional / skippable)
# ---------------------------------------------------------------------------

diarization: ## Speaker changes (not required for basic pipeline)
	@$(call require_module,Speaker detection,speaker_segments.json,app/pipeline/diarization.py)
	@if [ -n "$(JOB_DIR)" ]; then \
		$(PYTHON) -m app.pipeline.diarization --job-dir "$(JOB_DIR)"; \
	else \
		$(PYTHON) -c "from app.pipeline.diarization import diarize_words"; \
		echo "    diarization module ready (JOB_DIR=outputs/job_... to run)"; \
	fi

sound-events: ## [laughter] [music] etc. above threshold only
	@$(call require_module,Sound-event detection,sound_events.json,app/pipeline/sound_events.py)
	@if [ -n "$(JOB_DIR)" ]; then \
		$(PYTHON) -m app.pipeline.sound_events --job-dir "$(JOB_DIR)"; \
	else \
		$(PYTHON) -c "from app.pipeline.sound_events import format_sound_event"; \
		echo "    sound-events module ready (JOB_DIR=outputs/job_... to run)"; \
	fi

vocabulary: ## Custom names, toys, phrases
	@$(call require_module,Custom vocabulary,vocabulary lists,app/pipeline/vocabulary.py)
	@if [ -n "$(JOB_DIR)" ]; then \
		$(PYTHON) -m app.pipeline.vocabulary --job-dir "$(JOB_DIR)"; \
	else \
		$(PYTHON) -m app.pipeline.vocabulary; \
	fi

phase4: diarization sound-events vocabulary test-phase4 ## Advanced captions (optional)

test-phase4: ## Unit tests: speakers / sound events / vocab
	@$(call require_module,Phase 4 tests,pytest,tests/test_diarization.py)
	$(PYTHON) -m pytest tests/test_diarization.py tests/test_sound_events.py tests/test_vocabulary.py -q

# ---------------------------------------------------------------------------
# Phase 5 — Dashboard (not required to run the pipeline)
# ---------------------------------------------------------------------------

api: ## FastAPI upload / jobs / captions / settings
	@$(call require_module,HTTP API,job status machine,app/api/upload.py)
	@if [ "$(SERVE)" = "1" ]; then \
		$(PYTHON) -m app.main --serve; \
	else \
		$(PYTHON) -m app.main; \
	fi

dashboard: ## Creator review UI
	@$(call require_module,Creator dashboard,review UI,dashboard/package.json)
	@echo "    Dashboard: $(PYTHON) -m app.main --serve"
	@echo "    Then open http://127.0.0.1:8000"
	@echo "    CLI still works without it: $(PYTHON) -m app.pipeline VIDEO.mp4"

export: ## Reviewed SRT, raw ASR SRT, correction report
	@$(call require_module,Export,reviewed.srt raw.srt report,app/api/captions.py)
	@if [ -n "$(JOB_DIR)" ]; then \
		$(PYTHON) -m app.api.captions --export --job-dir "$(JOB_DIR)"; \
	else \
		$(PYTHON) -m app.api.captions --help; \
	fi

test-phase5: ## Unit tests: API, review, export
	@$(call require_module,Phase 5 tests,pytest,tests/test_api.py)
	$(PYTHON) -m pytest tests/test_api.py -q

phase5: api dashboard export test-phase5 ## Dashboard + export

# ---------------------------------------------------------------------------
# Phase 6 — Evaluation
# ---------------------------------------------------------------------------

evaluate: ## WER/CER, timestamps, safety metrics
	@$(call require_module,Evaluation,metric report,evaluate.py)
	$(PYTHON) evaluate.py --dataset evaluation_data --model pasketti_first $(if $(OUTPUT),--output "$(OUTPUT)",)

ablation: ## ASR only through full pipeline
	@$(call require_module,Ablation experiments,per-config metrics,evaluate.py)
	$(PYTHON) evaluate.py --dataset evaluation_data --model pasketti_first --ablation $(if $(OUTPUT),--output "$(OUTPUT)",)

test-phase6: ## Unit tests: WER/CER, safety metrics, ablation
	@$(call require_module,Phase 6 tests,pytest,tests/test_evaluate.py)
	$(PYTHON) -m pytest tests/test_evaluate.py -q

phase6: evaluate ablation test-phase6 ## Evaluation + ablation

wrap-audio: ## Wrap an audio clip in a black-frame mp4 for dashboard upload
	@if [ -z "$(AUDIO)" ]; then echo "Usage: make wrap-audio AUDIO=clip.wav OUTPUT=clip.mp4"; exit 1; fi
	$(PYTHON) -m app.pipeline.audio --wrap-audio "$(AUDIO)" $(if $(OUTPUT),--output "$(OUTPUT)",)

# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------

pipeline: ## Full CLI: make pipeline VIDEO=path.mp4
	@$(call require_module,Full pipeline,final.srt,app/pipeline/orchestrator.py)
	@if [ -z "$(VIDEO)" ] && [ "$(SKIP_ASR)" != "1" ]; then \
		echo "Usage: make pipeline VIDEO=path.mp4 [OUTPUT=out.srt] [SKIP_ASR=1 JOB_DIR=outputs/job_...]"; \
		exit 1; \
	fi
	$(PYTHON) -m app.pipeline $(if $(VIDEO),"$(VIDEO)",) \
		$(if $(OUTPUT),--output "$(OUTPUT)",) \
		$(if $(JOB_DIR),--job-dir "$(JOB_DIR)",) \
		$(if $(filter 1,$(SKIP_ASR)),--skip-asr,) \
		$(if $(SAFETY_MODE),--safety-mode $(SAFETY_MODE),) \
		$(if $(filter 1,$(ENABLE_DIARIZATION)),--enable-diarization,) \
		$(if $(filter 1,$(ENABLE_SOUND_EVENTS)),--enable-sound-events,)

test: test-phase1 test-phase2 test-phase3 test-phase4 test-phase5 test-phase6 ## All phase tests

all: phase1 phase2 phase3 phase4 phase5 phase6 ## All phases (fails until implemented)
