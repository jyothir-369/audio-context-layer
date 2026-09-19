# Phase 5 — Technical Report (Plan Section 8, Sections 1-8)

## 1. Problem Formulation
Audio Question Answering over synthetic multi-event scenes. Grounded in structured event timelines (Plan Section 2). Supported types: perceptual, counting, temporal, causal, plus bonus negation and comparative.

## 2. Research Study
References adopted from the plan as **literature context**: CLAP (zero-shot event matching), PANNs/AudioSet tagging (Plan 3.1), Qwen2.5-Instruct / Phi-3.5-mini for an LLM-grounded answerer (Plan 3.3), Pengi / LLaVA-style frozen-encoder + adapter pattern for Track B (Plan 4.1).

**What was actually built:** a spectral matched-filter tagger over the synthesis vocabulary, a deterministic scene-graph text context, and a **rule-based** answerer. CLAP/PANNs weights are **not** loaded. The LLM-grounded answerer is **not** implemented. Track B is a **skeleton + real projector** (model/train/infer implemented; smoke test passed; full dataset training NOT executed � documented). PANNs Cnn14 wire verified (weight at C:/Users/raghava/panns_data/Cnn14_mAP=0.431.pth; panns_inference loads; tagger tries PANNs first, falls back to spectral).

**PoC simplifications:** class-specific synthetic tones instead of downloaded ESC-50/UrbanSound8K clips; structured intermediate representation instead of end-to-end audio→LLM; no gradient training on Track A (Plan 3.4).

## 3. Dataset Description
- Synthetic scenes: 300 (Plan 2.3 target: 300–500)
- Audio: valid 16 kHz mono PCM_16 WAVs, 25.0 s, 800044 bytes/file — **not** placeholder markers
- Mix method: `audio_signatures.synthesize_event` tones + low sine ambience (descoped from mixing real ESC-50/UrbanSound8K clips)
- QA pairs: 1404 train / 300 val / 306 test
- Scene-level split: 210 / 45 / 45, zero leakage
- Scenarios: street, kitchen, park, office, construction_site — **all five appear in every split**
- Event vocabulary is shared across splits; Plan 2.6 unseen-class holdout was **not** implemented
- Causal questions are fewer than counting (test: 36 vs 45) because they are emitted only when a CAUSAL_TABLE pair co-occurs

## 4. Method
Track A structured pipeline: `event_tagger` (spectral matched-filter, 1s window / 0.5s hop, no gradient training) → `scene_graph_builder` (deterministic text; scene type inferred from predicted labels) → `qa_answerer` (rule-based intent classification + timeline reasoning).

**Event Tagging:** Spectral matched-filter against EVENT_ACOUSTICS (19 classes including `door_slam`). Detections filtered by duration (≥0.5s) and **runtime confidence threshold = 0.15**. A validation label-count sweep (`results/best_threshold.json`) selected 0.10; that snapshot is **not** the runtime filter.

**Scene Graph:** Converts the predicted timeline to a chronological event list. Scene type is inferred from detected labels (kitchen/office/park/street/construction_site/unknown). Ground-truth `scenario` from annotations is **not** copied into the context.

**QA Answering:** Intent-based routing: counting, temporal, negation, comparative, causal (CAUSAL_TABLE matching), perceptual (first event by start time for "what sound"; heuristic label-set mapping for environment questions).

**Threshold Tuning:** Optional validation sweep per Plan 3.4 exists as a historical snapshot. Runtime tagging uses 0.15.

## 5. Experimental Setup
Python 3.13 (verified); numpy, pyyaml, soundfile; synthetic dataset; no GPU required; no gradient-based parameter training. Reproduce with `python src/eval/run_eval.py`.

## 6. Results

### Threshold Tuning (Validation Set, label-count matching — snapshot only)
Swept 7 thresholds [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40] using **label-count** matching (not temporal IoU). Best F1=0.590 at threshold=0.10. **This snapshot is not used at runtime.** Runtime filter = 0.15.

### Test Set Event Detection (Temporal IoU >= 0.30, unique scenes)
Measured by `src/eval/run_eval.py` on the 45 unique test scenes (tag_audio cached; not repeated per QA pair). QA eval still covers all 306 test pairs (failed/skipped = 0).

| Metric | Value |
|---|---|
| Evaluation unit | unique_scene |
| Scenes evaluated | 45 |
| TP | 105 |
| FP | 249 |
| FN | 54 |
| Precision | 0.2966 |
| Recall | 0.6604 |
| F1 | 0.4094 |
| IoU threshold | 0.30 |
| Runtime confidence threshold | 0.15 |

Do not use the older label-count snapshot (Precision 43.5%, Recall 96.9%, F1 60.0%; 154 TP, 200 FP, 5 FN) as the Track A event-detection result.

### Question Answering Results (Rule-Based, Test Set)
| Question Type | Count | Correct | Accuracy | Notes | Most-Common Baseline |
|---|---|---|---|---|---|
| Perceptual | 90 | 41 | 45.56% |  | 13.3% |
| Counting | 45 | 28 | 62.22% | MAE=0.5778 | 62.2% |
| Temporal | 45 | 19 | 42.22% |  | 17.8% |
| Negation | 45 | 45 | 100.0% | FPR=0.0 | 84.4% |
| Comparative | 45 | 22 | 48.89% |  | 55.6% |
| Causal | 36 | 20 | 55.56% | avg_overlap=0.6391; exact=20 | 19.4% |
| **Overall** | **306** | **175** | **57.19%** | failed/skipped=0 | — |

QA numbers re-verified by `python src/eval/run_eval.py` after the scene-level IoU / tag-cache / scene-type-inference changes: overall still 175/306 = 0.5719. Scene-type inference in the graph builder did not change QA answers because the answerer uses the predicted timeline for environment questions, not the context scene-type string.

All question types exceed most-common-answer baseline except comparative (48.89% vs 55.6% baseline). Negation is perfect (100%, FPR=0.0) due to explicit absence-checking.

## 7. Loss Curves
Not applicable — Track A has no gradient-based training (Plan 3.4). Track B was not trained, so `results/loss_curves.png` was not produced.

## 8. Observations and Limitations

**Strengths:**
- Negation accuracy 100% (FPR=0.0): explicit absence-checking prevents hallucination.
- Causal reasoning 55.56% (avg word-overlap 0.6391; exact=20/36): above 19.4% baseline with a 29-pair CAUSAL_TABLE.

**Limitations:**
- Spectral matched-filter over-detects on synthetic tones (high FP expected on IoU evaluation).
- Perceptual accuracy 45.6%: "what sound" uses the first detected event; the tagger often emits an early spurious footstep.
- Environment inference uses heuristic label-set mapping; generic labels (footstep) dominate.
- Temporal accuracy 42.2%: depends on tagger alignment.
- Comparative below baseline (48.9% vs 55.6%): counting errors propagate.
- No unseen-class/scenario holdout.

**Descoped:**
- LLM-grounded sub-approach (Plan 3.3).
- Pretrained CLAP/PANNs/YAMNet tagger (Plan 3.1).
- Real ESC-50/UrbanSound8K mix sources (Plan 2.2).
- Track B training (Plan 4; optional/bonus).

**Production Improvements:**
- Replace spectral matched-filter with CLAP/PANNs on real event clips.
- Expand CAUSAL_TABLE; train a scene classifier; reduce spurious early detections.

### Known Issues Found and Fixed
- Placeholder audio files (11-byte markers) replaced with valid 16 kHz mono PCM_16 WAVs (300 files).
- Hardcoded event-tagger timeline (`_synthetic_prediction`) is fallback-only when audio is unreadable.
- Event IoU evaluation changed from per-QA-pair to **once per unique scene**, with `tag_audio` cached.
- Runtime threshold documented as **0.15** (not the unused 0.10 label-count snapshot).
- Scene type inferred from predicted labels during eval (GT scenario not leaked).
- False unseen-class holdout claim removed from error-analysis generator.
