# Phase 4 — Track A Metrics (Plan 5.1, 5.2, 5.3)

## Method Note
- Tagger: spectral matched-filter (deliberate synthetic-PoC deviation from plan-required CLAP/PANNs)
- Runtime confidence threshold: 0.15 (not the val-sweep 0.10 label-count snapshot)
- Rule-based: keyword/intent classification + timeline reasoning
- LLM-grounded: not_implemented (Qwen2.5-0.5B-Instruct when available)
- Test samples processed: 306; failed/skipped: 0
- Event detection unit: unique scene (45 scenes); tag_audio cached per scene

## Aggregate QA Metrics (Rule-Based)

| Type | Count | Correct | Accuracy | Notes |
|---|---|---|---|---|
| perceptual | 90 | 41 | 0.4556 |  |
| counting | 45 | 28 | 0.6222 | MAE=0.5778 |
| temporal | 45 | 19 | 0.4222 |  |
| causal | 36 | 20 | 0.5556 | avg_overlap=0.6391; exact=20 |
| negation | 45 | 45 | 1.0 | FPR=0.0 |
| comparative | 45 | 22 | 0.4889 |  |
| overall | 306 | 175 | 0.5719 |  |

**Overall:** 175/306 = 0.5719

## Event Detection Metrics (Temporal IoU >= 0.30, unique scenes)

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
| IoU threshold | 0.3 |

## Per-Type Counts

### Rule-Based
- perceptual: 90 predictions
- counting: 45 predictions
- temporal: 45 predictions
- negation: 45 predictions
- comparative: 45 predictions
- causal: 36 predictions
