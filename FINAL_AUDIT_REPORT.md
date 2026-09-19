# FINAL REVERSE-ENGINEERING AUDIT — AUDIO CONTEXT LAYER

WORKING DIR: C:\Users\raghava\OneDrive\Desktop\audio-context-layer
HEAD at last audit: 4f8644c — subsequent SHOULD-FIX pass updates threshold docs, scene-level IoU, tag cache, scene-type inference, and documentation.

## A. EXECUTIVE STATUS

Track A: COMPLETE with documented deviations (spectral matched-filter; no LLM-grounded answerer; synthetic tones).
Track B: NOT IMPLEMENTED (skeleton only; explicitly documented; no false training claims).

## B. ACTUAL DATASET STATISTICS (verified from files)

- Source dataset: Synthetic tones (ESC-50/UrbanSound8K not downloaded — documented descope)
- Synthetic scenes: 300 annotations + 300 valid 16 kHz WAVs (25.0s, 800044 bytes)
- Vocabulary: 19 classes including door_slam
- Scenarios: street, kitchen, park, office, construction_site
- QA pairs: 2010 (train 1404 + val 300 + test 306)
- Train scenes: 210 | Val: 45 | Test: 45 (zero leakage)
- Holdout: NOT implemented (same scenarios and labels in all splits)
- Test question counts: perceptual 90, counting 45, temporal 45, negation 45, comparative 45, causal 36

## C. TRACK A PATH

AUDIO → spectral matched-filter `tag_audio` (1s / 0.5s hop, runtime confidence >= 0.15) → `build_context` (scene type inferred from predicted labels) → rule-based `answer` → prediction.

No annotation timeline is injected into the answerer. Ground-truth scenario is not copied into context.

## D. EVALUATION

- QA: 306/306 pairs, failed_or_skipped=0
- Event detection: **unique scene** unit (45 test scenes), temporal IoU >= 0.30, cached tag_audio
- Runtime threshold: 0.15
- Do not quote results/best_threshold.json (P=43.5%, R=96.9%, F1=60.0%, 154/200/5) as the IoU result — that file is a label-count validation snapshot at 0.10

Latest verified run (`python src/eval/run_eval.py`): QA 175/306 = 0.5719, failed_or_skipped=0. Event IoU (45 unique scenes): TP=105 FP=249 FN=54 P=0.2966 R=0.6604 F1=0.4094.

## E. REMAINING LIMITATIONS (honest)

- No ESC-50/UrbanSound8K download
- No CLAP/PANNs
- No LLM-grounded answerer
- Track B skeleton only
- No unseen-class holdout
- Causal metric is word-overlap, not BERTScore
