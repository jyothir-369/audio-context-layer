# Audio Context Layer (Audio Question Answering) — Track A PoC

This repository implements a structured/grounded audio-QA pipeline (Plan Section 3), with synthetic dataset construction (Section 2), evaluation (Section 5), error analysis (Section 6), and documentation (Section 8).

## Architecture

- Phase 1: dataset synthesis (class-specific synthetic tones mixed into 25s scenes + ground-truth timelines). ESC-50 / UrbanSound8K downloads are **descoped**; `source_prep.py` fails closed if they are missing.
- Phase 2: QA generation + scene-level split + documentation
- Phase 3: Track A core — **spectral matched-filter** tagger (deliberate PoC deviation from plan-required CLAP/PANNs) + scene graph + **rule-based** QA
- Phase 4: evaluation harness (rule-based metrics + per-unique-scene event IoU). LLM-grounded sub-approach is **descoped** (`not_implemented`)
- Phase 5: error analysis + technical report
- Phase 6: Track B: skeleton + real projector (model/train/infer); smoke test + real loss curve pass; full dataset train NOT executed (honest)

No gradient-based training is performed for Track A (Plan Section 3.4).

## Dataset (verified)

- 300 synthetic scenes (16 kHz mono PCM_16, 25.0s, 800044 bytes/file)
- Scene-level split: 210 train / 45 val / 45 test (zero scene leakage)
- QA pairs: 1404 train / 300 val / 306 test
- All five scenarios appear in every split; **no unseen-class holdout**

## Reproduce

```
pip install -r requirements.txt

# Phase 1 dataset synthesis (synthetic tones; does not require ESC-50/UrbanSound8K)
python src/data_synthesis/scene_composer.py

# Phase 2 QA
python src/qa_generation/generate_qa.py

# Phase 3 Track A core
python -c "
import sys; sys.path.insert(0,'src/track_a_structured')
from event_tagger import tag_audio; from scene_graph_builder import build_context; from qa_answerer import answer
res = tag_audio('data/synthesized_audio/scene_00000.wav')
ctx = build_context(res)
print(answer('How many car horns?', ctx, res['timeline']))
"

# Phase 4 evaluation (306 test QA pairs; event IoU once per unique scene)
python src/eval/run_eval.py

# Phase 5 error analysis
python src/eval/error_analysis.py
```

## Runtime notes

- Tagger: spectral matched-filter, 1.0s window / 0.5s hop
- Runtime confidence threshold: **0.15** (validation label-count sweep selected 0.10 in `results/best_threshold.json`; that snapshot is not the runtime filter)
- Event detection in `run_eval.py` is computed **once per unique test scene** (45 scenes), reusing cached `tag_audio()` results
- Scene type in the structured context is **inferred from predicted labels**, not copied from ground-truth annotations
