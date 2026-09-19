# Implementation Plan: Audio Context Layer (Audio Question Answering)

This plan turns the problem statement into a concrete, buildable, end-to-end pipeline: dataset → model → training → evaluation → documentation. It is written so you (or anyone on the team) can execute it top to bottom without re-deriving decisions.

---

## 0. Strategy Decision (read this first)

Given this is a time-boxed PoC/assessment, two tracks are proposed. **Build Track A fully — it is the core deliverable.** Attempt Track B only if time remains; it is the "bonus" that shows depth.

| | Track A — Structured/Grounded Pipeline (core) | Track B — End-to-End Trainable Audio-LLM (bonus) |
|---|---|---|
| Idea | Detect audio events with a pretrained tagger → build a structured "scene representation" → answer questions over that structured context (template + small LLM) | Audio encoder embeddings → trainable projection → frozen/LoRA-tuned small LLM, trained directly on (audio, question) → answer |
| Why | Fast to build, fully controllable, ground truth is known by construction (since you synthesize the audio), easy to evaluate per question type, highly interpretable — good for error analysis | Closer to "real" audio-QA research (Pengi/Audio Flamingo/Qwen2-Audio style), shows you can train a multimodal model, more impressive if it works |
| Risk | Looks more like an NLP pipeline than a "model" | Needs GPU, longer training time, harder to debug, easy to run out of time |
| Recommendation | Do this first, completely, well-documented | Do this second, as an ablation/extension, even on a subset |

Both tracks share the **same dataset** and the **same evaluation harness** — this is the key design decision that makes the two tracks comparable and lets you report both in one results table.

---

## 1. Repository Structure

```
audio-context-layer/
├── README.md
├── requirements.txt
├── configs/
│   ├── dataset_synthesis.yaml
│   ├── track_a.yaml
│   └── track_b.yaml
├── data/
│   ├── raw_sources/           # downloaded ESC-50 / UrbanSound8K / FSD50K clips (gitignored)
│   ├── synthesized_audio/     # generated .wav scenes
│   ├── annotations/           # ground-truth event timelines (json)
│   └── qa_pairs/              # final QA dataset (train/val/test .jsonl)
├── src/
│   ├── data_synthesis/
│   │   ├── source_prep.py         # download + index source sound-event clips
│   │   ├── scene_composer.py      # mixes events into multi-event audio scenes
│   │   └── metadata_writer.py     # writes per-scene ground-truth timeline JSON
│   ├── qa_generation/
│   │   ├── templates.py           # question templates per type
│   │   ├── generate_qa.py         # timeline -> QA pairs
│   │   └── paraphrase.py          # optional LLM-based paraphrasing for diversity
│   ├── track_a_structured/
│   │   ├── event_tagger.py        # wraps pretrained PANNs/YAMNet/CLAP tagger
│   │   ├── scene_graph_builder.py # tagger output -> structured context
│   │   └── qa_answerer.py         # structured context + question -> answer (rules + small LLM)
│   ├── track_b_e2e/
│   │   ├── model.py               # audio encoder + projector + LLM head
│   │   ├── train.py
│   │   └── infer.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── run_eval.py
│   │   └── error_analysis.py
│   └── utils/
│       ├── audio_io.py
│       └── seed.py
├── notebooks/
│   └── exploration.ipynb
├── results/
│   ├── metrics_track_a.json
│   ├── metrics_track_b.json
│   ├── loss_curves.png
│   └── error_analysis.md
└── docs/
    └── technical_report.md    # final deliverable, see Section 8
```

Set up with a clean virtual environment and pin versions in `requirements.txt` from day one — reproducibility is explicitly graded.

---

## 2. Dataset Construction (Synthetic — for bonus points)

### 2.1 Why synthetic

Real audio-QA datasets (Clotho-AQA, AudioCaps) exist, but building your own **synthetic dataset gives you ground truth by construction** — you know exactly what events are in the audio, when, and how many times, which means QA pairs can be generated programmatically and are verifiably correct. This directly satisfies "bonus points for curating a synthetic dataset."

### 2.2 Source material

Use existing labeled sound-event clips as building blocks (do not synthesize raw waveforms from scratch):

- **ESC-50**: 2,000 clips, 5 seconds each, 50 balanced classes (animals, natural soundscapes, human non-speech, interior/domestic, exterior/urban).
- **UrbanSound8K**: 8,732 labeled urban sound clips, 10 classes (dog bark, car horn, siren, drilling, etc.), variable length.
- **FSD50K** (optional, larger vocabulary): weakly labeled, more event diversity.

Download via their official pages/Zenodo links; store under `data/raw_sources/`, index into a manifest CSV (`label, filepath, duration`).

### 2.3 Scene synthesis procedure (`scene_composer.py`)

For each synthetic sample:

1. **Sample a scenario** from a curated list of scenario templates, e.g. `"street"`, `"kitchen"`, `"park"`, `"office"`, `"construction_site"`. Each scenario defines a plausible pool of event classes and a background ambience.
2. **Sample 2–5 foreground events** from that pool's allowed classes (with repetition allowed — this is what makes counting questions meaningful).
3. **Place events on a timeline** of fixed total length (e.g. 20–30s): sample start times, allow some overlap for a subset of scenes (to create harder "who happened when" cases) and clear separation for others (to create clean temporal-ordering cases).
4. **Optionally add a low-level ambience/background track** (e.g. traffic hum, room tone) mixed at lower SNR, using `pydub`/`librosa` for mixing and gain control.
5. **Render to `.wav`** (16kHz mono is sufficient) and save.
6. **Write ground-truth timeline JSON** alongside it:

```json
{
  "scene_id": "scene_00042",
  "audio_path": "synthesized_audio/scene_00042.wav",
  "duration_sec": 24.0,
  "scenario": "street",
  "events": [
    {"label": "car_horn", "start": 2.1, "end": 3.4},
    {"label": "dog_bark", "start": 5.0, "end": 5.6},
    {"label": "dog_bark", "start": 12.3, "end": 12.9},
    {"label": "siren", "start": 15.0, "end": 19.0}
  ],
  "background": "traffic_ambience_low"
}
```

Target dataset size for a PoC: **~300–500 synthetic scenes** is enough to produce several thousand QA pairs once you factor in multiple questions per scene — small enough to build and verify by hand-spot-checking, large enough for a meaningful train/val/test split. State this scale honestly in the documentation as an intentional PoC-scale choice.

### 2.4 QA generation from timelines (`qa_generation/`)

Because the timeline JSON is exact, questions and answers can be templated deterministically, then optionally paraphrased for lexical diversity. Cover at minimum these types:

| Type | Example question template | How answer is derived |
|---|---|---|
| **Perceptual/apparent** | "What sound is present in this audio?" / "What kind of environment does this audio suggest?" | Direct lookup of `events[].label`; environment inferred from `scenario` field |
| **Counting** | "How many times does a dog bark?" | `count(events where label == X)` |
| **Temporal** | "What happens right after the car horn?" / "Does the siren start before or after the second dog bark?" | Sort `events` by `start`, compare timestamps |
| **Causal/reasoning** | "Why might the dog be barking?" / "Why does this sound like a street scene?" | From a curated **causal knowledge table** mapping event/co-occurrence patterns → plausible explanations (e.g. `dog_bark + doorbell → "someone likely arrived at the door"`); this table is hand-authored (10–20 entries) since causal explanation isn't automatically derivable from timestamps alone |
| **Negation/absence** (recommended addition) | "Is there a siren in this audio?" (when there isn't) | Directly checkable from timeline; important for testing hallucination |
| **Comparative** (recommended addition) | "Which happened more often, dog barks or car horns?" | Compare counts |

For each scene, generate a fixed quota of questions per type (e.g. 2 perceptual, 2 counting, 2 temporal, 1–2 causal, 1 negation) so the dataset is balanced across types by construction — this matters for the required "breakdown across question types."

Optionally run generated questions through an LLM paraphraser (one call per template, batched) to avoid every question in the dataset looking machine-templated; keep the original template + ground truth linked in metadata so correctness is never at risk from paraphrasing.

### 2.5 Final QA schema (`data/qa_pairs/*.jsonl`)

```json
{"id": "scene_00042_q03", "scene_id": "scene_00042", "audio_path": "...", "question": "How many times does a dog bark?", "answer": "2", "question_type": "counting", "supporting_events": ["dog_bark@5.0", "dog_bark@12.3"]}
```

Keeping `supporting_events` lets the error analysis later distinguish "wrong perception" errors from "wrong reasoning-over-correct-perception" errors.

### 2.6 Splits

- Split at the **scene level** (never split questions from the same scene across sets — avoids leakage).
- 70% train / 15% validation / 15% test.
- Stratify the split so each set has a similar distribution of scenarios and question types (`sklearn.model_selection.train_test_split` with a composite stratification key, or manual bucketing).
- Hold out at least a few **event classes or scenario types entirely from train** and only present in test, to report a "generalization to unseen sound classes" number — a nice thing to highlight in results.

### 2.7 Data documentation

Write `data/README.md` (feeds into the final report) covering: source datasets and licenses, synthesis procedure, QA template list with counts, class balance table, split statistics, and 3–5 fully worked examples (audio description + timeline + generated QA).

---

## 3. Track A — Structured/Grounded Pipeline (core deliverable)

This is the primary system. It is interpretable and lets every answer be traced back to a detected event, which makes both evaluation and error analysis clean.

### 3.1 Audio event tagging (`event_tagger.py`)

Use an **off-the-shelf pretrained audio tagging model** — do not train this from scratch:
- **PANNs (CNN14)** trained on AudioSet — good general-purpose tagger with a broad label set, easy `pip install panns-inference`.
- Alternative: **YAMNet** (TF Hub) or **CLAP** (LAION-CLAP) for zero-shot event matching against your event vocabulary via text-audio similarity (this is convenient because you can zero-shot match directly to your synthesis label set instead of AudioSet's 527 classes).
- Recommendation: use CLAP zero-shot classification restricted to the closed vocabulary you used for synthesis (this gives higher precision since you're not trying to map 527 AudioSet labels back to your ~30–50 synthesis classes).

Run tagging with a **sliding window** (e.g. 1s windows, 0.5s hop) over each audio file to get a time-resolved activation curve per class, then threshold + merge consecutive active windows into detected event segments (`start`, `end`, `label`, `confidence`). This reconstructs a predicted timeline structurally identical to your ground-truth timeline — which is what makes evaluation straightforward (compare predicted timeline vs ground-truth timeline directly, independent of the QA layer).

### 3.2 Scene graph / structured context (`scene_graph_builder.py`)

Convert the predicted timeline into a compact **textual context block** that will be given to the QA answering step, e.g.:

```
Detected events (chronological):
1. car_horn at 2.1–3.4s
2. dog_bark at 5.0–5.6s
3. dog_bark at 12.3–12.9s
4. siren at 15.0–19.0s
Scene type (inferred): street
```

This is effectively a deterministic "audio → text" translation, which reframes multimodal QA as text QA grounded in a retrieved/generated context — a well-justified design simplification to call out explicitly in the report.

### 3.3 Question answering over structured context (`qa_answerer.py`)

Two sub-approaches, implement both and compare:

1. **Rule-based answerer**: parse the question type (simple intent classifier — keyword/regex based, e.g. "how many" → counting, "before/after" → temporal, "why" → causal) and compute the answer directly from the structured timeline (counting, sorting, lookup, causal-table matching). Zero hallucination risk, but brittle to question phrasing.
2. **LLM-grounded answerer**: feed `{structured context + question}` to a small instruction-tuned LLM (e.g. **Qwen2.5-1.5B-Instruct**, **Phi-3.5-mini**, or an API-based model if permitted) with a system prompt instructing it to answer *only* from the given context and say "not enough information" if the context doesn't support an answer. This handles paraphrased/free-form questions the rule-based system can't parse, at the cost of occasional hallucination — which is exactly the kind of trade-off worth reporting.

Report both numbers; this becomes a nice ablation in the results section ("does grounding actually help vs. a free LLM guessing from the question alone").

### 3.4 "No training required" framing

Track A has **no learned parameters trained by you** except the optional zero-shot threshold tuning on the validation set — be explicit about this in documentation. The "Model Implementation — reproducible scripts for training and evaluation" requirement is satisfied by the **tagger threshold-tuning script + the eval script**, since there's no gradient-based training in this track. This is a legitimate design choice for a PoC, not a shortcut — say so explicitly rather than implying training happened.

---

## 4. Track B — End-to-End Trainable Audio-LLM (bonus, optional)

Only pursue this after Track A is fully working and evaluated. Keep scope small.

### 4.1 Architecture

```
Audio waveform
   │
   ▼
Frozen audio encoder (CLAP or Whisper encoder) → sequence of audio embeddings
   │
   ▼
Trainable linear/MLP projector → maps audio embedding dim → LLM hidden dim
   │
   ▼
Projected audio tokens prepended to text prompt
   │
   ▼
Small LLM (e.g. Qwen2.5-0.5B/1.5B or TinyLlama) — LoRA fine-tuned
   │
   ▼
Generated answer text
```

This mirrors the Pengi / LLaVA-style "frozen encoder + trainable adapter + LoRA-tuned LLM" recipe, scaled down for PoC compute budgets.

### 4.2 Training setup

- **Only train**: the projector (from scratch) + LoRA adapters on the LLM (rank 8–16). Keep the audio encoder and base LLM weights frozen — this is what makes training feasible on a single consumer/free-tier GPU.
- **Loss**: standard next-token cross-entropy on the answer tokens only (mask the prompt/context tokens from the loss).
- **Batching**: pad/truncate audio to a fixed max duration (e.g. 30s) for batch collation.
- **Hyperparameters to log**: learning rate, batch size, LoRA rank, number of epochs, warmup steps — put these in `configs/track_b.yaml` so the run is reproducible from the config alone.
- **Track and save**: training/validation loss curves every N steps → `results/loss_curves.png` (required deliverable).
- **Checkpointing**: save best checkpoint by validation loss; also save the final checkpoint.

### 4.3 Scope control

If compute/time is limited, train on a subset (e.g. train split only, skip full hyperparameter search) and say so explicitly — a documented, honest limitation is worth more than a silently under-trained model presented as final.

---

## 5. Evaluation (`src/eval/`)

Use **one evaluation harness for both tracks** so results are directly comparable.

### 5.1 Metrics by question type

| Question type | Primary metric | Notes |
|---|---|---|
| Perceptual (what/environment) | Accuracy (exact/normalized match) + F1 against multi-label ground truth if multiple correct events exist | Normalize synonyms (e.g. "puppy" ~ "dog") via a small alias table |
| Counting | Exact-match accuracy + Mean Absolute Error (MAE) | MAE is more informative than accuracy alone since "off by one" is a different failure than "off by five" |
| Temporal | Accuracy (before/after/order correctness) | For "what happens after X" style questions, match against the correct next event label |
| Causal/reasoning | Semantic similarity to reference explanation (BERTScore or cosine similarity of sentence embeddings) + human/LLM-judge rubric score (1–5) | Free-form text rarely has one exact correct string; combine an automatic proxy metric with a small manual/LLM-judged rubric pass on a sample |
| Negation/absence | Accuracy (yes/no) + false-positive rate | Specifically tracks hallucination — report this prominently |
| Overall | Weighted average of the above; also report exact-match and a text-similarity metric (BLEU/ROUGE-L or BERTScore) uniformly across all answers for a single headline number | |

Also report **event-detection metrics independently** for Track A (precision/recall/F1 of the tagger's predicted timeline vs. ground truth, e.g. at IoU ≥ 0.3 on time segments) — this separates "did the model hear it right" from "did it reason about it right," which is essential for good error analysis.

### 5.2 Running evaluation

`run_eval.py` should:
1. Load the test split.
2. Run inference for each track (and each sub-approach in Track A).
3. Compute per-question-type metric tables.
4. Save a results JSON + a Markdown table (auto-generated) for direct inclusion in the report.

### 5.3 Statistical care

- Report metrics with the **test-set sample count per question type** alongside the numbers (small-N types will have noisy metrics — say so).
- If feasible, bootstrap a rough confidence interval on the headline accuracy number.

---

## 6. Error Analysis (`error_analysis.py`, `results/error_analysis.md`)

Produce, at minimum:

1. **Confusion by question type**: a bar chart of accuracy per type, immediately shows weakest capability.
2. **Perception vs. reasoning error split**: using `supporting_events` from the QA schema and the tagger's predicted timeline, classify each wrong answer as either (a) the underlying event wasn't detected/was misdetected (perception failure) or (b) the event was detected correctly but the answer over it was wrong (reasoning failure). This is one of the most valuable analyses you can present — it directly tells you where to invest next.
3. **Qualitative examples**: 8–10 hand-picked failure cases with audio description, question, ground truth, model answer, and a one-line diagnosis each.
4. **Systematic patterns**: e.g. "counting accuracy drops when events overlap in time," "causal questions fail more on unseen scenario types," "negation questions have a high false-positive rate — model tends to assume presence." State these as hypotheses backed by the confusion breakdown, not just anecdotes.

---

## 7. Suggested Build Timeline

| Phase | Content | Rough effort |
|---|---|---|
| 1 | Source data download + indexing, scene synthesis pipeline, generate ~300–500 scenes | 1 day |
| 2 | QA template design + generation + optional paraphrasing + manual spot-check of ~30 samples for correctness | 1 day |
| 3 | Track A: tagger integration, scene graph builder, rule-based answerer | 1 day |
| 4 | Track A: LLM-grounded answerer + prompt iteration | 0.5 day |
| 5 | Evaluation harness + metrics + error analysis for Track A | 1 day |
| 6 (optional) | Track B: model/training/eval | 1–2 days |
| 7 | Documentation write-up, plots, final polish | 1 day |

Adjust to your actual deadline — if compressed, drop Track B first, then reduce dataset size, but never skip the error analysis or the documentation.

---

## 8. Final Documentation Deliverable (`docs/technical_report.md`)

Structure it to mirror the assessment's requirements exactly, so a grader can map each section 1:1:

1. **Problem formulation** — restate the task, define what "grounded audio QA" means here, define the question taxonomy you support and why (what/counting/temporal/causal/+ negation/comparative if added).
2. **Research study** — brief literature grounding: reference the general ideas behind CLAP, PANNs/YAMNet, Clotho-AQA/AudioCaps as related datasets, and Pengi/Audio-Flamingo/Qwen2-Audio as related architectures. State clearly what you adopted vs. what you simplified for PoC scope, and why (compute/time budget).
3. **Dataset description** — synthesis methodology (Section 2), scale, class balance, split stats, example entries, licensing note for source corpora.
4. **Method** — describe Track A end-to-end (and Track B if built) with an architecture diagram, and explicitly justify each design decision (why zero-shot CLAP over training a tagger; why a structured intermediate representation instead of raw end-to-end from the start; why LoRA + frozen encoder for Track B; why these specific metrics per question type).
5. **Experimental setup** — hardware, library versions, hyperparameters (link to configs), how to reproduce (`README.md` commands).
6. **Results** — metric tables per question type per track, overall headline numbers, event-detection metrics, Track A vs Track B comparison if both built.
7. **Loss curves** — Track B training/validation loss plot, with a short note if Track A was not trained (explain why, per Section 3.4).
8. **Observations and limitations** — dataset scale limitations, causal-question evaluation being inherently approximate, known failure modes from Section 6, what you'd do with more time/compute (larger dataset, real recorded audio, full Track B hyperparameter sweep, human evaluation for causal answers).

Keep the report readable end-to-end without needing to open the code — figures and tables should be self-contained with captions.

---

## 9. Key Libraries/Tools Checklist

- Audio I/O/mixing: `librosa`, `soundfile`, `pydub`
- Tagging: `panns-inference` or `laion-clap`
- LLM inference: `transformers`, `peft` (LoRA), optionally `bitsandbytes` for quantized loading
- Metrics: `bert-score`, `scikit-learn`, `nltk`/`rouge-score`
- Experiment tracking (optional but recommended): `wandb` or a simple CSV logger — either way, produce the loss curve plot with `matplotlib`
- Config management: `pyyaml` + a simple dataclass loader

---

## 10. Definition of Done Checklist

- [ ] Source audio downloaded and indexed
- [ ] ≥300 synthetic scenes generated with exact ground-truth timelines
- [ ] QA pairs generated, covering all required types + at least one bonus type, balanced by type
- [ ] Train/val/test split done at scene level, documented with stats
- [ ] Track A tagger + structured context + rule-based and LLM-grounded answerers implemented and runnable end-to-end from a single script/command
- [ ] Evaluation harness produces per-type metrics + overall metrics + event-detection metrics
- [ ] Error analysis produced, including perception-vs-reasoning split and qualitative examples
- [ ] (Optional) Track B trained, loss curves saved, evaluated with the same harness
- [ ] `docs/technical_report.md` written covering all 8 required sections
- [ ] `README.md` at repo root with exact setup + run commands for a fresh clone to reproduce everything
