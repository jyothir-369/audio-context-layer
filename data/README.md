# Data README — Phase 2 (Plan Sections 2.2–2.7)

## Sources

Plan 2.2 named ESC-50, UrbanSound8K, and optional FSD50K as mix sources. Those corpora are **not downloaded** in this PoC (`data/raw_sources/` is empty). `src/data_synthesis/source_prep.py` exits with a blocker if they are missing rather than fabricating clips.

Scenes are mixed from **class-specific synthetic tones** (`src/data_synthesis/audio_signatures.py`) plus a low sine ambience bed. This is a documented PoC deviation from “do not synthesize raw waveforms from scratch.”

## Synthetic dataset (verified)

- 300 scenes (`data/annotations/scene_*.json` and `data/synthesized_audio/scene_*.wav`)
- Duration: 25.0 s, 16 kHz mono PCM_16, 800044 bytes/file
- Events per scene: 2–5 foreground events from a scenario pool
- Overlap probability: 0.35
- Scenarios: street, kitchen, park, office, construction_site
- Event vocabulary (19 labels): car_horn, dog_bark, siren, engine, footstep, dish, water, microwave_beep, bird_chirp, bicycle_bell, wind_rustle, keyboard, printer, phone_ring, door_open, door_slam, drilling, hammer, shouting

WAV files are valid audio, not 11-byte placeholders.

## QA construction (Section 2.4, 2.5)

- Perceptual: first-event label / scenario lookup
- Counting: `count(events where label == X)` from timeline
- Temporal: sort events by start time
- Causal: only from hand-authored CAUSAL_TABLE when co-occurrence matches
- Negation: genuinely absent event
- Comparative: count comparison when ≥2 distinct labels present

Schema: `id`, `scene_id`, `audio_path`, `question`, `answer`, `question_type`, `supporting_events`.

## Split (Section 2.6)

Scene-level 70/15/15; zero leakage (disjoint scene IDs).

| Split | Scenes | QA pairs |
|---|---|---|
| train | 210 | 1404 |
| val | 45 | 300 |
| test | 45 | 306 |

Question-type counts (test): perceptual 90, counting 45, temporal 45, negation 45, comparative 45, causal 36.

**Holdout:** Plan 2.6 suggested holding out some event classes or scenarios from train. This split does **not** do that. All five scenarios and the full event vocabulary appear in train, val, and test.

Causal volume is lower than counting (229 vs 300 overall) because causal questions are emitted only when a CAUSAL_TABLE pair co-occurs.
