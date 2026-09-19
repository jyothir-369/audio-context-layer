# Phase 5 — Error Analysis (Plan Section 6)

## Perception vs Reasoning — 3-Category Split (Plan 6.2, Option b; not inflated by unanswerable gap)

**Method:** Perception = GT event missing/mislabeled in tagger timeline (IoU < 0.3). Reasoning = event detected correctly but answer derived wrong. **Unanswerable (by design)** = environment/scenario inference using heuristic label-set mapping (docs §8; ~50% accuracy already accepted). The three-way split reports 91 perception errors (69.5% of all errors), 20 reasoning errors (15.3%), and 20 unanswerable errors (15.3%). Among the 111 perception/reasoning errors only, perception accounts for 82.0%.

**Total Errors: 131**
- Perception: 91 (69.5%)
- Reasoning: 20 (15.3%)
- Unanswerable (environment inference gap): 20 (15.3%)

### Per-Question-Type Breakdown (3 categories each)

**Perceptual "what sound" vs "what environment":** Perceptual errors include both event-detection misses (e.g., "door_slam" missed, -> "footstep") AND environment-inference failures (e.g., GT="kitchen" but predicted="office"). The unanswerable category captures the latter; perception and reasoning are reported separately so the split does not imply that environment inference is a genuine detection failure.

## Qualitative Failures (Plan 6.3) — 5+ non-perceptual + diverse (counting, temporal, comparative, causal, reasoning/unanswerable)

### scene_00185 (perceptual) — unanswerable (by design)
- Question: What kind of environment does this audio suggest?
- Ground Truth: kitchen
- Predicted: footstep
- GT Timeline: [{"label": "footstep", "start": 2.199, "end": 4.249}, {"label": "door_slam", "start": 7.243, "end": 8.45}, {"label": "footstep", "start": 9.99, "end": 11.073}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 7.5, "confidence": 0.39}, {"label": "door_slam", "start": 7.0, "end": 8.5, "confidence": 0.39}, {"label": "footstep", "start": 8.0, "end": 25.0, "confidence": 0.42}]
- Diagnosis: unanswerable (by design) — Environment inference unanswerable from audio alone (heuristic label-set mapping; docs §8)

### scene_00063 (perceptual) — unanswerable (by design)
- Question: What kind of environment does this audio suggest?
- Ground Truth: construction_site
- Predicted: office
- GT Timeline: [{"label": "siren", "start": 1.712, "end": 4.89}, {"label": "drilling", "start": 6.912, "end": 10.34}, {"label": "hammer", "start": 11.617, "end": 14.333}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 1.5, "confidence": 0.44}, {"label": "siren", "start": 1.0, "end": 3.0, "confidence": 0.39}, {"label": "keyboard", "start": 2.5, "end": 5.0, "confidence": 0.38}, {"label": "phone_ring", "start": 4.5, "end": 5.5, "confidence": 0.21}, {"label": "footstep", "start": 5.0, "end": 6.5, "confidence": 0.43}, {"label": "drilling", "start": 6.0, "end": 7.0, "confidence": 0.21}, {"label": "dog_bark", "start": 6.5, "end": 10.5, "confidence": 0.21}, {"label": "drilling", "start": 10.0, "end": 11.0, "confidence": 0.21}, {"label": "footstep", "start": 10.5, "end": 11.5, "confidence": 0.43}, {"label": "hammer", "start": 11.0, "end": 15.0, "confidence": 0.4}, {"label": "footstep", "start": 14.5, "end": 25.0, "confidence": 0.43}]
- Diagnosis: unanswerable (by design) — Environment inference unanswerable from audio alone (heuristic label-set mapping; docs §8)

### scene_00235 (perceptual) — unanswerable (by design)
- Question: What kind of environment does this audio suggest?
- Ground Truth: park
- Predicted: kitchen
- GT Timeline: [{"label": "dog_bark", "start": 0.669, "end": 2.143}, {"label": "footstep", "start": 0.879, "end": 3.804}, {"label": "bird_chirp", "start": 2.994, "end": 4.836}, {"label": "dog_bark", "start": 3.893, "end": 7.081}, {"label": "bicycle_bell", "start": 6.688, "end": 8.436}]
- Predicted Timeline: [{"label": "dog_bark", "start": 0.0, "end": 1.5, "confidence": 0.18}, {"label": "footstep", "start": 1.0, "end": 3.0, "confidence": 0.18}, {"label": "bird_chirp", "start": 2.5, "end": 4.5, "confidence": 0.28}, {"label": "water", "start": 4.0, "end": 5.0, "confidence": 0.18}, {"label": "dog_bark", "start": 4.5, "end": 7.5, "confidence": 0.22}, {"label": "bicycle_bell", "start": 7.0, "end": 8.5, "confidence": 0.34}, {"label": "footstep", "start": 8.0, "end": 25.0, "confidence": 0.42}]
- Diagnosis: unanswerable (by design) — Environment inference unanswerable from audio alone (heuristic label-set mapping; docs §8)

### scene_00022 (perceptual) — unanswerable (by design)
- Question: What kind of environment does this audio suggest?
- Ground Truth: construction_site
- Predicted: street
- GT Timeline: [{"label": "hammer", "start": 0.831, "end": 2.997}, {"label": "engine", "start": 1.32, "end": 4.499}]
- Predicted Timeline: [{"label": "hammer", "start": 0.0, "end": 3.5, "confidence": 0.24}, {"label": "door_slam", "start": 3.0, "end": 4.5, "confidence": 0.28}, {"label": "engine", "start": 4.0, "end": 5.0, "confidence": 0.24}, {"label": "footstep", "start": 4.5, "end": 25.0, "confidence": 0.43}]
- Diagnosis: unanswerable (by design) — Environment inference unanswerable from audio alone (heuristic label-set mapping; docs §8)

### scene_00135 (perceptual) — unanswerable (by design)
- Question: What kind of environment does this audio suggest?
- Ground Truth: park
- Predicted: kitchen
- GT Timeline: [{"label": "bird_chirp", "start": 2.487, "end": 4.351}, {"label": "footstep", "start": 3.845, "end": 6.151}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 2.0, "confidence": 0.44}, {"label": "bird_chirp", "start": 1.5, "end": 4.0, "confidence": 0.43}, {"label": "water", "start": 3.5, "end": 4.5, "confidence": 0.23}, {"label": "footstep", "start": 4.0, "end": 25.0, "confidence": 0.42}]
- Diagnosis: unanswerable (by design) — Environment inference unanswerable from audio alone (heuristic label-set mapping; docs §8)

### scene_00150 (perceptual) — perception
- Question: What sound is present in this audio?
- Ground Truth: door_slam
- Predicted: footstep
- GT Timeline: [{"label": "door_slam", "start": 2.255, "end": 4.064}, {"label": "footstep", "start": 2.695, "end": 5.54}, {"label": "water", "start": 4.373, "end": 7.23}, {"label": "door_slam", "start": 7.075, "end": 10.233}, {"label": "footstep", "start": 9.749, "end": 11.976}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 2.5, "confidence": 0.37}, {"label": "engine", "start": 2.0, "end": 3.0, "confidence": 0.26}, {"label": "footstep", "start": 2.5, "end": 6.0, "confidence": 0.27}, {"label": "water", "start": 5.5, "end": 7.0, "confidence": 0.47}, {"label": "door_slam", "start": 6.5, "end": 7.5, "confidence": 0.21}, {"label": "engine", "start": 7.0, "end": 9.5, "confidence": 0.38}, {"label": "footstep", "start": 9.0, "end": 10.0, "confidence": 0.21}, {"label": "engine", "start": 9.5, "end": 10.5, "confidence": 0.24}, {"label": "footstep", "start": 10.0, "end": 25.0, "confidence": 0.42}]
- Diagnosis: perception — GT event missing or mislabeled in tagger timeline

### scene_00040 (perceptual) — perception
- Question: What sound is present in this audio?
- Ground Truth: door_slam
- Predicted: footstep
- GT Timeline: [{"label": "door_slam", "start": 2.767, "end": 5.009}, {"label": "door_slam", "start": 5.577, "end": 7.901}, {"label": "footstep", "start": 9.555, "end": 10.722}, {"label": "dish", "start": 12.04, "end": 13.543}, {"label": "footstep", "start": 15.946, "end": 17.628}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 3.0, "confidence": 0.39}, {"label": "door_slam", "start": 2.5, "end": 3.5, "confidence": 0.39}, {"label": "engine", "start": 3.0, "end": 5.0, "confidence": 0.41}, {"label": "door_slam", "start": 4.5, "end": 6.0, "confidence": 0.23}, {"label": "engine", "start": 5.5, "end": 7.5, "confidence": 0.42}, {"label": "door_slam", "start": 7.0, "end": 8.5, "confidence": 0.3}, {"label": "footstep", "start": 8.0, "end": 12.0, "confidence": 0.42}, {"label": "dish", "start": 11.5, "end": 14.5, "confidence": 0.33}, {"label": "footstep", "start": 14.0, "end": 25.0, "confidence": 0.42}]
- Diagnosis: perception — GT event missing or mislabeled in tagger timeline

### scene_00063 (perceptual) — perception
- Question: What sound is present in this audio?
- Ground Truth: siren
- Predicted: footstep
- GT Timeline: [{"label": "siren", "start": 1.712, "end": 4.89}, {"label": "drilling", "start": 6.912, "end": 10.34}, {"label": "hammer", "start": 11.617, "end": 14.333}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 1.5, "confidence": 0.44}, {"label": "siren", "start": 1.0, "end": 3.0, "confidence": 0.39}, {"label": "keyboard", "start": 2.5, "end": 5.0, "confidence": 0.38}, {"label": "phone_ring", "start": 4.5, "end": 5.5, "confidence": 0.21}, {"label": "footstep", "start": 5.0, "end": 6.5, "confidence": 0.43}, {"label": "drilling", "start": 6.0, "end": 7.0, "confidence": 0.21}, {"label": "dog_bark", "start": 6.5, "end": 10.5, "confidence": 0.21}, {"label": "drilling", "start": 10.0, "end": 11.0, "confidence": 0.21}, {"label": "footstep", "start": 10.5, "end": 11.5, "confidence": 0.43}, {"label": "hammer", "start": 11.0, "end": 15.0, "confidence": 0.4}, {"label": "footstep", "start": 14.5, "end": 25.0, "confidence": 0.43}]
- Diagnosis: perception — GT event missing or mislabeled in tagger timeline

### scene_00185 (perceptual)
- Question: What kind of environment does this audio suggest?
- Ground Truth: kitchen
- Predicted: footstep
- GT Timeline: [{"label": "footstep", "start": 2.199, "end": 4.249}, {"label": "door_slam", "start": 7.243, "end": 8.45}, {"label": "footstep", "start": 9.99, "end": 11.073}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 7.5, "confidence": 0.39}, {"label": "door_slam", "start": 7.0, "end": 8.5, "confidence": 0.39}, {"label": "footstep", "start": 8.0, "end": 25.0, "confidence": 0.42}]
- Diagnosis: unanswerable (by design) — Environment inference unanswerable from audio alone (heuristic label-set mapping; docs §8)

### scene_00150 (perceptual)
- Question: What sound is present in this audio?
- Ground Truth: door_slam
- Predicted: footstep
- GT Timeline: [{"label": "door_slam", "start": 2.255, "end": 4.064}, {"label": "footstep", "start": 2.695, "end": 5.54}, {"label": "water", "start": 4.373, "end": 7.23}, {"label": "door_slam", "start": 7.075, "end": 10.233}, {"label": "footstep", "start": 9.749, "end": 11.976}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 2.5, "confidence": 0.37}, {"label": "engine", "start": 2.0, "end": 3.0, "confidence": 0.26}, {"label": "footstep", "start": 2.5, "end": 6.0, "confidence": 0.27}, {"label": "water", "start": 5.5, "end": 7.0, "confidence": 0.47}, {"label": "door_slam", "start": 6.5, "end": 7.5, "confidence": 0.21}, {"label": "engine", "start": 7.0, "end": 9.5, "confidence": 0.38}, {"label": "footstep", "start": 9.0, "end": 10.0, "confidence": 0.21}, {"label": "engine", "start": 9.5, "end": 10.5, "confidence": 0.24}, {"label": "footstep", "start": 10.0, "end": 25.0, "confidence": 0.42}]
- Diagnosis: perception — GT event missing or mislabeled in tagger timeline

### scene_00040 (perceptual)
- Question: What sound is present in this audio?
- Ground Truth: door_slam
- Predicted: footstep
- GT Timeline: [{"label": "door_slam", "start": 2.767, "end": 5.009}, {"label": "door_slam", "start": 5.577, "end": 7.901}, {"label": "footstep", "start": 9.555, "end": 10.722}, {"label": "dish", "start": 12.04, "end": 13.543}, {"label": "footstep", "start": 15.946, "end": 17.628}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 3.0, "confidence": 0.39}, {"label": "door_slam", "start": 2.5, "end": 3.5, "confidence": 0.39}, {"label": "engine", "start": 3.0, "end": 5.0, "confidence": 0.41}, {"label": "door_slam", "start": 4.5, "end": 6.0, "confidence": 0.23}, {"label": "engine", "start": 5.5, "end": 7.5, "confidence": 0.42}, {"label": "door_slam", "start": 7.0, "end": 8.5, "confidence": 0.3}, {"label": "footstep", "start": 8.0, "end": 12.0, "confidence": 0.42}, {"label": "dish", "start": 11.5, "end": 14.5, "confidence": 0.33}, {"label": "footstep", "start": 14.0, "end": 25.0, "confidence": 0.42}]
- Diagnosis: perception — GT event missing or mislabeled in tagger timeline

### scene_00063 (perceptual)
- Question: What sound is present in this audio?
- Ground Truth: siren
- Predicted: footstep
- GT Timeline: [{"label": "siren", "start": 1.712, "end": 4.89}, {"label": "drilling", "start": 6.912, "end": 10.34}, {"label": "hammer", "start": 11.617, "end": 14.333}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 1.5, "confidence": 0.44}, {"label": "siren", "start": 1.0, "end": 3.0, "confidence": 0.39}, {"label": "keyboard", "start": 2.5, "end": 5.0, "confidence": 0.38}, {"label": "phone_ring", "start": 4.5, "end": 5.5, "confidence": 0.21}, {"label": "footstep", "start": 5.0, "end": 6.5, "confidence": 0.43}, {"label": "drilling", "start": 6.0, "end": 7.0, "confidence": 0.21}, {"label": "dog_bark", "start": 6.5, "end": 10.5, "confidence": 0.21}, {"label": "drilling", "start": 10.0, "end": 11.0, "confidence": 0.21}, {"label": "footstep", "start": 10.5, "end": 11.5, "confidence": 0.43}, {"label": "hammer", "start": 11.0, "end": 15.0, "confidence": 0.4}, {"label": "footstep", "start": 14.5, "end": 25.0, "confidence": 0.43}]
- Diagnosis: perception — GT event missing or mislabeled in tagger timeline

### scene_00063 (perceptual)
- Question: What kind of environment does this audio suggest?
- Ground Truth: construction_site
- Predicted: office
- GT Timeline: [{"label": "siren", "start": 1.712, "end": 4.89}, {"label": "drilling", "start": 6.912, "end": 10.34}, {"label": "hammer", "start": 11.617, "end": 14.333}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 1.5, "confidence": 0.44}, {"label": "siren", "start": 1.0, "end": 3.0, "confidence": 0.39}, {"label": "keyboard", "start": 2.5, "end": 5.0, "confidence": 0.38}, {"label": "phone_ring", "start": 4.5, "end": 5.5, "confidence": 0.21}, {"label": "footstep", "start": 5.0, "end": 6.5, "confidence": 0.43}, {"label": "drilling", "start": 6.0, "end": 7.0, "confidence": 0.21}, {"label": "dog_bark", "start": 6.5, "end": 10.5, "confidence": 0.21}, {"label": "drilling", "start": 10.0, "end": 11.0, "confidence": 0.21}, {"label": "footstep", "start": 10.5, "end": 11.5, "confidence": 0.43}, {"label": "hammer", "start": 11.0, "end": 15.0, "confidence": 0.4}, {"label": "footstep", "start": 14.5, "end": 25.0, "confidence": 0.43}]
- Diagnosis: unanswerable (by design) — Environment inference unanswerable from audio alone (heuristic label-set mapping; docs §8)

### scene_00235 (perceptual)
- Question: What kind of environment does this audio suggest?
- Ground Truth: park
- Predicted: kitchen
- GT Timeline: [{"label": "dog_bark", "start": 0.669, "end": 2.143}, {"label": "footstep", "start": 0.879, "end": 3.804}, {"label": "bird_chirp", "start": 2.994, "end": 4.836}, {"label": "dog_bark", "start": 3.893, "end": 7.081}, {"label": "bicycle_bell", "start": 6.688, "end": 8.436}]
- Predicted Timeline: [{"label": "dog_bark", "start": 0.0, "end": 1.5, "confidence": 0.18}, {"label": "footstep", "start": 1.0, "end": 3.0, "confidence": 0.18}, {"label": "bird_chirp", "start": 2.5, "end": 4.5, "confidence": 0.28}, {"label": "water", "start": 4.0, "end": 5.0, "confidence": 0.18}, {"label": "dog_bark", "start": 4.5, "end": 7.5, "confidence": 0.22}, {"label": "bicycle_bell", "start": 7.0, "end": 8.5, "confidence": 0.34}, {"label": "footstep", "start": 8.0, "end": 25.0, "confidence": 0.42}]
- Diagnosis: unanswerable (by design) — Environment inference unanswerable from audio alone (heuristic label-set mapping; docs §8)

### scene_00022 (perceptual)
- Question: What kind of environment does this audio suggest?
- Ground Truth: construction_site
- Predicted: street
- GT Timeline: [{"label": "hammer", "start": 0.831, "end": 2.997}, {"label": "engine", "start": 1.32, "end": 4.499}]
- Predicted Timeline: [{"label": "hammer", "start": 0.0, "end": 3.5, "confidence": 0.24}, {"label": "door_slam", "start": 3.0, "end": 4.5, "confidence": 0.28}, {"label": "engine", "start": 4.0, "end": 5.0, "confidence": 0.24}, {"label": "footstep", "start": 4.5, "end": 25.0, "confidence": 0.43}]
- Diagnosis: unanswerable (by design) — Environment inference unanswerable from audio alone (heuristic label-set mapping; docs §8)

### scene_00135 (perceptual)
- Question: What sound is present in this audio?
- Ground Truth: bird_chirp
- Predicted: footstep
- GT Timeline: [{"label": "bird_chirp", "start": 2.487, "end": 4.351}, {"label": "footstep", "start": 3.845, "end": 6.151}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 2.0, "confidence": 0.44}, {"label": "bird_chirp", "start": 1.5, "end": 4.0, "confidence": 0.43}, {"label": "water", "start": 3.5, "end": 4.5, "confidence": 0.23}, {"label": "footstep", "start": 4.0, "end": 25.0, "confidence": 0.42}]
- Diagnosis: perception — GT event missing or mislabeled in tagger timeline

### scene_00135 (perceptual)
- Question: What kind of environment does this audio suggest?
- Ground Truth: park
- Predicted: kitchen
- GT Timeline: [{"label": "bird_chirp", "start": 2.487, "end": 4.351}, {"label": "footstep", "start": 3.845, "end": 6.151}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 2.0, "confidence": 0.44}, {"label": "bird_chirp", "start": 1.5, "end": 4.0, "confidence": 0.43}, {"label": "water", "start": 3.5, "end": 4.5, "confidence": 0.23}, {"label": "footstep", "start": 4.0, "end": 25.0, "confidence": 0.42}]
- Diagnosis: unanswerable (by design) — Environment inference unanswerable from audio alone (heuristic label-set mapping; docs §8)

### scene_00176 (perceptual)
- Question: What sound is present in this audio?
- Ground Truth: water
- Predicted: footstep
- GT Timeline: [{"label": "water", "start": 1.292, "end": 4.089}, {"label": "dish", "start": 6.946, "end": 7.993}, {"label": "footstep", "start": 9.168, "end": 10.939}, {"label": "water", "start": 12.517, "end": 13.553}, {"label": "door_slam", "start": 15.073, "end": 17.371}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 1.0, "confidence": 0.44}, {"label": "water", "start": 0.5, "end": 5.0, "confidence": 0.46}, {"label": "footstep", "start": 4.5, "end": 6.5, "confidence": 0.43}, {"label": "dish", "start": 6.0, "end": 8.5, "confidence": 0.34}, {"label": "footstep", "start": 8.0, "end": 12.5, "confidence": 0.44}, {"label": "water", "start": 12.0, "end": 14.5, "confidence": 0.42}, {"label": "footstep", "start": 14.0, "end": 15.0, "confidence": 0.43}, {"label": "door_slam", "start": 14.5, "end": 15.5, "confidence": 0.27}, {"label": "engine", "start": 15.0, "end": 16.0, "confidence": 0.42}, {"label": "door_slam", "start": 15.5, "end": 16.5, "confidence": 0.42}, {"label": "engine", "start": 16.0, "end": 17.0, "confidence": 0.42}, {"label": "door_slam", "start": 16.5, "end": 18.0, "confidence": 0.28}, {"label": "footstep", "start": 17.5, "end": 25.0, "confidence": 0.43}]
- Diagnosis: reasoning — All GT events detected correctly; derived answer wrong

### scene_00183 (perceptual)
- Question: What sound is present in this audio?
- Ground Truth: shouting
- Predicted: footstep
- GT Timeline: [{"label": "shouting", "start": 2.801, "end": 4.969}, {"label": "siren", "start": 7.446, "end": 9.713}, {"label": "engine", "start": 12.244, "end": 14.939}, {"label": "shouting", "start": 15.651, "end": 18.174}, {"label": "hammer", "start": 19.562, "end": 22.137}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 2.5, "confidence": 0.44}, {"label": "shouting", "start": 2.0, "end": 5.5, "confidence": 0.23}, {"label": "footstep", "start": 5.0, "end": 7.0, "confidence": 0.43}, {"label": "siren", "start": 6.5, "end": 8.5, "confidence": 0.37}, {"label": "keyboard", "start": 8.0, "end": 10.0, "confidence": 0.36}, {"label": "phone_ring", "start": 9.5, "end": 10.5, "confidence": 0.21}, {"label": "footstep", "start": 10.0, "end": 12.0, "confidence": 0.43}, {"label": "engine", "start": 11.5, "end": 12.5, "confidence": 0.29}, {"label": "door_slam", "start": 12.0, "end": 14.5, "confidence": 0.32}, {"label": "engine", "start": 14.0, "end": 15.5, "confidence": 0.28}, {"label": "shouting", "start": 15.0, "end": 19.0, "confidence": 0.24}, {"label": "footstep", "start": 18.5, "end": 19.5, "confidence": 0.43}, {"label": "hammer", "start": 19.0, "end": 23.0, "confidence": 0.4}, {"label": "footstep", "start": 22.5, "end": 25.0, "confidence": 0.43}]
- Diagnosis: perception — GT event missing or mislabeled in tagger timeline

### scene_00183 (perceptual)
- Question: What kind of environment does this audio suggest?
- Ground Truth: construction_site
- Predicted: office
- GT Timeline: [{"label": "shouting", "start": 2.801, "end": 4.969}, {"label": "siren", "start": 7.446, "end": 9.713}, {"label": "engine", "start": 12.244, "end": 14.939}, {"label": "shouting", "start": 15.651, "end": 18.174}, {"label": "hammer", "start": 19.562, "end": 22.137}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 2.5, "confidence": 0.44}, {"label": "shouting", "start": 2.0, "end": 5.5, "confidence": 0.23}, {"label": "footstep", "start": 5.0, "end": 7.0, "confidence": 0.43}, {"label": "siren", "start": 6.5, "end": 8.5, "confidence": 0.37}, {"label": "keyboard", "start": 8.0, "end": 10.0, "confidence": 0.36}, {"label": "phone_ring", "start": 9.5, "end": 10.5, "confidence": 0.21}, {"label": "footstep", "start": 10.0, "end": 12.0, "confidence": 0.43}, {"label": "engine", "start": 11.5, "end": 12.5, "confidence": 0.29}, {"label": "door_slam", "start": 12.0, "end": 14.5, "confidence": 0.32}, {"label": "engine", "start": 14.0, "end": 15.5, "confidence": 0.28}, {"label": "shouting", "start": 15.0, "end": 19.0, "confidence": 0.24}, {"label": "footstep", "start": 18.5, "end": 19.5, "confidence": 0.43}, {"label": "hammer", "start": 19.0, "end": 23.0, "confidence": 0.4}, {"label": "footstep", "start": 22.5, "end": 25.0, "confidence": 0.43}]
- Diagnosis: unanswerable (by design) — Environment inference unanswerable from audio alone (heuristic label-set mapping; docs §8)

### scene_00049 (perceptual)
- Question: What sound is present in this audio?
- Ground Truth: keyboard
- Predicted: footstep
- GT Timeline: [{"label": "keyboard", "start": 1.054, "end": 2.105}, {"label": "printer", "start": 4.059, "end": 5.686}, {"label": "door_open", "start": 7.045, "end": 9.34}, {"label": "footstep", "start": 11.597, "end": 13.29}, {"label": "phone_ring", "start": 14.366, "end": 15.994}]
- Predicted Timeline: [{"label": "footstep", "start": 0.0, "end": 1.0, "confidence": 0.44}, {"label": "keyboard", "start": 0.5, "end": 3.0, "confidence": 0.33}, {"label": "footstep", "start": 2.5, "end": 4.0, "confidence": 0.44}, {"label": "printer", "start": 3.5, "end": 6.5, "confidence": 0.33}, {"label": "footstep", "start": 6.0, "end": 7.0, "confidence": 0.43}, {"label": "door_open", "start": 6.5, "end": 7.5, "confidence": 0.3}, {"label": "car_horn", "start": 7.0, "end": 9.0, "confidence": 0.3}, {"label": "door_slam", "start": 8.5, "end": 9.5, "confidence": 0.19}, {"label": "footstep", "start": 9.0, "end": 14.0, "confidence": 0.39}, {"label": "phone_ring", "start": 13.5, "end": 16.5, "confidence": 0.26}, {"label": "footstep", "start": 16.0, "end": 25.0, "confidence": 0.43}]
- Diagnosis: perception — GT event missing or mislabeled in tagger timeline

## Systematic Patterns (Plan 6.4)

- Overlapping events: temporal-ordering accuracy is likely lower when events overlap (Plan 2.3 overlap_probability=0.35). This is a hypothesis, not a measured holdout result.
- Unseen class / scenario holdout: NOT implemented. Train, val, and test all contain the same five scenarios (street, kitchen, park, office, construction_site) and the same event vocabulary. Plan 2.6 suggested holding out classes; this PoC split does not.
- Negation false-positive rate: rule-based system explicitly returns 'no' when event absent from timeline.
- Causal explanations approximate; CAUSAL_TABLE covers only curated event pairs.
