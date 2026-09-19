#!/usr/bin/env python3
"""Phase 2 — Generate deterministic QA pairs from timeline JSON (Plan Sections 2.4-2.7)."""
import json, random, os

from templates import TEMPLATES

# Extended causal co-occurrence table — populated with actual synthesis vocabulary
# pairs that occur naturally in scenes (verified against dataset co-occurrence counts).
CAUSAL_TABLE = {
    ("dog_bark", "doorbell"): "someone likely arrived at the door",
    ("siren", "car_horn"): "an emergency vehicle may be approaching in traffic",
    ("dog_bark", "bicycle_bell"): "a dog may have chased the cyclist or is reacting to them",
    ("dog_bark", "footstep"): "a dog is reacting to someone walking nearby",
    ("engine", "siren"): "a vehicle engine and siren indicate emergency traffic response",
    ("engine", "shouting"): "construction or heavy machinery operation may cause workers to call out",
    ("shouting", "siren"): "a siren triggers community response or alert shouting",
    ("microwave_beep", "water"): "someone started a timer then went to the kitchen tap or sink",
    ("dish", "water"): "dishwashing often involves both plate sounds and running water",
    ("dish", "footstep"): "walking while carrying dishes or kitchen activity is ongoing",
    ("keyboard", "door_open"): "typing activity continues as someone enters the office",
    ("printer", "footstep"): "footstep activity near the office printer during operation",
    ("printer", "keyboard"): "office printing and keyboard input occur together during work",
    ("drilling", "engine"): "construction drilling and engine activity at a work site",
    ("drilling", "shouting"): "loud drilling prompts workers to shout communication",
    ("hammer", "engine"): "hammering and engine sounds indicate active construction site work",
    ("door_open", "keyboard"): "someone enters the office and begins typing",
    ("phone_ring", "keyboard"): "a call comes in during active computer use",
    ("bird_chirp", "footstep"): "birds active while someone walks through a park scene",
    ("wind_rustle", "bird_chirp"): "windy conditions affect bird activity in the park",
    ("bicycle_bell", "footstep"): "a cyclist and pedestrian share a park or street path",
    ("car_horn", "engine"): "city traffic with car horns and engines on a street scene",
    ("car_horn", "footstep"): "city street activity combines traffic and pedestrian sounds",
    ("siren", "engine"): "emergency response with siren and accompanying engine",
    ("siren", "shouting"): "emergency siren triggers shouting responses",
    ("water", "microwave_beep"): "kitchen water use during microwave preparation",
    ("dish", "microwave_beep"): "kitchen dish use near microwave operation",
    ("keyboard", "footstep"): "office typing while someone walks by",
    ("hammer", "footstep"): "construction hammering with foot traffic nearby",
    ("door_slam", "keyboard"): "a door slamming while typing continues in the office",
    ("door_slam", "printer"): "door slam near active office printer",
}


def load_scenes():
    scenes = []
    for f in sorted(os.listdir("data/annotations")):
        if f.endswith(".json"):
            scenes.append(json.load(open(os.path.join("data/annotations", f))))
    return scenes


def generate_qa(scenes):
    pairs = []
    random.seed(42)
    for meta in scenes:
        sid = meta["scene_id"]
        events = meta.get("events", [])
        scenario = meta.get("scenario", "unknown")
        if not events:
            continue

        # Perceptual ×2
        pairs.append({"id": f"{sid}_p1", "scene_id": sid, "audio_path": meta["audio_path"],
                      "question": "What sound is present in this audio?",
                      "answer": events[0]["label"], "question_type": "perceptual",
                      "supporting_events": [f"{events[0]['label']}@{events[0]['start']}"]})
        pairs.append({"id": f"{sid}_p2", "scene_id": sid, "audio_path": meta["audio_path"],
                      "question": "What kind of environment does this audio suggest?",
                      "answer": scenario, "question_type": "perceptual",
                      "supporting_events": [f"scenario={scenario}"]})

        # Counting — count occurrences of first event label
        label = events[0]["label"]
        count = sum(1 for e in events if e["label"] == label)
        pairs.append({"id": f"{sid}_c1", "scene_id": sid, "audio_path": meta["audio_path"],
                      "question": f"How many times does a {label} occur?",
                      "answer": str(count), "question_type": "counting",
                      "supporting_events": [f"{e['label']}@{e['start']}" for e in events if e["label"] == label]})

        # Temporal — what happens right after the first event
        pairs.append({"id": f"{sid}_t1", "scene_id": sid, "audio_path": meta["audio_path"],
                      "question": f"What happens right after the {label}?",
                      "answer": events[1]["label"] if len(events) > 1 else "none",
                      "question_type": "temporal",
                      "supporting_events": [f"{events[0]['label']}@{events[0]['start']}"]})

        # Negation — absent event from scenario pool or vocab not in scene
        absent_label = "microwave_beep" if meta["scenario"] == "street" else "siren"
        if absent_label not in [e["label"] for e in events]:
            answer_val = "no"
        else:
            answer_val = "yes"
        pairs.append({"id": f"{sid}_n1", "scene_id": sid, "audio_path": meta["audio_path"],
                      "question": f"Is there a {absent_label} in this audio?",
                      "answer": answer_val, "question_type": "negation",
                      "supporting_events": []})

        # Comparative — requires >=2 distinct labels
        distinct_labels = []
        for e in events:
            if e["label"] not in distinct_labels:
                distinct_labels.append(e["label"])
        if len(distinct_labels) >= 2:
            l_a, l_b = distinct_labels[0], distinct_labels[1]
            cnt_a = sum(1 for e in events if e["label"] == l_a)
            cnt_b = sum(1 for e in events if e["label"] == l_b)
            if cnt_a > cnt_b:
                comp_answer = l_a
            elif cnt_b > cnt_a:
                comp_answer = l_b
            else:
                comp_answer = "equal"
            pairs.append({"id": f"{sid}_comp1", "scene_id": sid, "audio_path": meta["audio_path"],
                          "question": f"Which happened more often, {l_a} or {l_b}?",
                          "answer": comp_answer, "question_type": "comparative",
                          "supporting_events": [f"{l_a}@{c}" for e in events for c in [e['start']] if e['label'] == l_a][:2]})

        # Causal — check if any CAUSAL_TABLE pair is co-occurring in the scene
        labels_present = {e["label"] for e in events}
        causal_question_added = False
        for (a, b), explanation in CAUSAL_TABLE.items():
            if a in labels_present and b in labels_present:
                # Generate causal question using this co-occurrence
                pairs.append({"id": f"{sid}_cause1", "scene_id": sid, "audio_path": meta["audio_path"],
                              "question": f"Why might the {a} occur with the {b}?",
                              "answer": explanation, "question_type": "causal",
                              "supporting_events": [f"{a}@{e['start']}" for e in events if e['label'] == a][:1] +
                                                   [f"{b}@{e['start']}" for e in events if e['label'] == b][:1]})
                causal_question_added = True
                break  # one causal question per scene
    return pairs


def make_splits(scenes, seed=42):
    """Deterministic scene-level 70/15/15 split with no leakage (disjoint scene IDs)."""
    random.Random(seed).shuffle(scenes)
    n = len(scenes)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train = scenes[:n_train]
    val = scenes[n_train:n_train + n_val]
    test = scenes[n_train + n_val:]
    return train, val, test


if __name__ == "__main__":
    scenes = load_scenes()
    train_scenes, val_scenes, test_scenes = make_splits(scenes, seed=42)

    train_pairs = generate_qa(train_scenes)
    val_pairs = generate_qa(val_scenes)
    test_pairs = generate_qa(test_scenes)

    for split_name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        with open(f"data/qa_pairs/{split_name}.jsonl", "w") as out:
            for p in pairs:
                out.write(json.dumps(p) + "\n")

    total = len(train_pairs) + len(val_pairs) + len(test_pairs)
    by_type = {}
    for p in train_pairs + val_pairs + test_pairs:
        by_type[p["question_type"]] = by_type.get(p["question_type"], 0) + 1

    print(f"Phase 2 QA generated: {total} pairs from {len(scenes)} scenes.")
    print(f"  Train: {len(train_pairs)} pairs from {len(train_scenes)} scenes")
    print(f"  Val:   {len(val_pairs)} pairs from {len(val_scenes)} scenes")
    print(f"  Test:  {len(test_pairs)} pairs from {len(test_scenes)} scenes")
    for t, c in sorted(by_type.items()):
        print(f"  {t}: {c} total")
    print(f"  Train split: {len(train_scenes)}, Val split: {len(val_scenes)}, Test split: {len(test_scenes)} (70/15/15, no leakage)")
    # Print causal counts per split
    for split_name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        causal = sum(1 for p in pairs if p["question_type"] == "causal")
        counting = sum(1 for p in pairs if p["question_type"] == "counting")
        print(f"  {split_name}: causal={causal}, counting={counting}, ratio={causal/max(1,counting):.2%}")
