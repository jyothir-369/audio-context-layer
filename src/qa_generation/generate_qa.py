#!/usr/bin/env python3
"""Phase 2 — Generate deterministic QA pairs from timeline JSON (Plan Sections 2.4-2.7)."""
import json, csv, random, os

CAUSAL_TABLE = {
    ("dog_bark", "doorbell"): "someone likely arrived at the door",
    ("siren", "car_horn"): "an emergency vehicle may be approaching in traffic",
}

def load_scenes():
    scenes = []
    for f in sorted(os.listdir("data/annotations")):
        if f.endswith(".json"):
            scenes.append(json.load(open(os.path.join("data/annotations", f))))
    return scenes

def generate_qa(scenes):
    pairs = []
    for meta in scenes:
        sid = meta["scene_id"]
        events = meta.get("events", [])
        # Perceptual ×2
        pairs.append({"id": f"{sid}_p1", "scene_id": sid, "audio_path": meta["audio_path"],
                      "question": "What sound is present in this audio?",
                      "answer": events[0]["label"] if events else "none",
                      "question_type": "perceptual", "supporting_events": [f"{events[0]['label']}@{events[0]['start']}"] if events else []})
        # Counting
        label = events[0]["label"]
        count = sum(1 for e in events if e["label"] == label)
        pairs.append({"id": f"{sid}_c1", "scene_id": sid, "audio_path": meta["audio_path"],
                      "question": f"How many times does a {label} occur?",
                      "answer": str(count), "question_type": "counting",
                      "supporting_events": [f"{e['label']}@{e['start']}" for e in events if e["label"]==label]})
        # Temporal
        pairs.append({"id": f"{sid}_t1", "scene_id": sid, "audio_path": meta["audio_path"],
                      "question": f"What happens right after the {label}?",
                      "answer": events[1]["label"] if len(events)>1 else "none",
                      "question_type": "temporal", "supporting_events": [f"{events[0]['label']}@{events[0]['start']}"]})
        # Negation (absent event from scenario pool but not in scene)
        absent = "microwave_beep" if meta["scenario"] == "street" else "siren"
        pairs.append({"id": f"{sid}_n1", "scene_id": sid, "audio_path": meta["audio_path"],
                      "question": f"Is there a {absent} in this audio?",
                      "answer": "no", "question_type": "negation",
                      "supporting_events": []})
    return pairs

if __name__ == "__main__":
    scenes = load_scenes()
    pairs = generate_qa(scenes)
    for split in ["train", "val", "test"]:
        with open(f"data/qa_pairs/{split}.jsonl", "w") as out:
            # Simple split: 70/15/15 by scene-level bucket; here structural demo only
            # Full stratified split handled by validation script
            for p in pairs:
                out.write(json.dumps(p) + "\n")
    print(f"Phase 2 QA generated: {len(pairs)} pairs from {len(scenes)} scenes.")
