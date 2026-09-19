#!/usr/bin/env python3
"""Phase 3 — Scene graph / structured context builder (Plan Section 3.2)."""
import json, os


def infer_scenario(events):
    """Infer scene type from detected labels (no ground-truth annotation leak)."""
    labels_set = {e.get("label") for e in events if e.get("label")}
    if {"dish", "water"} & labels_set or {"microwave_beep"} & labels_set:
        return "kitchen"
    if {"keyboard", "printer", "phone_ring"} & labels_set:
        return "office"
    if {"bird_chirp", "wind_rustle"} & labels_set:
        return "park"
    if {"siren", "car_horn", "engine"} & labels_set:
        return "street"
    if {"drilling", "hammer", "shouting"} & labels_set:
        return "construction_site"
    return "unknown"


def build_context(timeline_dict, annotation_path=None):
    events = timeline_dict.get("timeline", [])
    events = sorted(events, key=lambda e: e.get("start", 0))
    lines = ["Detected events (chronological):"]
    for i, evt in enumerate(events, 1):
        lines.append(f"{i}. {evt['label']} at {evt['start']:.1f}–{evt['end']:.1f}s (conf={evt.get('confidence', '?')})")
    scenario = infer_scenario(events)
    # Optional annotation path is accepted for callers, but scene type is always
    # inferred from the predicted timeline so ground-truth scenario is not leaked.
    if annotation_path and os.path.isfile(annotation_path):
        try:
            with open(annotation_path) as f:
                json.load(f)
        except Exception:
            pass
    lines.append(f"Scene type (inferred): {scenario}")
    return "\n".join(lines)
