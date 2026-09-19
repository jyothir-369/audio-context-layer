#!/usr/bin/env python3
"""Phase 3 — Scene graph / structured context builder (Plan Section 3.2)."""
import json, os

def build_context(timeline_dict, annotation_path=None):
    events = timeline_dict.get("timeline", [])
    # Sort chronologically
    events = sorted(events, key=lambda e: e.get("start", 0))
    lines = ["Detected events (chronological):"]
    for i, evt in enumerate(events, 1):
        lines.append(f"{i}. {evt['label']} at {evt['start']:.1f}–{evt['end']:.1f}s (conf={evt.get('confidence', '?')})")
    scenario = "unknown"
    if annotation_path and os.path.isfile(annotation_path):
        try:
            with open(annotation_path) as f:
                ann = json.load(f)
            scenario = ann.get("scenario", "unknown")
        except Exception:
            pass
    lines.append(f"Scene type (annotated): {scenario}")
    return "\n".join(lines)
