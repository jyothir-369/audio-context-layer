#!/usr/bin/env python3
"""Phase 3 — Scene graph / structured context builder (Plan Section 3.2)."""

def build_context(timeline_dict):
    events = timeline_dict.get("timeline", [])
    # Sort chronologically
    events = sorted(events, key=lambda e: e.get("start", 0))
    lines = ["Detected events (chronological):"]
    for i, evt in enumerate(events, 1):
        lines.append(f"{i}. {evt['label']} at {evt['start']:.1f}–{evt['end']:.1f}s")
    # Scene type inference not done by tagger; kept deterministic / empty for PoC honesty
    lines.append("Scene type (inferred): unknown (requires scenario metadata from Phase 1 timeline)")
    return "\n".join(lines)
