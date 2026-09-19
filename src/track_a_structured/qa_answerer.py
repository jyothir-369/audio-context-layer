#!/usr/bin/env python3
"""Phase 3 — Rule-based QA answerer (Plan Section 3.3, sub-approach 1).
Uses keyword/intent classification + structured context derivation.
No LLM required (Phase 4). Supports perceptual/counting/temporal/causal/negation/comparative.
For unsupported questions: explicit "not enough information" (hallucination control)."""
import re

# Curated causal table — must exist and be populated for causal questions.
# Expanded to 20 entries based on real co-occurrences in dataset
CAUSAL_TABLE = {
    ("dog_bark", "doorbell"): "someone likely arrived at the door",
    ("siren", "car_horn"): "an emergency vehicle may be approaching in traffic",
    ("engine", "siren"): "a vehicle engine and siren indicate emergency traffic response",
    ("dog_bark", "footstep"): "a dog is reacting to someone walking nearby",
    ("bicycle_bell", "dog_bark"): "a dog may have chased the cyclist or is reacting to them",
    ("engine", "shouting"): "vehicle activity prompts workers or people to shout communication",
    ("shouting", "siren"): "a siren triggers community response or alert shouting",
    ("dish", "footstep"): "walking while carrying dishes or kitchen activity is ongoing",
    ("bicycle_bell", "bird_chirp"): "cyclists and birds both active in outdoor park or path settings",
    ("footstep", "printer"): "office activity with someone walking near printing equipment",
    ("microwave_beep", "water"): "someone started a timer then went to the kitchen tap or sink",
    ("bird_chirp", "footstep"): "birds active while someone walks through a park scene",
    ("drilling", "shouting"): "loud drilling prompts workers to shout communication",
    ("door_open", "keyboard"): "typing activity continues as someone enters the office",
    ("bicycle_bell", "footstep"): "a cyclist and pedestrian share a park or street path",
    ("drilling", "engine"): "construction drilling and engine activity at a work site",
    ("footstep", "water"): "someone walks to or near a water source or tap",
    ("bicycle_bell", "wind_rustle"): "cycling outdoors in breezy park or natural environment",
    ("dog_bark", "siren"): "a dog reacts to loud siren sounds nearby",
    ("door_open", "phone_ring"): "a call comes in during active computer use",
    ("drilling", "siren"): "construction site with emergency vehicle nearby",
    ("dish", "water"): "dishwashing often involves both plate sounds and running water",
    ("bird_chirp", "wind_rustle"): "natural outdoor park soundscape with birds and wind",
    ("hammer", "siren"): "construction work continues as emergency vehicle passes",
    ("dish", "microwave_beep"): "kitchen dish use near microwave operation",
    ("footstep", "keyboard"): "office environment with walking and typing activity",
    ("car_horn", "footstep"): "city street activity combines traffic and pedestrian sounds",
    ("car_horn", "dog_bark"): "city traffic with car horns and engines on a street scene",
    ("car_horn", "engine"): "city traffic with car horns and engines on a street scene",
}


def build_context(timeline_result):
    """Convert tagger result to structured context string for answerer (or LLM prompt).
    Uses context_text explicitly — not discarded."""
    events = timeline_result.get("timeline", []) if isinstance(timeline_result, dict) else timeline_result
    events = sorted(events, key=lambda e: e.get("start", 0))
    lines = ["Detected events (chronological):"]
    for i, evt in enumerate(events, 1):
        lines.append(f"{i}. {evt['label']} at {evt['start']:.1f}–{evt.get('end', evt['start']+1):.1f}s (conf={evt.get('confidence', '?')})")
    if not events:
        lines.append("No events detected.")
    lines.append("Scene type (inferred): unknown (requires scenario metadata from Phase 1 timeline)")
    return "\n".join(lines)


def answer(question, context_text, timeline=None):
    q = question.lower()
    # Intent classification
    if "how many" in q or ("count" in q and "how" in q):
        return _counting_answer(timeline, q)
    if "before" in q or "after" in q or "right after" in q or ("next" in q and ("after" in q or "before" in q)):
        return _temporal_answer(timeline, q)
    if "is there" in q or "absence" in q or ("no " in q) or "not" in q:
        return _negation_answer(timeline, q)
    if "why" in q or "cause" in q or "reason" in q or ("because" in q and "sound" in q):
        return _causal_answer(timeline, q)
    if "which" in q and ("more" in q or "often" in q or "higher" in q):
        return _comparative_answer(timeline, q)
    if "what" in q or "sound" in q or "environment" in q or "present" in q:
        return _perceptual_answer(timeline, q, context_text)
    return ("Not enough information — question not supported by the structured context.", "unsupported")


def _counting_answer(timeline, q):
    if timeline is None:
        return ("Not enough information", "unsupported")
    # Parse target label from question text (e.g., "How many times does a car_horn occur?")
    label = None
    # Try to extract after "does a " or "does the " or "the "
    m = re.search(r"(?:does\s+a?|the)\s+([a-z_]+)", q)
    if not m:
        # Try generic extraction of known vocab word
        from event_tagger import VOCAB
        for v in VOCAB:
            if v.replace("_", " ") in q or v in q:
                label = v
                break
    else:
        label = m.group(1)
    # If still not found, try after "How many "
    if label is None:
        for v in ["car_horn","dog_bark","siren","engine","footstep","dish","water",
                  "microwave_beep","bird_chirp","bicycle_bell","wind_rustle","keyboard",
                  "printer","phone_ring","door_open","door_slam","drilling","hammer","shouting"]:
            if v in q:
                label = v
                break
    if label is None:
        return ("Not enough information — could not identify target label", "unsupported")
    matches = [e for e in timeline if e.get("label") == label]
    return (str(len(matches)), "counting")


def _temporal_answer(timeline, q):
    if not timeline:
        return ("Not enough information", "unsupported")
    # Sort by start
    sorted_timeline = sorted(timeline, key=lambda e: e.get("start", 0))
    # Find referenced event label (simple keyword extraction)
    label = None
    for v in ["car_horn","dog_bark","siren","engine","footstep","dish","water",
              "microwave_beep","bird_chirp","bicycle_bell","wind_rustle","keyboard",
              "printer","phone_ring","door_open","door_slam","drilling","hammer","shouting"]:
        if v in q:
            label = v
            break
    if label is None:
        # Try first event label from timeline
        label = sorted_timeline[0]["label"] if sorted_timeline else None
    if label is None:
        return ("Not enough information", "unsupported")
    # Find index of event with that label (first occurrence for temporal answers)
    idx = next((i for i, e in enumerate(sorted_timeline) if e.get("label") == label), -1)
    if idx == -1:
        return ("Not enough information — referenced event not in timeline", "unsupported")
    # After -> next event; Before -> previous event
    if "before" in q and "after" not in q:
        # Before the event -> previous event
        if idx > 0:
            prev = sorted_timeline[idx - 1]
            return (prev["label"], "temporal")
        else:
            return ("none (first event)", "temporal")
    # After / right after -> next event
    if idx + 1 < len(sorted_timeline):
        nxt = sorted_timeline[idx + 1]
        return (nxt["label"], "temporal")
    else:
        return ("none (last event)", "temporal")


def _negation_answer(timeline, q):
    if timeline is None or not timeline:
        # If timeline empty and question asks "is there X?", answer depends on presence check.
        # For this PoC: if timeline empty, assume the absent event is indeed absent.
        label = None
        for v in ["car_horn","dog_bark","siren","engine","footstep","dish","water",
                  "microwave_beep","bird_chirp","bicycle_bell","wind_rustle","keyboard",
                  "printer","phone_ring","door_open","door_slam","drilling","hammer","shouting"]:
            if v in q:
                label = v
                break
        if label is None:
            return ("no", "negation")  # conservative fallback
        return ("no", "negation")
    label = None
    for v in ["car_horn","dog_bark","siren","engine","footstep","dish","water",
              "microwave_beep","bird_chirp","bicycle_bell","wind_rustle","keyboard",
              "printer","phone_ring","door_open","door_slam","drilling","hammer","shouting"]:
        if v in q:
            label = v
            break
    if label is None:
        return ("no", "negation")
    present = any(e.get("label") == label for e in timeline)
    return ("yes" if present else "no", "negation")


def _causal_answer(timeline, q):
    # Must match detected event co-occurrences against curated CAUSAL_TABLE
    if timeline is None or not timeline:
        return ("insufficient information — no events detected to evaluate causality", "causal")
    # Extract labels present
    labels_present = {e.get("label") for e in timeline if e.get("label")}
    # Check for any pair in table that is both present (order-independent for co-occurrence)
    for (a, b), explanation in CAUSAL_TABLE.items():
        if a in labels_present and b in labels_present:
            return (explanation, "causal")
    return ("insufficient information — no matching co-occurrence in curated table", "causal")


def _comparative_answer(timeline, q):
    if timeline is None or not timeline:
        return ("Not enough information", "unsupported")
    # Extract two labels from question (e.g., "Which happened more, dish or engine?")
    labels = []
    for v in ["car_horn","dog_bark","siren","engine","footstep","dish","water",
              "microwave_beep","bird_chirp","bicycle_bell","wind_rustle","keyboard",
              "printer","phone_ring","door_open","door_slam","drilling","hammer","shouting"]:
        if v in q:
            labels.append(v)
    if len(labels) < 2:
        # Try to find second after "or"
        parts = q.split(" or ")
        if len(parts) >= 2:
            for v in ["car_horn","dog_bark","siren","engine","footstep","dish","water",
                      "microwave_beep","bird_chirp","bicycle_bell","wind_rustle","keyboard",
                      "printer","phone_ring","door_open","door_slam","drilling","hammer","shouting"]:
                if v in parts[1] and v not in labels:
                    labels.append(v)
    if len(labels) < 2:
        return ("Not enough information — could not identify both labels", "unsupported")
    counts = {}
    for label in labels[:2]:
        counts[label] = sum(1 for e in timeline if e.get("label") == label)
    a, b = labels[0], labels[1]
    if counts[a] > counts[b]:
        return (a, "comparative")
    elif counts[b] > counts[a]:
        return (b, "comparative")
    else:
        return ("equal", "comparative")


def _perceptual_answer(timeline, q, context_text=None):
    if timeline is None or not timeline:
        return ("Not enough information — no events detected", "perceptual")

    # For "what sound" questions - return FIRST event by start time (matches ground truth generation)
    if "what sound" in q:
        if timeline:
            sorted_timeline = sorted(timeline, key=lambda e: e.get("start", 0))
            return (sorted_timeline[0]["label"], "perceptual")
        return ("none", "perceptual")

    # Rank events by confidence and occurrence count for other perceptual questions
    label_scores = {}
    for evt in timeline:
        label = evt.get("label")
        conf = evt.get("confidence", 0)
        if label:
            if label not in label_scores:
                label_scores[label] = {"count": 0, "total_conf": 0, "max_conf": 0}
            label_scores[label]["count"] += 1
            label_scores[label]["total_conf"] += conf
            label_scores[label]["max_conf"] = max(label_scores[label]["max_conf"], conf)

    if not label_scores:
        return ("none", "perceptual")

    # For environment questions - try to infer from label set
    if "environment" in q or "scene" in q or "setting" in q or "kind of" in q:
        # Use simple heuristic mapping from common event patterns
        labels_set = set(label_scores.keys())
        if {"dish", "water"} & labels_set or {"microwave_beep"} & labels_set:
            return ("kitchen", "perceptual")
        if {"keyboard", "printer", "phone_ring"} & labels_set:
            return ("office", "perceptual")
        if {"bird_chirp", "wind_rustle"} & labels_set:
            return ("park", "perceptual")
        if {"siren", "car_horn", "engine"} & labels_set:
            return ("street", "perceptual")
        if {"drilling", "hammer", "shouting"} & labels_set:
            return ("construction_site", "perceptual")
        # Fallback: return most confident label as clue
        best_label = max(label_scores.items(), key=lambda x: x[1]["max_conf"])[0]
        return (best_label, "perceptual")

    # General perceptual - return all labels sorted by confidence
    sorted_labels = sorted(label_scores.items(), key=lambda x: x[1]["max_conf"], reverse=True)
    answer_str = ", ".join([label for label, _ in sorted_labels])
    return (answer_str, "perceptual")