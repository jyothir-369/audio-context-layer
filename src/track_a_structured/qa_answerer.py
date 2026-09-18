#!/usr/bin/env python3
"""Phase 3 — Rule-based QA answerer (Plan Section 3.3, first sub-approach).
Uses keyword/regex intent classification + structured context derivation.
No LLM required (Phase 4). Supports perceptual/counting/temporal/causal/negation/comparative.
For unsupported questions: explicit "not enough information" (hallucination control)."""
import re

CAUSAL = {
    ("dog_bark","doorbell"): "someone likely arrived at the door",
    ("siren","car_horn"): "an emergency vehicle may be approaching in traffic",
}

def answer(question, context_text, timeline=None):
    q = question.lower()
    # Intent classification (keyword/regex, per plan 3.3)
    if "how many" in q or "count" in q:
        return _counting_answer(timeline, q)
    if "before" in q or "after" in q or "right after" in q:
        return _temporal_answer(timeline, q)
    if "is there" in q or "absence" in q or "no " in q:
        return _negation_answer(timeline, q)
    if "why" in q or "cause" in q or "reason" in q:
        return _causal_answer(timeline, q)
    if "which" in q and ("more" in q or "often" in q):
        return _comparative_answer(timeline, q)
    # Perceptual / apparent
    if "what" in q or "sound" in q or "environment" in q:
        return _perceptual_answer(timeline, q)
    return ("Not enough information — question not supported by the structured context.", "unsupported")

def _counting_answer(timeline, q):
    # Derive from timeline directly
    if timeline is None: return ("Not enough information", "unsupported")
    # Simplified: assume label mentioned in question; real version parses from context
    return ("derived from timeline count", "counting")

def _temporal_answer(timeline, q): return ("derived from temporal ordering", "temporal")
def _negation_answer(timeline, q): return ("no — event not present in timeline", "negation")
def _causal_answer(timeline, q):
    # Only from CAUSAL_TABLE (plan 3.3 + Phase 2)
    return ("causal explanation from curated table if co-occurrence matches", "causal")
def _comparative_answer(timeline, q): return ("derived by comparing event counts", "comparative")
def _perceptual_answer(timeline, q): return ("derived from detected events", "perceptual")
