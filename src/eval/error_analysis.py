#!/usr/bin/env python3
"""Phase 5 — Real error analysis with IoU-based perception/reasoning split."""
import json, os, sys
sys.path.insert(0, "src/track_a_structured")
from event_tagger import tag_audio

IOU_THRESH = 0.3


def iou(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def best_match(ge, timeline):
    """Return (best_iou, best_label_or_None) for a GT event vs timeline."""
    best = 0.0
    best_label = None
    for ev in timeline:
        sc = iou(ge["start"], ge["end"], ev["start"], ev["end"])
        if sc > best:
            best = sc
            best_label = ev["label"]
    return best, best_label


def classify_errors():
    test_items = []
    with open("data/qa_pairs/test.jsonl") as f:
        for line in f:
            test_items.append(json.loads(line))
    test_by_id = {it["id"]: it for it in test_items}

    with open("results/metrics_track_a.json") as f:
        preds_data = json.load(f)
    per_type_breakdown = preds_data["track_a_evaluation"]["per_type_breakdown"]
    rule_based = per_type_breakdown.get("rule_based", per_type_breakdown)

    perception_failures = 0
    reasoning_failures = 0
    unanswerable_failures = 0  # environment questions unanswerable by design (docs Item 3)
    total_errors = 0
    examples = []
    tag_cache = {}

    def get_tag_result(scene_id, audio_path):
        if scene_id not in tag_cache:
            try:
                tag_cache[scene_id] = tag_audio(audio_path)
            except Exception as e:
                tag_cache[scene_id] = {"timeline": [], "note": f"tagger error: {e}"}
        return tag_cache[scene_id]

    for qtype, qdata in rule_based.items():
        for pred in qdata["predictions"]:
            gt = pred.get("ground_truth", "").strip().lower()
            answer_pred = pred.get("answer_pred", "").strip().lower()
            if gt == answer_pred:
                continue  # correct, skip
            total_errors += 1
            scene_id = pred["scene_id"]

            ann_path = f"data/annotations/{scene_id}.json"
            gt_events = []
            if os.path.isfile(ann_path):
                with open(ann_path) as af:
                    ann = json.load(af)
                    gt_events = ann.get("events", [])

            audio_path = f"data/synthesized_audio/{scene_id}.wav"
            try:
                tagger_result = get_tag_result(scene_id, audio_path)
            except Exception as e:
                tagger_result = {"timeline": [], "note": f"tagger error: {e}"}
            timeline = tagger_result.get("timeline", [])

            # Determine if environment question (architecturally unanswerable by design — docs Item 3 / tech_report §8)
            question = test_by_id.get(pred.get("id"), {}).get("question", "")
            q_lower = question.lower()
            is_env_question = ("environment" in q_lower or ("what kind of" in q_lower and "scenario" not in q_lower))

            # Perception = missing/mislabeled GT event in timeline (genuine event-detection failure)
            # Unanswerable = environment inference from label co-occurrence (documented ~50% accuracy; not event-detection)
            is_perception = False
            for ge in gt_events:
                best_sc, best_lbl = best_match(ge, timeline)
                if best_sc < IOU_THRESH or best_lbl != ge["label"]:
                    is_perception = True
                    break

            if is_env_question:
                # Architecture gap per docs/tech_report §8: environment inference uses heuristic label-set mapping
                # Report separately so the perception/reasoning split is not inflated by this already-accepted limitation
                unanswerable_failures += 1
                diag = "unanswerable (by design)"
                detail = "Environment inference unanswerable from audio alone (heuristic label-set mapping; docs §8)"
                split_counts["unanswerable"] += 1
            elif is_perception:
                perception_failures += 1
                diag = "perception"
                detail = "GT event missing or mislabeled in tagger timeline"
                split_counts["perception"] += 1
            else:
                reasoning_failures += 1
                diag = "reasoning"
                detail = "All GT events detected correctly; derived answer wrong"
                split_counts["reasoning"] += 1

            if len(examples) < 13:  # collect 5 non-perceptual + enough for all types
                question = test_by_id.get(pred.get("id"), {}).get("question", "N/A")
                gt_timeline = [{"label": e["label"], "start": e["start"], "end": e["end"]} for e in gt_events]
                # Force diversity: collect at most 2 perceptual, include counting/temporal/causal/comparative
                type_key = f"{diag}:{qtype}:{scene_id}"
                examples.append({
                    "scene_id": scene_id,
                    "qtype": qtype,
                    "question": question,
                    "ground_truth_answer": gt,
                    "predicted_answer": answer_pred,
                    "gt_timeline": gt_timeline,
                    "predicted_timeline": timeline,
                    "diagnosis": diag,
                    "detail": detail,
                    "type_key": type_key,
                })

    # Per-question-type split (Option b: 3-category reporting so the split is not inflated)
    by_qtype = {}
    for qtype in ["perceptual", "counting", "temporal", "negation", "comparative", "causal"]:
        by_qtype[qtype] = {"perception": 0, "reasoning": 0, "unanswerable": 0, "total_errors": 0}
    # Track which errors were which kind per question type
    # (We accumulate during loop above; easiest is to re-derive from examples + totals)
    # Instead, we report aggregate counts per category with per-qtype explanation.

    # Instead, do proper second pass specifically for per-qtype reporting
    for qtype, qdata in rule_based.items():
        count = {"perception": 0, "reasoning": 0, "unanswerable": 0}
        for pred in qdata["predictions"]:
            gt = pred.get("ground_truth", "").strip().lower()
            ans = pred.get("answer_pred", "").strip().lower()
            if gt == ans: continue
            sid = pred["scene_id"]
            q = test_by_id.get(pred.get("id"),{}).get("question","")
            is_env = ("environment" in q.lower() or ("what kind of" in q.lower() and "scenario" not in q.lower()))
            if is_env:
                count["unanswerable"] += 1
            else:
                # Check timeline for this scene (fast since tag_result already computed above for examples; for all, recompute quickly)
                ann_path = f"data/annotations/{sid}.json"
                gt_events = []
                if os.path.isfile(ann_path):
                    with open(ann_path) as af: gt_events = json.load(af).get("events",[])
                try:
                    tr = get_tag_result(sid, f"data/synthesized_audio/{sid}.wav")
                    tl = tr.get("timeline",[])
                except Exception:
                    tl = []
                is_perc = False
                for ge in gt_events:
                    best_sc, best_lbl = best_match(ge, tl)
                    if best_sc < IOU_THRESH or best_lbl != ge["label"]:
                        is_perc = True; break
                if is_perc:
                    count["perception"] += 1
                else:
                    count["reasoning"] += 1
        by_qtype[qtype] = count
    pct = perception_failures / total_errors if total_errors else 0
    rct = reasoning_failures / total_errors if total_errors else 0
    unct = unanswerable_failures / total_errors if total_errors else 0

    # Force 5 non-perceptual examples (counting, temporal, comparative, causal, reasoning/unanswerable)
    selected = []
    seen_types = set()
    for ex in examples:
        if ex["diagnosis"] != "perception" and len(selected) < 5:
            selected.append(ex)
            seen_types.add(ex["qtype"])
    # If still under 5, find first wrong of missing types from predictions
    missing = ["counting", "temporal", "comparative", "causal", "negation"]
    for miss in missing:
        if miss in seen_types: continue
        for pred in rule_based.get(miss, {}).get("predictions", []):
            gt = pred.get("ground_truth","").strip().lower()
            ans = pred.get("answer_pred","").strip().lower()
            if gt == ans: continue
            sid = pred["scene_id"]
            ann_path = f"data/annotations/{sid}.json"
            gt_events = []
            if os.path.isfile(ann_path):
                with open(ann_path) as af: gt_events = json.load(af).get("events",[])
            try:
                tl = get_tag_result(sid, f"data/synthesized_audio/{sid}.wav").get("timeline",[])
            except Exception:
                tl = []
            q = test_by_id.get(pred.get("id"),{}).get("question","")
            is_env = ("environment" in q.lower() or ("what kind of" in q.lower() and "scenario" not in q.lower()))
            is_perc = False
            for ge in gt_events:
                best_sc, best_lbl = best_match(ge, tl)
                if best_sc < IOU_THRESH or best_lbl != ge["label"]:
                    is_perc = True; break
            diag = "unanswerable (by design)" if is_env else ("perception" if is_perc else "reasoning")
            detail = ("Environment inference unanswerable from audio alone" if is_env else ("GT event missing or mislabeled" if is_perc else "All GT events detected correctly; derived answer wrong"))
            selected.append({
                "scene_id": sid,
                "qtype": miss,
                "question": q,
                "ground_truth_answer": gt,
                "predicted_answer": ans,
                "gt_timeline": [{"label":e["label"],"start":e["start"],"end":e["end"]} for e in gt_events],
                "predicted_timeline": tl,
                "diagnosis": diag,
                "detail": detail,
                "type_key": f"{diag}:{miss}:{sid}",
            })
            break

    # Select 5 non-perceptual + up to 3 perception for total 8 diverse
    final_examples = selected[:5]
    # Add perception (max 2) for comparison
    perc_exs = [ex for ex in examples if ex["diagnosis"] == "perception"]
    final_examples.extend(perc_exs[:2])
    # Fill to 8 with any remaining diverse
    for ex in examples:
        if len(final_examples) >= 8: break
        if ex not in final_examples:
            final_examples.append(ex)

    report = f"""# Phase 5 — Error Analysis (Plan Section 6)

## Perception vs Reasoning — 3-Category Split (Plan 6.2, Option b; not inflated by unanswerable gap)

**Method:** Perception = GT event missing/mislabeled in tagger timeline (IoU < {IOU_THRESH}). Reasoning = event detected correctly but answer derived wrong. **Unanswerable (by design)** = environment/scenario inference using heuristic label-set mapping (docs §8; ~50% accuracy already accepted). The three-way split reports {perception_failures} perception errors ({pct:.1%} of all errors), {reasoning_failures} reasoning errors ({rct:.1%}), and {unanswerable_failures} unanswerable errors ({unct:.1%}). Among the {perception_failures + reasoning_failures} perception/reasoning errors only, perception accounts for {(perception_failures / (perception_failures + reasoning_failures) if perception_failures + reasoning_failures else 0):.1%}.

**Total Errors: {total_errors}**
- Perception: {perception_failures} ({pct:.1%})
- Reasoning: {reasoning_failures} ({rct:.1%})
- Unanswerable (environment inference gap): {unanswerable_failures} ({unct:.1%})

### Per-Question-Type Breakdown (3 categories each)
"""
    for qtype in ["perceptual", "counting", "temporal", "negation", "comparative", "causal"]:
        c = by_qtype.get(qtype, {"perception":0,"reasoning":0,"unanswerable":0,"errors":0,"errors":0})
        # Use "errors" key consistently; earlier initialization uses "errors", but the loop above uses "errors" too (check both)
        # Fix: the initialization above creates {"perception":0,"reasoning":0,"unanswerable":0,"total_errors":0} — that uses "total_errors" not "errors".
        errors_key = c.get("errors", c.get("total_errors", 0))
        if errors_key > 0:
            report += f"- **{qtype}** (errors={errors_key}): perception={c['perception']}, reasoning={c['reasoning']}, unanswerable={c['unanswerable']}\n"

    report += f"""
**Perceptual "what sound" vs "what environment":** Perceptual errors include both event-detection misses (e.g., "door_slam" missed, -> "footstep") AND environment-inference failures (e.g., GT="kitchen" but predicted="office"). The unanswerable category captures the latter; perception and reasoning are reported separately so the split does not imply that environment inference is a genuine detection failure.

## Qualitative Failures (Plan 6.3) — 5+ non-perceptual + diverse (counting, temporal, comparative, causal, reasoning/unanswerable)

"""
    for ex in final_examples:
        report += f"### {ex['scene_id']} ({ex['qtype']}) — {ex['diagnosis']}\n"
        report += f"- Question: {ex['question']}\n"
        report += f"- Ground Truth: {ex['ground_truth_answer']}\n"
        report += f"- Predicted: {ex['predicted_answer']}\n"
        report += f"- GT Timeline: {json.dumps(ex['gt_timeline'])}\n"
        report += f"- Predicted Timeline: {json.dumps(ex['predicted_timeline'])}\n"
        report += f"- Diagnosis: {ex['diagnosis']} — {ex['detail']}\n\n"
    report += """## Systematic Patterns (Plan 6.4)

- Overlapping events: temporal-ordering accuracy is likely lower when events overlap (Plan 2.3 overlap_probability=0.35). This is a hypothesis, not a measured holdout result.
- Unseen class / scenario holdout: NOT implemented. Train, val, and test all contain the same five scenarios (street, kitchen, park, office, construction_site) and the same event vocabulary. Plan 2.6 suggested holding out classes; this PoC split does not.
- Negation false-positive rate: rule-based system explicitly returns 'no' when event absent from timeline.
- Causal explanations approximate; CAUSAL_TABLE covers only curated event pairs.
"""
    with open("results/error_analysis.md", "w") as f:
        f.write(report)

    return {"total_errors": total_errors, "perception_failures": perception_failures,
            "reasoning_failures": reasoning_failures, "perception_pct": pct,
            "reasoning_pct": rct, "examples": examples}


if __name__ == "__main__":
    r = classify_errors()
    print(f"Total errors: {r['total_errors']}")
    print(f"Perception failures: {r['perception_failures']} ({r['perception_pct']:.1%})")
    print(f"Reasoning failures: {r['reasoning_failures']} ({r['reasoning_pct']:.1%})")
    print(f"Report written to results/error_analysis.md")
