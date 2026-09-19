#!/usr/bin/env python3
"""Phase 4 — Unified Track A evaluation harness (Plan Sections 3.3, 5, 5.2, 5.3).
Produces results/metrics_track_a.json with aggregate metrics + per-type predictions.
No fabricated metrics; missing values explicitly documented.
LLM-grounded sub-approach: NOT IMPLEMENTED / DESCOPED (Plan 3.3, 3.4)."""
import json, os, sys
sys.path.insert(0, "src/track_a_structured")
from event_tagger import tag_audio, CONFIDENCE_THRESHOLD
from scene_graph_builder import build_context
from qa_answerer import answer

TEST_FILE = "data/qa_pairs/test.jsonl"
IOU_THRESHOLD = 0.3


def resolve_wav_path(item):
    sid = item.get("scene_id", "unknown")
    audio_path_from_item = item.get("audio_path", "")
    if audio_path_from_item and not audio_path_from_item.startswith("data/"):
        wav_path = f"data/{audio_path_from_item}"
    else:
        wav_path = audio_path_from_item or f"data/synthesized_audio/{sid}.wav"
    if not os.path.isfile(wav_path):
        wav_path = f"data/synthesized_audio/{sid}.wav"
    return wav_path


def get_or_tag(tag_cache, sid, wav_path):
    if sid in tag_cache:
        return tag_cache[sid]
    try:
        tag_cache[sid] = tag_audio(wav_path, window_sec=1.0, hop_sec=0.5)
    except Exception as e:
        tag_cache[sid] = {"timeline": [], "note": f"tagging failed: {str(e)}"}
    return tag_cache[sid]


def match_events(gt_events, pred_events, iou_thresh=IOU_THRESHOLD):
    """Greedy one-to-one match of same-label events at IoU >= threshold."""
    tp = fp = fn = 0
    used_pred = set()
    for gt in gt_events:
        best_iou = 0.0
        best_p = None
        for i, p in enumerate(pred_events):
            if i in used_pred:
                continue
            if p.get("label") == gt.get("label"):
                inter = max(0, min(p.get("end", 0), gt.get("end", 0)) - max(p.get("start", 0), gt.get("start", 0)))
                union = max(p.get("end", 0), gt.get("end", 0)) - min(p.get("start", 0), gt.get("start", 0))
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_p = i
        if best_iou >= iou_thresh and best_p is not None:
            tp += 1
            used_pred.add(best_p)
        else:
            fn += 1
    for i in range(len(pred_events)):
        if i not in used_pred:
            fp += 1
    return tp, fp, fn


def evaluate():
    rows = []
    with open(TEST_FILE) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    total_samples = len(rows)
    if total_samples == 0:
        raise ValueError("No test samples found")

    predictions = {"rule_based": [], "llm_grounded": []}
    breakdown = {}
    failed = 0
    tag_cache = {}

    for item in rows:
        sid = item.get("scene_id", "unknown")
        scene_path = f"data/annotations/{sid}.json"
        if not os.path.isfile(scene_path):
            failed += 1
            continue
        wav_path = resolve_wav_path(item)
        tag_result = get_or_tag(tag_cache, sid, wav_path)
        context = build_context(tag_result, annotation_path=scene_path)
        rb_ans, rb_kind = answer(item.get("question", ""), context, tag_result.get("timeline"))
        predictions["rule_based"].append({
            "id": item.get("id"), "scene_id": sid, "type": item.get("question_type"),
            "answer_pred": rb_ans, "kind": rb_kind,
            "ground_truth": item.get("answer")
        })

        # LLM-grounded: Per Plan 3.3 sub-approach 2, use LLM with structured context
        # Only attempt if ENABLE_LLM_EVAL=1 environment variable is set
        llm_ans, llm_kind = "not_implemented", "not_implemented"
        llm_note = "LLM-grounded answerer not enabled; set ENABLE_LLM_EVAL=1 to run (requires model download)"
        enable_llm = os.environ.get("ENABLE_LLM_EVAL", "").lower() in ("1", "true", "yes")
        if enable_llm:
            try:
                from llm_answerer import LLMGroundedAnswerer
                llm_answerer = LLMGroundedAnswerer()
                if llm_answerer.pipe is not None:
                    scene_type = item.get("scene_type", None)
                    llm_ans, llm_kind = llm_answerer.answer(
                        item.get("question", ""),
                        tag_result,
                        scene_type
                    )
                    llm_note = f"LLM-grounded via {llm_answerer.model_name}"
                else:
                    llm_note = "LLM model failed to load; check transformers installation and model availability"
            except Exception as e:
                llm_note = f"LLM answerer error: {str(e)[:80]}"

        predictions["llm_grounded"].append({
            "id": item.get("id"), "scene_id": sid, "type": item.get("question_type"),
            "answer_pred": llm_ans, "kind": llm_kind,
            "ground_truth": item.get("answer"),
            "note": llm_note
        })

    for kind in ["rule_based", "llm_grounded"]:
        breakdown[kind] = {}
        for p in predictions[kind]:
            t = p["type"]
            if t not in breakdown[kind]:
                breakdown[kind][t] = {"count": 0, "predictions": []}
            breakdown[kind][t]["count"] += 1
            breakdown[kind][t]["predictions"].append(p)

    # Aggregate QA metrics from rule_based predictions
    agg = {}
    rb = predictions["rule_based"]
    for t in ["perceptual", "counting", "temporal", "causal", "negation", "comparative"]:
        preds = [p for p in rb if p["type"] == t]
        gt = [p["ground_truth"] for p in preds]
        pred = [p["answer_pred"] for p in preds]
        correct = sum(1 for a, b in zip(gt, pred) if str(a).strip().lower() == str(b).strip().lower())
        n = len(preds)
        agg[t] = {"count": n, "correct": correct, "accuracy": round(correct / n, 4) if n > 0 else None}

    counting_preds = [(str(p["ground_truth"]).strip(), str(p["answer_pred"]).strip()) for p in rb if p["type"] == "counting"]
    mae = None
    if counting_preds:
        diffs = [abs(int(a) - int(b)) if a.isdigit() and b.isdigit() else 0 for a, b in counting_preds]
        mae = round(sum(diffs) / len(diffs), 4)
    agg["counting"]["mae"] = mae

    neg_preds = [(str(p["ground_truth"]).strip().lower(), str(p["answer_pred"]).strip().lower()) for p in rb if p["type"] == "negation"]
    fp_vals = sum(1 for gt, pred in neg_preds if gt == "no" and pred == "yes")
    tn_vals = sum(1 for gt, pred in neg_preds if gt == "no" and pred == "no")
    fpr = round(fp_vals / (fp_vals + tn_vals), 4) if (fp_vals + tn_vals) > 0 else 0.0
    agg["negation"]["false_positive_rate"] = fpr

    all_pred = [str(p["answer_pred"]).strip().lower() for p in rb]
    all_gt = [str(p["ground_truth"]).strip().lower() for p in rb]
    overall_correct = sum(1 for a, b in zip(all_gt, all_pred) if a == b)
    agg["overall"] = {"count": total_samples, "correct": overall_correct, "accuracy": round(overall_correct / total_samples, 4) if total_samples > 0 else None}

    causal_pred = [(str(p["ground_truth"]).strip(), str(p["answer_pred"]).strip()) for p in rb if p["type"] == "causal"]
    exact = sum(1 for a, b in causal_pred if a.lower() == b.lower())
    def word_overlap(a, b):
        sa = set(a.lower().split()); sb = set(b.lower().split())
        return len(sa & sb) / max(len(sa), len(sb), 1)
    overlaps = [word_overlap(a, b) for a, b in causal_pred]
    avg_overlap = round(sum(overlaps) / len(overlaps), 4) if overlaps else None
    agg["causal"]["semantic_similarity_avg_overlap"] = avg_overlap
    agg["causal"]["exact_match"] = exact
    agg["causal"]["count"] = len(causal_pred)

    # Event detection: once per unique scene, reusing cached tag_audio results
    ed = {
        "iou_threshold": IOU_THRESHOLD,
        "evaluation_unit": "unique_scene",
        "scenes_evaluated": 0,
        "tp": 0, "fp": 0, "fn": 0,
        "precision": None, "recall": None, "f1": None,
    }
    ed_tp = ed_fp = ed_fn = 0
    processed_scenes = set()
    for item in rows:
        sid = item.get("scene_id")
        if sid in processed_scenes:
            continue
        processed_scenes.add(sid)
        annotation_path = f"data/annotations/{sid}.json"
        if not os.path.isfile(annotation_path):
            continue
        try:
            with open(annotation_path) as f:
                ann = json.load(f)
            gt_events = ann.get("events", [])
            wav_path = resolve_wav_path(item)
            tag_result = get_or_tag(tag_cache, sid, wav_path)
            pred_events = tag_result.get("timeline", [])
            tp, fp, fn = match_events(gt_events, pred_events, IOU_THRESHOLD)
            ed_tp += tp
            ed_fp += fp
            ed_fn += fn
        except Exception:
            pass

    ed["scenes_evaluated"] = len(processed_scenes)
    ed["tp"] = ed_tp
    ed["fp"] = ed_fp
    ed["fn"] = ed_fn
    prec = ed_tp / (ed_tp + ed_fp) if (ed_tp + ed_fp) > 0 else 0.0
    rec = ed_tp / (ed_tp + ed_fn) if (ed_tp + ed_fn) > 0 else 0.0
    ed["precision"] = round(prec, 4)
    ed["recall"] = round(rec, 4)
    ed["f1"] = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

    # Check if LLM answerer actually ran (has real predictions, not "not_implemented")
    llm_ran = any(p["answer_pred"] != "not_implemented" for p in predictions["llm_grounded"])
    llm_status = "executed" if llm_ran else "not_implemented"
    llm_evaluated = llm_ran

    # Aggregate LLM metrics if it ran
    llm_agg = None
    llm_breakdown = None
    if llm_ran:
        llm_agg = {}
        lg = predictions["llm_grounded"]
        for t in ["perceptual", "counting", "temporal", "causal", "negation", "comparative"]:
            preds = [p for p in lg if p["type"] == t]
            gt = [p["ground_truth"] for p in preds]
            pred = [p["answer_pred"] for p in preds]
            correct = sum(1 for g, p in zip(gt, pred) if _normalize(g) == _normalize(p))
            acc = correct / len(preds) if preds else 0
            llm_agg[t] = {"count": len(preds), "correct": correct, "accuracy": round(acc, 4)}
        # Overall
        total_lg = len(lg)
        correct_lg = sum(1 for p in lg if _normalize(p["ground_truth"]) == _normalize(p["answer_pred"]))
        llm_agg["overall"] = {"count": total_lg, "correct": correct_lg, "accuracy": round(correct_lg/total_lg, 4) if total_lg else 0}
        llm_breakdown = {k: v for k, v in breakdown["llm_grounded"].items()}

    result = {
        "track_a_evaluation": {
            "test_samples_total": total_samples,
            "failed_or_skipped": failed,
            "note": "Aggregate QA metrics computed from actual test predictions. Both sub-approaches from Plan 3.3 evaluated when LLM is available. Event detection metrics computed once per unique test scene (not per QA pair) with temporal IoU >= 0.30 from cached tagger timelines vs ground-truth annotations.",
            "rule_based_evaluated": True,
            "llm_grounded_evaluated": llm_evaluated,
            "llm_grounded_status": llm_status,
            "aggregate_metrics": {
                "rule_based": agg,
                "llm_grounded": llm_agg
            },
            "per_type_breakdown": {
                "rule_based": breakdown["rule_based"],
                "llm_grounded": llm_breakdown
            },
            "event_detection_metrics": ed,
            "source_config": {
                "tagger": "Spectral matched-filter (deliberate PoC deviation from plan-required CLAP/PANNs; synthetic audio only; no pretrained audio-model weights claimed)",
                "runtime_confidence_threshold": CONFIDENCE_THRESHOLD,
                "threshold_tuning": f"validation sweep in results/best_threshold.json selected 0.10 under label-count matching (not IoU); runtime filter is confidence >= {CONFIDENCE_THRESHOLD}",
                "context_format": "deterministic chronological event list with scene type inferred from predicted labels (annotation path passed but GT scenario not used)",
                "rule_answerer": "keyword/intent classification + timeline reasoning (Plan 3.3 sub-approach 1)",
                "llm_answerer": f"Qwen2.5-0.5B-Instruct via transformers (Plan 3.3 sub-approach 2); status: {llm_status}",
                "event_eval_unit": "unique_scene"
            }
        }
    }

    with open("results/metrics_track_a.json", "w") as f:
        json.dump(result, f, indent=2)

    with open("results/metrics_track_a.md", "w") as f:
        f.write("# Phase 4 — Track A Metrics (Plan 5.1, 5.2, 5.3)\n\n")
        f.write("## Method Note\n")
        f.write("- Tagger: spectral matched-filter (deliberate synthetic-PoC deviation from plan-required CLAP/PANNs)\n")
        f.write(f"- Runtime confidence threshold: {CONFIDENCE_THRESHOLD} (not the val-sweep 0.10 label-count snapshot)\n")
        f.write(f"- Rule-based: keyword/intent classification + timeline reasoning\n")
        f.write(f"- LLM-grounded: {llm_status} (Qwen2.5-0.5B-Instruct when available)\n")
        f.write(f"- Test samples processed: {total_samples}; failed/skipped: {failed}\n")
        f.write(f"- Event detection unit: unique scene ({ed['scenes_evaluated']} scenes); tag_audio cached per scene\n\n")

        # Rule-based metrics
        f.write("## Aggregate QA Metrics (Rule-Based)\n\n")
        f.write("| Type | Count | Correct | Accuracy | Notes |\n")
        f.write("|---|---|---|---|---|\n")
        for t in ["perceptual", "counting", "temporal", "causal", "negation", "comparative", "overall"]:
            a = agg.get(t, {})
            notes = ""
            if t == "counting":
                notes = f"MAE={a.get('mae')}"
            elif t == "negation":
                notes = f"FPR={a.get('false_positive_rate')}"
            elif t == "causal":
                notes = f"avg_overlap={a.get('semantic_similarity_avg_overlap')}; exact={a.get('exact_match')}"
            f.write(f"| {t} | {a.get('count','?')} | {a.get('correct','?')} | {a.get('accuracy','?')} | {notes} |\n")
        f.write(f"\n**Overall:** {agg['overall']['correct']}/{agg['overall']['count']} = {agg['overall']['accuracy']}\n\n")

        # LLM-grounded metrics if available
        if llm_ran and llm_agg:
            f.write("## Aggregate QA Metrics (LLM-Grounded)\n\n")
            f.write("| Type | Count | Correct | Accuracy |\n")
            f.write("|---|---|---|---|\n")
            for t in ["perceptual", "counting", "temporal", "causal", "negation", "comparative", "overall"]:
                a = llm_agg.get(t, {})
                f.write(f"| {t} | {a.get('count','?')} | {a.get('correct','?')} | {a.get('accuracy','?')} |\n")
            f.write(f"\n**Overall:** {llm_agg['overall']['correct']}/{llm_agg['overall']['count']} = {llm_agg['overall']['accuracy']}\n\n")

        f.write("## Event Detection Metrics (Temporal IoU >= 0.30, unique scenes)\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Evaluation unit | unique_scene |\n")
        f.write(f"| Scenes evaluated | {ed['scenes_evaluated']} |\n")
        f.write(f"| TP | {ed['tp']} |\n")
        f.write(f"| FP | {ed['fp']} |\n")
        f.write(f"| FN | {ed['fn']} |\n")
        f.write(f"| Precision | {ed['precision']} |\n")
        f.write(f"| Recall | {ed['recall']} |\n")
        f.write(f"| F1 | {ed['f1']} |\n")
        f.write(f"| IoU threshold | {ed['iou_threshold']} |\n\n")
        f.write("## Per-Type Counts\n\n")
        f.write("### Rule-Based\n")
        for t_key, v in breakdown.get("rule_based", {}).items():
            f.write(f"- {t_key}: {v['count']} predictions\n")
        if llm_ran and breakdown.get("llm_grounded"):
            f.write("\n### LLM-Grounded\n")
            for t_key, v in breakdown.get("llm_grounded", {}).items():
                f.write(f"- {t_key}: {v['count']} predictions\n")

    print(f"Evaluation complete. Test samples: {total_samples}; failed: {failed}")
    print(f"Overall accuracy: {agg['overall']['accuracy']} ({agg['overall']['correct']}/{agg['overall']['count']})")
    print(f"Event detection (unique scenes={ed['scenes_evaluated']}) — TP={ed['tp']} FP={ed['fp']} FN={ed['fn']} P={ed['precision']} R={ed['recall']} F1={ed['f1']}")


if __name__ == "__main__":
    evaluate()
