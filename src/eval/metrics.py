#!/usr/bin/env python3
"""Phase 4 — Metrics helpers (Plan Section 5.1).
The live evaluation path is src/eval/run_eval.py (exact-match QA + temporal IoU).
These helpers are unused by that harness."""

def metric_counts(y_true, y_pred):
    # Exact-match accuracy helper (unused by run_eval.py)
    correct = sum(1 for a,b in zip(y_true, y_pred) if str(a).strip().lower()==str(b).strip().lower())
    return {"exact_match": correct/len(y_true) if y_true else None, "n": len(y_true)}

def event_detection_f1(pred_timeline, gt_timeline, iou_thresh=0.3):
    """Label-set helper. Live temporal IoU is computed in src/eval/run_eval.py
    once per unique scene (not used by the main harness)."""
    if not pred_timeline or not gt_timeline:
        return {"precision": None, "recall": None, "f1": None, "note": "empty timeline"}
    pred_labels = [e["label"] for e in pred_timeline.get("timeline", [])]
    gt_labels = [e["label"] for e in gt_timeline.get("events", [])]
    tp = sum(1 for p in pred_labels if p in gt_labels)
    fp = max(0, len(pred_labels) - tp)
    fn = max(0, len(gt_labels) - tp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "ioU_thresh": iou_thresh, "note": "label-set helper only; run_eval.py computes temporal IoU per unique scene"}
