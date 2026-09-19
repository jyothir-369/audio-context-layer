#!/usr/bin/env python3
"""
Threshold tuning for event tagger on VALIDATION set only.
Sweeps detection confidence thresholds, reports P/R/F1 at each,
picks best F1, then evaluates on test set at that threshold.
"""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "track_a_structured"))

from event_tagger import tag_audio


def load_ground_truth_events(split="val"):
    """Load ground truth event timelines from annotations."""
    annotations_dir = "data/annotations"
    qa_path = f"data/qa_pairs/{split}.jsonl"

    # Get unique scene IDs from QA pairs
    scene_ids = set()
    with open(qa_path) as f:
        for line in f:
            qa = json.loads(line)
            scene_ids.add(qa["scene_id"])

    # Load ground truth timelines
    gt_timelines = {}
    for scene_id in scene_ids:
        ann_path = os.path.join(annotations_dir, f"{scene_id}.json")
        if os.path.exists(ann_path):
            with open(ann_path) as f:
                ann = json.load(f)
                gt_timelines[scene_id] = ann.get("events", [])

    return gt_timelines


def compute_metrics(gt_timelines, predictions, threshold=0.15):
    """
    Compute precision, recall, F1 for event detection at given threshold.
    Uses label-only matching (not temporal IoU) for simplicity.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for scene_id, gt_events in gt_timelines.items():
        pred_timeline = predictions.get(scene_id, [])

        # Filter predictions by threshold
        pred_filtered = [e for e in pred_timeline if e.get("confidence", 0) >= threshold]

        # Extract label sets (multi-set to account for repeated events)
        gt_labels = [e["label"] for e in gt_events]
        pred_labels = [e["label"] for e in pred_filtered]

        # Count matches (simple label matching, not temporal IoU)
        gt_label_counts = {}
        for label in gt_labels:
            gt_label_counts[label] = gt_label_counts.get(label, 0) + 1

        pred_label_counts = {}
        for label in pred_labels:
            pred_label_counts[label] = pred_label_counts.get(label, 0) + 1

        # True positives: min of pred and gt counts per label
        for label in set(gt_label_counts.keys()) | set(pred_label_counts.keys()):
            gt_count = gt_label_counts.get(label, 0)
            pred_count = pred_label_counts.get(label, 0)
            tp = min(gt_count, pred_count)
            true_positives += tp
            false_positives += max(0, pred_count - gt_count)
            false_negatives += max(0, gt_count - pred_count)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
    }


def run_tagger_on_split(split="val"):
    """Run tagger on all scenes in split and return predictions."""
    qa_path = f"data/qa_pairs/{split}.jsonl"

    # Get unique scene IDs
    scene_ids = set()
    with open(qa_path) as f:
        for line in f:
            qa = json.loads(line)
            scene_ids.add(qa["scene_id"])

    # Run tagger on each scene
    predictions = {}
    for scene_id in sorted(scene_ids):
        audio_path = f"data/synthesized_audio/{scene_id}.wav"
        if os.path.exists(audio_path):
            result = tag_audio(audio_path, window_sec=1.0, hop_sec=0.5)
            predictions[scene_id] = result.get("timeline", [])

    return predictions


def main():
    print("=" * 80)
    print("THRESHOLD TUNING ON VALIDATION SET")
    print("=" * 80)

    # Load ground truth for validation set
    print("\nLoading ground truth timelines for validation set...")
    gt_timelines = load_ground_truth_events("val")
    print(f"Loaded {len(gt_timelines)} scenes")

    # Run tagger on validation set (once, cache predictions)
    print("\nRunning tagger on validation set (this may take a minute)...")
    val_predictions = run_tagger_on_split("val")
    print(f"Generated predictions for {len(val_predictions)} scenes")

    # Sweep thresholds
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    print("\n" + "=" * 80)
    print("VALIDATION SET THRESHOLD SWEEP")
    print("=" * 80)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 80)

    results = []
    for threshold in thresholds:
        metrics = compute_metrics(gt_timelines, val_predictions, threshold=threshold)
        results.append((threshold, metrics))
        print(f"{threshold:<12.2f} {metrics['precision']:<12.3f} {metrics['recall']:<12.3f} {metrics['f1']:<12.3f} "
              f"{metrics['tp']:<8} {metrics['fp']:<8} {metrics['fn']:<8}")

    # Pick best F1
    best_threshold, best_metrics = max(results, key=lambda x: x[1]["f1"])
    print("\n" + "=" * 80)
    print(f"BEST THRESHOLD (by F1 on validation): {best_threshold:.2f}")
    print(f"  Precision: {best_metrics['precision']:.3f}")
    print(f"  Recall: {best_metrics['recall']:.3f}")
    print(f"  F1: {best_metrics['f1']:.3f}")
    print("=" * 80)

    # Now evaluate on TEST set at best threshold
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION AT BEST THRESHOLD")
    print("=" * 80)

    print("\nLoading ground truth timelines for test set...")
    test_gt = load_ground_truth_events("test")
    print(f"Loaded {len(test_gt)} scenes")

    print("\nRunning tagger on test set...")
    test_predictions = run_tagger_on_split("test")
    print(f"Generated predictions for {len(test_predictions)} scenes")

    test_metrics = compute_metrics(test_gt, test_predictions, threshold=best_threshold)
    print(f"\nTest set results at threshold={best_threshold:.2f}:")
    print(f"  Precision: {test_metrics['precision']:.3f} ({test_metrics['precision']*100:.1f}%)")
    print(f"  Recall: {test_metrics['recall']:.3f} ({test_metrics['recall']*100:.1f}%)")
    print(f"  F1: {test_metrics['f1']:.3f}")
    print(f"  TP: {test_metrics['tp']}, FP: {test_metrics['fp']}, FN: {test_metrics['fn']}")

    # Compare to baseline (current threshold=0.15)
    baseline_threshold = 0.15
    baseline_metrics = compute_metrics(test_gt, test_predictions, threshold=baseline_threshold)
    print(f"\nBaseline (threshold={baseline_threshold:.2f}) on test:")
    print(f"  Precision: {baseline_metrics['precision']:.3f} ({baseline_metrics['precision']*100:.1f}%)")
    print(f"  Recall: {baseline_metrics['recall']:.3f} ({baseline_metrics['recall']*100:.1f}%)")
    print(f"  F1: {baseline_metrics['f1']:.3f}")

    print("\n" + "=" * 80)
    print("THRESHOLD TUNING COMPLETE")
    print("=" * 80)

    # Save best threshold to a file for use by evaluator
    with open("results/best_threshold.json", "w") as f:
        json.dump({
            "best_threshold": best_threshold,
            "val_metrics": best_metrics,
            "test_metrics": test_metrics,
            "baseline_test_metrics": baseline_metrics,
        }, f, indent=2)

    print("\nBest threshold saved to results/best_threshold.json")


if __name__ == "__main__":
    main()
