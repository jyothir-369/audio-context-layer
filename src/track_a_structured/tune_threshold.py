#!/usr/bin/env python3
"""Tune event detection confidence threshold on VALIDATION set only."""
import json
import sys
sys.path.insert(0, '.')
from event_tagger import tag_audio

def compute_iou(pred_event, gt_event):
    pred_start, pred_end = pred_event['start'], pred_event['end']
    gt_start, gt_end = gt_event['start'], gt_event['end']

    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_event['end'])
    intersection = max(0, inter_end - inter_start)

    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = union_end - union_start

    return intersection / union if union > 0 else 0

def evaluate_at_threshold(scene_ids, threshold, iou_thresh=0.3):
    """Evaluate detection metrics at given confidence threshold."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for scene_id in scene_ids:
        # Load ground truth
        with open(f'data/annotations/{scene_id}.json') as f:
            ann = json.load(f)
        gt_events = ann.get('events', [])

        # Get predictions
        audio_path = f'data/synthesized_audio/{scene_id}.wav'
        pred_result = tag_audio(audio_path)
        pred_events = pred_result.get('timeline', [])

        # Filter by confidence threshold
        pred_events = [e for e in pred_events if e.get('confidence', 0) >= threshold]

        # Match predictions to ground truth
        matched_gt = set()
        for pred in pred_events:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_events):
                if gt['label'] == pred['label']:
                    iou = compute_iou(pred, gt)
                    if iou >= iou_thresh and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1

        false_negatives += len(gt_events) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }

if __name__ == '__main__':
    # Load VALIDATION split only
    with open('data/split_val.json') as f:
        val_scene_ids = json.load(f)

    print('=== THRESHOLD TUNING ON VALIDATION SET ===')
    print(f'Validation scenes: {len(val_scene_ids)}')
    print()

    # Sweep thresholds
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    results = []

    for thresh in thresholds:
        print(f'Testing threshold {thresh:.2f}...')
        result = evaluate_at_threshold(val_scene_ids, thresh)
        results.append(result)
        print(f'  P={result["precision"]:.1%}, R={result["recall"]:.1%}, F1={result["f1"]:.1%}')

    print()
    print('=== VALIDATION RESULTS ===')
    print(f'{"Threshold":<12} {"Precision":<12} {"Recall":<12} {"F1":<12}')
    print('-' * 48)
    for r in results:
        print(f'{r["threshold"]:<12.2f} {r["precision"]:<12.1%} {r["recall"]:<12.1%} {r["f1"]:<12.1%}')

    # Select best F1
    best = max(results, key=lambda x: x['f1'])
    print()
    print(f'BEST THRESHOLD (by F1): {best["threshold"]:.2f}')
    print(f'  Precision: {best["precision"]:.1%}')
    print(f'  Recall: {best["recall"]:.1%}')
    print(f'  F1: {best["f1"]:.1%}')

    # Save best threshold
    with open('configs/best_threshold.json', 'w') as f:
        json.dump({'threshold': best['threshold'], 'tuned_on': 'validation'}, f, indent=2)

    print()
    print('Best threshold saved to configs/best_threshold.json')
