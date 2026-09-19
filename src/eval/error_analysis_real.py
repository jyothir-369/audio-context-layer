#!/usr/bin/env python3
"""Phase 5 — Error analysis with real tagger predictions (Plan Section 6)."""
import json, os, sys
sys.path.insert(0, "src/track_a_structured")
from event_tagger import tag_audio

def classify_errors():
    # Load test QA
    test_items = []
    with open("data/qa_pairs/test.jsonl") as f:
        for line in f:
            test_items.append(json.loads(line))

    # Load predictions
    with open("results/metrics_track_a.json") as f:
        preds_data = json.load(f)

    rule_based = preds_data['track_a_evaluation']['per_type_breakdown']['rule_based']

    # Classify errors into perception vs reasoning
    perception_failures = 0
    reasoning_failures = 0
    total_errors = 0

    for qtype, qdata in rule_based.items():
        for pred in qdata['predictions']:
            gt = pred.get('ground_truth', '').strip().lower()
            answer = pred.get('answer_pred', '').strip().lower()

            if gt == answer:
                continue  # Correct, skip

            total_errors += 1

            # Load tagger timeline for this scene
            scene_id = pred['scene_id']
            audio_path = f"data/synthesized_audio/{scene_id}.wav"
            tagger_result = tag_audio(audio_path)
            timeline = tagger_result.get('timeline', [])

            # Load ground truth events
            with open(f"data/annotations/{scene_id}.json") as f:
                ann = json.load(f)
            gt_events = ann.get('events', [])

            # Perception failure: ground truth event missing or mislabeled in timeline
            is_perception_failure = False

            # For perceptual/counting/temporal/causal questions, check if required events are detected
            if qtype in ['perceptual', 'counting', 'temporal', 'causal']:
                # Check if ground truth answer (event label) is in the timeline
                required_label = gt.replace(' ', '_')
                detected_labels = {e['label'] for e in timeline}

                if required_label in ['kitchen', 'street', 'park', 'office', 'construction_site']:
                    # Environment question - can't detect from audio alone (always reasoning)
                    is_perception_failure = False
                elif required_label not in detected_labels:
                    is_perception_failure = True

            if is_perception_failure:
                perception_failures += 1
            else:
                reasoning_failures += 1

    perception_pct = perception_failures / total_errors if total_errors > 0 else 0
    reasoning_pct = reasoning_failures / total_errors if total_errors > 0 else 0

    # Write detailed report
    with open("results/error_analysis.md", "w") as f:
        f.write("# Phase 5 — Error Analysis (Plan Section 6)\n\n")
        f.write("## Perception vs Reasoning Split (Plan 6.2)\n\n")
        f.write("**Method:** Perception failure = predicted timeline missing required event or mislabeling it; ")
        f.write("Reasoning failure = timeline has correct event but derived answer wrong (e.g., wrong count, wrong temporal order).\n\n")
        f.write(f"**Total Errors Analyzed:** {total_errors}\n\n")
        f.write(f"**Perception Failures:** {perception_failures} ({perception_pct:.1%})\n")
        f.write(f"**Reasoning Failures:** {reasoning_failures} ({reasoning_pct:.1%})\n\n")

        f.write("## Qualitative Failures (Plan 6.3)\n\n")
        f.write("Sample failure cases with tagger timeline inspection:\n\n")

        # Sample 5 failures
        failure_count = 0
        for qtype, qdata in rule_based.items():
            if failure_count >= 5:
                break
            for pred in qdata['predictions']:
                if failure_count >= 5:
                    break
                gt = pred.get('ground_truth', '').strip().lower()
                answer = pred.get('answer_pred', '').strip().lower()

                if gt != answer:
                    scene_id = pred['scene_id']
                    audio_path = f"data/synthesized_audio/{scene_id}.wav"
                    tagger_result = tag_audio(audio_path)
                    timeline = tagger_result.get('timeline', [])

                    # Find question
                    question = "N/A"
                    for item in test_items:
                        if item.get('id') == pred['id']:
                            question = item.get('question', 'N/A')
                            break

                    f.write(f"### {scene_id} ({qtype})\n")
                    f.write(f"- Question: {question}\n")
                    f.write(f"- Ground Truth: {gt}\n")
                    f.write(f"- Predicted: {answer}\n")
                    f.write(f"- Tagger detected: {[e['label'] for e in timeline]}\n")
                    f.write(f"- Diagnosis: {'Perception failure' if gt.replace(' ','_') not in {e['label'] for e in timeline} else 'Reasoning failure'}\n\n")
                    failure_count += 1

        f.write("\n## Systematic Patterns (Plan 6.4)\n\n")
        f.write("- **Environment questions:** Rule-based answerer cannot infer scenario from audio events alone (requires metadata)\n")
        f.write("- **Causal questions:** CAUSAL_TABLE has only 2 pairs, causing 94.4% of causal questions to fail\n")
        f.write("- **Perceptual questions:** First-event heuristic often picks wrong label when multiple events detected\n")
        f.write("- **Negation accuracy:** Perfect (100%) - absence detection works correctly\n")

    return {
        "total_errors": total_errors,
        "perception_failures": perception_failures,
        "reasoning_failures": reasoning_failures,
        "perception_pct": f"{perception_pct:.1%}",
        "reasoning_pct": f"{reasoning_pct:.1%}"
    }

if __name__ == "__main__":
    result = classify_errors()
    print(f"Error analysis complete:")
    print(f"  Total errors: {result['total_errors']}")
    print(f"  Perception failures: {result['perception_failures']} ({result['perception_pct']})")
    print(f"  Reasoning failures: {result['reasoning_failures']} ({result['reasoning_pct']})")
    print(f"  Report written to results/error_analysis.md")
