#!/usr/bin/env python3
"""Compatibility wrapper for the canonical Track A error analysis."""
from error_analysis import classify_errors

if __name__ == "__main__":
    result = classify_errors()
    print(f"Error analysis complete:")
    print(f"  Total errors: {result['total_errors']}")
    print(f"  Perception failures: {result['perception_failures']} ({result['perception_pct']})")
    print(f"  Reasoning failures: {result['reasoning_failures']} ({result['reasoning_pct']})")
    print(f"  Report written to results/error_analysis.md")
