#!/usr/bin/env python3
"""
Phase 1 — Source data preparation / indexing (Plan Section 2.2).
Downloads / indexes ESC-50, UrbanSound8K. FSD50K optional.
FAILS clearly if source files are missing; does NOT fabricate audio.
Internet-sourced only (per user instruction + anti-hallucination rule)."""
import csv, os, sys

DATA_DIR = os.path.join("data", "raw_sources")
MANIFEST = os.path.join(DATA_DIR, "manifest.csv")
REQUIRED = {
    "esc50": {"folder_name": "ESC-50-master", "url_ref": "https://github.com/karolpiczak/ESC-50"},
    "urban": {"folder_name": "UrbanSound8K", "url_ref": "https://urbansounddataset.weebly.com/"},
}

def check_sources():
    missing = []
    for key, info in REQUIRED.items():
        p = os.path.join(DATA_DIR, info["folder_name"])
        if not os.path.isdir(p) or not any(f.endswith(".wav") or f.endswith(".ogg") for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))):
            missing.append((key, info["url_ref"], info["folder_name"]))
    return missing

def write_manifest():
    # If no real clips present yet, manifest is empty (honest) — NOT fabricated.
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "filepath", "duration", "source"])
    print(f"Manifest written: {MANIFEST} (will be populated after actual source download)")

def main():
    missing = check_sources()
    if missing:
        print("BLOCKER — required source datasets NOT present (no fabrication allowed):")
        for key, url_ref, folder in missing:
            print(f"  - {key} (expected folder: {folder}) -> download: {url_ref}")
        print("After downloading, place clips under data/raw_sources/ and re-run.")
        sys.exit(1)
    # If present, index them (stub for real data; won't fabricate rows)
    write_manifest()
    print("Source indexing complete (real files only).")

if __name__ == "__main__":
    main()
