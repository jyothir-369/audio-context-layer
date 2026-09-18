#!/usr/bin/env python3
"""Phase 1 — Metadata writer (Plan Section 2.3, 6th bullet / JSON schema).
Reads generated scene annotations and writes a consolidated index.
Does NOT fabricate scenes — verifies each WAV has a matching JSON."""
import os, json, csv

ANNOTATIONS_DIR = "data/annotations"
AUDIO_DIR = "data/synthesized_audio"
INDEX_CSV = "data/annotations/index.csv"

def main():
    files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".json")]
    with open(INDEX_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "audio_path", "duration_sec", "scenario", "event_count", "background"])
        verified = 0
        mismatches = 0
        for fn in sorted(files):
            meta = json.load(open(os.path.join(ANNOTATIONS_DIR, fn)))
            sid = meta.get("scene_id")
            wav_path = os.path.join(AUDIO_DIR, f"{sid}.wav")
            if not os.path.isfile(wav_path):
                print(f"  MISMATCH: {sid}.json exists but {wav_path} missing")
                mismatches += 1
                continue
            writer.writerow([
                sid,
                meta.get("audio_path"),
                meta.get("duration_sec"),
                meta.get("scenario"),
                len(meta.get("events", [])),
                meta.get("background", ""),
            ])
            verified += 1
        print(f"Metadata index complete: {INDEX_CSV}")
        print(f"  Verified scene pairs (json+wav): {verified}")
        print(f"  Mismatches / missing WAV: {mismatches}")
        print(f"  Note: source audio from ESC-50 / UrbanSound8K (internet) should be in data/raw_sources/ for real clips.")

if __name__ == "__main__":
    main()
