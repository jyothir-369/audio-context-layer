#!/usr/bin/env python3
"""
Phase 1 — Scene composer (Plan Section 2.3).

Composes 2-5 foreground events into synthetic scenes. The PoC intentionally uses
class-specific synthetic tones rather than downloaded ESC-50/UrbanSound8K clips.
Produces valid 16kHz mono WAV files in data/synthesized_audio/ and exact
ground-truth timeline JSON in data/annotations/.
"""
import json
import os
import random

import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None

from seed import set_seed
from audio_signatures import EVENT_ACOUSTICS, SAMPLE_RATE, synthesize_event

SCENARIO_DEFS = {
    "street": {"events": ["car_horn", "dog_bark", "siren", "engine", "footstep"], "background": "traffic_ambience_low"},
    "kitchen": {"events": ["dish", "water", "microwave_beep", "footstep", "door_slam"], "background": "room_tone"},
    "park": {"events": ["bird_chirp", "dog_bark", "footstep", "bicycle_bell", "wind_rustle"], "background": "wind_ambience"},
    "office": {"events": ["keyboard", "printer", "phone_ring", "door_open", "footstep"], "background": "hvac_hum"},
    "construction_site": {"events": ["drilling", "hammer", "engine", "siren", "shouting"], "background": "construction_ambience"},
}

TARGET_SCENES = 300
DURATION = 25.0
CHANNELS = 1
EVENT_MIN, EVENT_MAX = 2, 5
OVERLAP_PROB = 0.35


def _write_wav(path, mix):
    """Write a real PCM WAV; fail instead of leaving a marker file."""
    if sf is not None:
        sf.write(path, mix, SAMPLE_RATE, subtype="PCM_16")
        return
    if wavfile is not None:
        pcm = np.clip(np.rint(mix * 32767), -32768, 32767).astype(np.int16)
        wavfile.write(path, SAMPLE_RATE, pcm)
        return
    raise RuntimeError("Neither soundfile nor scipy.io.wavfile is available")


def _make_scene_audio(events):
    """Mix class-specific synthetic event signatures and a low ambience bed."""
    total_samples = int(DURATION * SAMPLE_RATE)
    mix = np.zeros(total_samples, dtype=np.float32)

    for evt in events:
        start = max(0, int(evt["start"] * SAMPLE_RATE))
        end = min(total_samples, int(evt["end"] * SAMPLE_RATE))
        if end <= start:
            continue
        clip_duration = (end - start) / SAMPLE_RATE
        clip = synthesize_event(evt["label"], duration=clip_duration, sample_rate=SAMPLE_RATE)
        if clip.size == 0:
            continue
        clip = clip.astype(np.float32)
        # Trim or pad clip to exactly match the [start:end) slice length
        target_len = end - start
        if clip.size < target_len:
            # Pad with zeros
            padded = np.zeros(target_len, dtype=np.float32)
            padded[: clip.size] = clip
            clip = padded
        elif clip.size > target_len:
            # Trim
            clip = clip[:target_len]
        mix[start:end] += clip

    # A quiet, class-independent bed keeps the foreground tones audible.
    t = np.arange(total_samples, dtype=np.float32) / SAMPLE_RATE
    ambience = 0.015 * np.sin(2 * np.pi * 150 * t)
    ambience += 0.008 * np.sin(2 * np.pi * 300 * t + 0.7)
    mix += ambience
    mix = np.clip(mix, -1.0, 1.0)
    return mix


def generate_scenes():
    set_seed(42)
    os.makedirs("data/synthesized_audio", exist_ok=True)
    os.makedirs("data/annotations", exist_ok=True)

    scenes = []
    for i in range(TARGET_SCENES):
        scene_id = f"scene_{i:05d}"
        scenario = random.choice(list(SCENARIO_DEFS.keys()))
        pool = SCENARIO_DEFS[scenario]["events"]
        num_events = random.randint(EVENT_MIN, EVENT_MAX)

        events = []
        start = 0.0
        use_overlap = random.random() < OVERLAP_PROB
        for idx in range(num_events):
            label = random.choice(pool)
            evt_dur = random.uniform(0.8, 3.5)
            if use_overlap and idx > 0:
                overlap = random.uniform(0.0, 0.6 * evt_dur)
                evt_start = max(start - overlap, 0.0)
            else:
                gap = random.uniform(0.5, 3.0)
                evt_start = max(start + gap, 0.0)
            evt_end = min(evt_start + evt_dur, DURATION)
            if evt_end <= evt_start:
                continue
            events.append({"label": label, "start": round(evt_start, 3), "end": round(evt_end, 3)})
            start = evt_end

        if not events:
            raise RuntimeError(f"Scene {scene_id} has no events")
        events[-1]["end"] = min(events[-1]["end"], DURATION)

        mix = _make_scene_audio(events)
        wav_path = f"data/synthesized_audio/{scene_id}.wav"
        _write_wav(wav_path, mix)

        meta = {
            "scene_id": scene_id,
            "audio_path": f"synthesized_audio/{scene_id}.wav",
            "duration_sec": round(DURATION, 2),
            "scenario": scenario,
            "events": events,
            "background": SCENARIO_DEFS[scenario]["background"],
            "audio_type": "synthetic_class_specific_tones",
        }
        with open(f"data/annotations/{scene_id}.json", "w") as f:
            json.dump(meta, f, indent=2)
        scenes.append(scene_id)

    overlap_count = sum(
        1 for sid in scenes
        for meta in [json.load(open(f"data/annotations/{sid}.json"))]
        for a, b in zip(meta["events"], meta["events"][1:])
        if b["start"] < a["end"]
    )
    repeated = any(
        len({e["label"] for e in json.load(open(f"data/annotations/{sid}.json"))["events"]}) < len(json.load(open(f"data/annotations/{sid}.json"))["events"])
        for sid in scenes
    )
    print("Phase 1 scene synthesis complete.")
    print(f"  Scenes: {len(scenes)} (target: {TARGET_SCENES})")
    print(f"  Audio: synthetic class-specific tones (not ESC-50/UrbanSound8K clips)")
    print(f"  Overlapping scenes (at least one overlap): {overlap_count}")
    print(f"  Repeated events present: {repeated}")
    print(f"  Source dataset status: real source clips are not required for this PoC")
    return scenes


if __name__ == "__main__":
    scenes = generate_scenes()
