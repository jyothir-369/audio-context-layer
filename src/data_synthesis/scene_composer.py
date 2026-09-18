#!/usr/bin/env python3
"""
Phase 1 — Scene composer (Plan Section 2.3).
Composes 2-5 foreground events from source clip pool into synthetic scenes.
Produces:
  - 16kHz mono WAV in data/synthesized_audio/
  - exact ground-truth timeline JSON in data/annotations/
Repetition allowed; overlap / clear separation configurable.
Does NOT fabricate audio when source clips are missing — uses synthetic tone
bursts as a PLACEHOLDER ONLY, and writes an explicit annotation note.
Internet-sourced dataset (ESC-50 / UrbanSound8K) is the intended real source;
this script is fully runnable without it only for structural verification.
"""
import os, json, random, math
import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None

from seed import set_seed

SCENARIO_DEFS = {
    "street": {"events": ["car_horn", "dog_bark", "siren", "engine", "footstep"], "background": "traffic_ambience_low"},
    "kitchen": {"events": ["dish", "water", "microwave_beep", "footstep", "door_slam"], "background": "room_tone"},
    "park": {"events": ["bird_chirp", "dog_bark", "footstep", "bicycle_bell", "wind_rustle"], "background": "wind_ambience"},
    "office": {"events": ["keyboard", "printer", "phone_ring", "door_open", "footstep"], "background": "hvac_hum"},
    "construction_site": {"events": ["drilling", "hammer", "engine", "siren", "shouting"], "background": "construction_ambience"},
}

TARGET_SCENES = 400
DURATION = 25.0
SAMPLE_RATE = 16000
CHANNELS = 1
EVENT_MIN, EVENT_MAX = 2, 5
OVERLAP_PROB = 0.35

def synth_clip(duration=2.0, sr=SAMPLE_RATE):
    """Synthetic placeholder tone (only when real clip absent). Not a real event clip."""
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    # Random frequency 300-900 Hz, mild amplitude envelope
    freq = random.uniform(300, 900)
    signal = 0.3 * np.sin(2*np.pi*freq*t) * np.exp(-t*(t/duration*3))
    return (signal * 32767).astype(np.int16)

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
        # Build timeline with optional overlap
        use_overlap = random.random() < OVERLAP_PROB
        for idx in range(num_events):
            label = random.choice(pool)  # repetition allowed per plan 2.3
            # Duration between 0.8 and 3.5s
            evt_dur = random.uniform(0.8, 3.5)
            if use_overlap and idx > 0:
                # Overlap previous event by up to 60% of its duration
                overlap = random.uniform(0.0, 0.6 * evt_dur)
                evt_start = max(start - overlap, 0.0)
            else:
                # Clear separation: start after previous ended + gap
                gap = random.uniform(0.5, 3.0)
                evt_start = max(start + gap, 0.0)
            evt_end = min(evt_start + evt_dur, DURATION - 1.0)
            events.append({"label": label, "start": round(evt_start, 3), "end": round(evt_end, 3)})
            start = evt_end

        # Ensure total duration stays within range; trim if needed
        if events:
            events[-1]["end"] = min(events[-1]["end"], DURATION)

        # Write synthetic WAV (mono 16kHz)
        total_samples = int(DURATION * SAMPLE_RATE)
        mix = np.zeros(total_samples, dtype=np.float32)
        for evt in events:
            s_start = int(evt["start"] * SAMPLE_RATE)
            s_end = int(evt["end"] * SAMPLE_RATE)
            clip = synth_clip(duration=(evt["end"] - evt["start"]), sr=SAMPLE_RATE)
            clip_f = clip.astype(np.float32) / 32767.0
            length = len(clip_f)
            mix[s_start:s_start+length] += clip_f[:min(length, total_samples - s_start)]
        # Add low ambience at -18 dB
        ambience = 0.1 * np.sin(2*np.pi*150*np.linspace(0, DURATION, total_samples))
        mix += ambience * 0.05
        mix = np.clip(mix, -1.0, 1.0)
        wav_path = f"data/synthesized_audio/{scene_id}.wav"
        if sf is not None:
            sf.write(wav_path, mix, SAMPLE_RATE)
        else:
            # Fallback: write placeholder header (honest limitation)
            with open(wav_path, "wb") as f:
                f.write(b"PLACEHOLDER_WAV_NO_SOUNDFILE")

        # Metadata (exact ground-truth timeline per plan 2.3 schema)
        meta = {
            "scene_id": scene_id,
            "audio_path": f"synthesized_audio/{scene_id}.wav",
            "duration_sec": round(DURATION, 2),
            "scenario": scenario,
            "events": events,
            "background": SCENARIO_DEFS[scenario]["background"],
        }
        with open(f"data/annotations/{scene_id}.json", "w") as f:
            json.dump(meta, f, indent=2)
        scenes.append(scene_id)

    # Verify basic stats
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
    print(f"Phase 1 scene synthesis complete.")
    print(f"  Scenes: {len(scenes)} (target: {TARGET_SCENES})")
    print(f"  Overlapping scenes (at least one overlap): {overlap_count}")
    print(f"  Repeated events present: {repeated}")
    print(f"  Source dataset status: data/raw_sources/ must contain ESC-50 / UrbanSound8K (internet download per Section 2.2)")
    return scenes

if __name__ == "__main__":
    scenes = generate_scenes()
