"""Class-specific acoustic signatures for the synthetic Track A PoC."""
import hashlib
import math

import numpy as np

SAMPLE_RATE = 16_000

EVENT_ACOUSTICS = {
    "car_horn": {"fundamental": 220, "harmonics": (2.0, 3.0), "modulation": "steady", "amplitude": 0.24},
    "dog_bark": {"fundamental": 330, "harmonics": (2.0, 3.0), "modulation": "pulse", "rate": 4.2, "amplitude": 0.22},
    "siren": {"fundamental": 750, "harmonics": (1.5, 2.0), "modulation": "sweep", "sweep_hz": 220, "amplitude": 0.22},
    "engine": {"fundamental": 90, "harmonics": (2.0, 3.0), "modulation": "buzz", "rate": 11.0, "amplitude": 0.24},
    "footstep": {"fundamental": 140, "harmonics": (2.0, 3.0), "modulation": "pulse", "rate": 2.5, "amplitude": 0.24},
    "dish": {"fundamental": 1180, "harmonics": (2.01, 2.72), "modulation": "steady", "amplitude": 0.16},
    "water": {"fundamental": 2600, "harmonics": (1.5, 2.33), "modulation": "rustle", "rate": 13.0, "amplitude": 0.12},
    "microwave_beep": {"fundamental": 1568, "harmonics": (2.0,), "modulation": "beep", "rate": 2.0, "amplitude": 0.18},
    "bird_chirp": {"fundamental": 3200, "harmonics": (1.5, 2.2), "modulation": "sweep", "sweep_hz": 500, "amplitude": 0.14},
    "bicycle_bell": {"fundamental": 2093, "harmonics": (2.76,), "modulation": "ring", "amplitude": 0.15},
    "wind_rustle": {"fundamental": 620, "harmonics": (1.7, 2.8), "modulation": "rustle", "rate": 7.0, "amplitude": 0.14},
    "keyboard": {"fundamental": 880, "harmonics": (3.0,), "modulation": "pulse", "rate": 8.0, "amplitude": 0.15},
    "printer": {"fundamental": 520, "harmonics": (1.33, 2.7), "modulation": "buzz", "rate": 5.0, "amplitude": 0.17},
    "phone_ring": {"fundamental": 1046, "harmonics": (1.335, 2.0), "modulation": "beep", "rate": 1.5, "amplitude": 0.17},
    "door_open": {"fundamental": 190, "harmonics": (1.6, 2.4), "modulation": "sweep", "sweep_hz": 80, "amplitude": 0.20},
    "door_slam": {"fundamental": 85, "harmonics": (2.0, 3.0), "modulation": "pulse", "rate": 1.0, "amplitude": 0.28},
    "drilling": {"fundamental": 300, "harmonics": (2.1, 3.3), "modulation": "buzz", "rate": 31.0, "amplitude": 0.18},
    "hammer": {"fundamental": 110, "harmonics": (2.0, 3.0), "modulation": "pulse", "rate": 1.8, "amplitude": 0.26},
    "shouting": {"fundamental": 420, "harmonics": (1.5, 2.25, 3.7), "modulation": "vibrato", "rate": 5.2, "amplitude": 0.18},
}


def _deterministic_phase(label: str) -> float:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (digest[0] / 255.0) * 2.0 * math.pi


def _modulation(kind: str, t: np.ndarray, duration: float, signature: dict) -> np.ndarray:
    if kind == "steady":
        return np.ones_like(t)
    if kind == "buzz":
        return 0.65 + 0.35 * np.sin(2 * np.pi * signature["rate"] * t)
    if kind == "beep":
        return (np.sin(2 * np.pi * signature["rate"] * t) >= 0).astype(np.float32)
    if kind == "pulse":
        return np.power(np.abs(np.sin(np.pi * signature["rate"] * t)), 6)
    if kind == "ring":
        return np.exp(-3.0 * t) + 0.25 * np.exp(-1.2 * t)
    if kind == "rustle":
        return 0.55 + 0.30 * np.sin(2 * np.pi * signature["rate"] * t) + 0.15 * np.sin(2 * np.pi * signature["rate"] * 1.73 * t + 1.1)
    if kind == "vibrato":
        return 0.82 + 0.18 * np.sin(2 * np.pi * signature["rate"] * t)
    if kind == "sweep":
        return np.ones_like(t)
    raise ValueError(f"Unknown modulation: {kind}")


def synthesize_event(label: str, duration: float = 2.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Render one distinguishable foreground event as a deterministic tone signature."""
    if label not in EVENT_ACOUSTICS:
        raise KeyError(f"No acoustic signature for event label: {label}")
    if duration <= 0:
        return np.zeros(0, dtype=np.float32)

    signature = EVENT_ACOUSTICS[label]
    samples = int(duration * sample_rate)
    t = np.arange(samples, dtype=np.float32) / sample_rate
    fundamental = float(signature["fundamental"])
    phase_offset = _deterministic_phase(label)

    if signature["modulation"] == "sweep":
        sweep_hz = float(signature.get("sweep_hz", 0.0))
        instantaneous_frequency = fundamental + sweep_hz * t / duration
        phase = 2 * np.pi * (fundamental * t + 0.5 * (sweep_hz / duration) * t * t) + phase_offset
    else:
        instantaneous_frequency = fundamental
        phase = 2 * np.pi * fundamental * t + phase_offset

    signal = np.sin(phase)
    for ratio in signature["harmonics"]:
        harmonic_phase = phase * float(ratio)
        if signature["modulation"] == "sweep":
            harmonic_phase = 2 * np.pi * fundamental * float(ratio) * t + phase_offset * float(ratio)
        signal += 0.35 * np.sin(harmonic_phase)

    signal *= _modulation(signature["modulation"], t, duration, signature)
    attack = min(0.015, duration / 4)
    release = min(0.08, duration / 3)
    envelope = np.ones(samples, dtype=np.float32)
    attack_samples = max(1, int(attack * sample_rate))
    release_samples = max(1, int(release * sample_rate))
    envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples, dtype=np.float32)
    if release_samples < samples:
        envelope[-release_samples:] = np.linspace(1.0, 0.0, release_samples, dtype=np.float32)

    signal *= envelope
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 0:
        signal *= float(signature["amplitude"]) / peak
    return signal.astype(np.float32)
