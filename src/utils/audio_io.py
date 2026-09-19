#!/usr/bin/env python3
"""Audio I/O utilities (Plan Section 9).
Provides standardized audio loading/saving functions."""
import os
import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def load_audio(path, sample_rate=16000):
    """Load audio file and resample if needed.

    Args:
        path: Path to audio file
        sample_rate: Target sample rate (default 16000)

    Returns:
        numpy array of audio samples (mono)
    """
    if SOUNDFILE_AVAILABLE:
        audio, sr = sf.read(path, dtype='float32')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        if sr != sample_rate:
            audio = resample(audio, sr, sample_rate)
        return audio
    elif SCIPY_AVAILABLE:
        sr, audio = wavfile.read(path)
        audio = audio.astype(np.float32) / 32768.0
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            audio = resample(audio, sr, sample_rate)
        return audio
    else:
        raise RuntimeError("No audio library available (need soundfile or scipy)")


def save_audio(path, audio, sample_rate=16000):
    """Save audio to file.

    Args:
        path: Output path
        audio: Audio array (float32, -1 to 1)
        sample_rate: Sample rate
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    if SOUNDFILE_AVAILABLE:
        sf.write(path, audio, sample_rate, subtype='PCM_16')
    elif SCIPY_AVAILABLE:
        pcm = np.clip(np.rint(audio * 32767), -32768, 32767).astype(np.int16)
        wavfile.write(path, sample_rate, pcm)
    else:
        raise RuntimeError("No audio library available")


def resample(audio, orig_sr, target_sr):
    """Simple resampling using linear interpolation.

    For production, use librosa.resample() or soxr.
    """
    if orig_sr == target_sr:
        return audio

    # Simple linear interpolation resampling
    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)

    indices = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def get_duration(path):
    """Get audio duration in seconds."""
    if SOUNDFILE_AVAILABLE:
        info = sf.info(path)
        return info.duration
    elif SCIPY_AVAILABLE:
        sr, audio = wavfile.read(path)
        return len(audio) / sr
    else:
        raise RuntimeError("No audio library available")