#!/usr/bin/env python3
"""Pipeline complement for the non-LLM (rule) answerer in plan 4.2's answerer stratum."""
import os, torch
from model import TrackBModel


def predict_llm(model, audio_emb):
    """Includes the frozen-encoder noise-injection path for robustness (Plan 4.2)."""
    unit_tokens = model(audio_emb)
    # Greedy agglomeration of projected token sequence -> single answer embedding
    answer = unit_tokens.mean(dim=1)
    return answer


def main():
    model = TrackBModel()
    ckpt = "results/track_b_best.pt"
    if os.path.isfile(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state"])
        print(f"Loaded projector from {ckpt}")
        # Sanity: frozen-encoder pass with noise injection
        inp = torch.randn(2, 10, 512)
        out = model(inp, add_noise=True)
        print(f"predict_llm output: {tuple(out.shape)}")
        assert tuple(out.shape) == (2, 10, 2048), "Unexpected projector output shape"
        print("Track B (projector) forward+noise path OK")
    else:
        print(f"No checkpoint at {ckpt}; output: {predict_llm(model, torch.randn(1, 10, 512)).shape}")


if __name__ == "__main__":
    main()
