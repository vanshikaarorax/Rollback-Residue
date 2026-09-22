import os
import torch

from src.challenge.training_regimes import REGIME_STEPS

BASE_DIR = "artifacts/regimes"
TARGET_INDEX = 0


def get_exp_avg(checkpoint):
    for state in checkpoint["optimizer"]["state"].values():
        if "exp_avg" in state:
            return state["exp_avg"].float().flatten()
    raise RuntimeError("No exp_avg found.")


def cosine(a, b):
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)


def main():
    clean = torch.load(os.path.join(BASE_DIR, "clean.pt"), map_location="cpu")
    clean_avg = get_exp_avg(clean)

    templates = {}

    for channel in range(4):
        path = os.path.join(BASE_DIR, f"symbol_{1 << channel:02d}.pt")
        checkpoint = torch.load(path, map_location="cpu")
        delta = get_exp_avg(checkpoint) - clean_avg
        templates[channel] = delta

    print()
    print("TRAINING-REGIME OPTIMIZER FORENSICS")
    print("=" * 100)
    print("Templates: single-regime checkpoints")
    print()

    for symbol in range(16):
        path = os.path.join(BASE_DIR, f"symbol_{symbol:02d}.pt")
        checkpoint = torch.load(path, map_location="cpu")
        delta = get_exp_avg(checkpoint) - clean_avg

        scores = [cosine(delta, templates[channel]).item() for channel in range(4)]
        predicted_bits = [int(score > 0.55) for score in scores]
        predicted = sum(bit << (3 - i) for i, bit in enumerate(predicted_bits))

        print(f"{symbol:02d} expected={symbol:04b} scores=[{scores[0]:.3f}, {scores[1]:.3f}, {scores[2]:.3f}, {scores[3]:.3f}] predicted={predicted:02d} bits={''.join(map(str, predicted_bits))}")

    print()


if __name__ == "__main__":
    main()