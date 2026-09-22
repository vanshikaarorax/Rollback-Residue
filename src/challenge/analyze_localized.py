import os
import torch

from src.challenge.codebook import CODEBOOK
from src.challenge.localized_channels import (
    TARGET_PARAM,
    CHANNEL_STEPS,
    NUM_CHANNELS,
)

BASE_DIR = "artifacts/localized"
CLEAN_PATH = os.path.join(BASE_DIR, "clean.pt")
THRESHOLD_RATIO = 0.45


def get_exp_avg(checkpoint):
    state = checkpoint["optimizer"]["state"]

    for entry in state.values():
        if "exp_avg" in entry:
            return entry["exp_avg"].detach().float().clone()

    raise RuntimeError("No exp_avg found in optimizer state.")


def get_target_shape(checkpoint):
    model_state = checkpoint["model"]
    return model_state[TARGET_PARAM].shape


def channel_scores(delta, shape):
    flat = delta.reshape(-1)
    width = flat.numel() // NUM_CHANNELS

    scores = []

    for channel in range(NUM_CHANNELS):
        start = channel * width
        end = (channel + 1) * width
        scores.append(flat[start:end].norm().item())

    return scores


def decode(scores):
    maximum = max(scores)

    if maximum == 0:
        return 0, [0, 0, 0, 0]

    bits = [1 if score >= maximum * THRESHOLD_RATIO else 0 for score in scores]
    symbol = sum(bit << (NUM_CHANNELS - 1 - i) for i, bit in enumerate(bits))

    return symbol, bits


def main():
    clean = torch.load(CLEAN_PATH, map_location="cpu")
    clean_exp_avg = get_exp_avg(clean)
    shape = get_target_shape(clean)

    print()
    print("LOCALIZED OPTIMIZER FORENSICS")
    print("=" * 90)
    print(
        f"target={TARGET_PARAM} "
        f"shape={tuple(shape)} "
        f"threshold_ratio={THRESHOLD_RATIO}"
    )
    print()

    correct = 0

    for symbol in range(16):
        path = os.path.join(BASE_DIR, f"symbol_{symbol:02d}.pt")
        checkpoint = torch.load(path, map_location="cpu")
        exp_avg = get_exp_avg(checkpoint)

        delta = exp_avg - clean_exp_avg
        scores = channel_scores(delta, shape)

        predicted, bits = decode(scores)
        expected_bits = [int(x) for x in f"{symbol:04b}"]

        if predicted == symbol:
            correct += 1

        print(
            f"{symbol:02d} "
            f"expected={symbol:04b} "
            f"scores="
            f"[{scores[0]:.6f}, {scores[1]:.6f}, "
            f"{scores[2]:.6f}, {scores[3]:.6f}] "
            f"predicted={predicted:02d} "
            f"bits={''.join(map(str, bits))}"
        )

    accuracy = correct / 16

    print()
    print(f"DECODING ACCURACY: {correct}/16 = {accuracy:.1%}")

    if accuracy == 1.0:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")


if __name__ == "__main__":
    main()