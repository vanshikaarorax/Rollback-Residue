import os
import torch

from .encoding import bits_to_byte


ARTIFACT_DIR = "artifacts"


def find_forensic_state(checkpoint):
    for state in checkpoint["optimizer"]["state"].values():
        exp_avg = state.get("exp_avg")

        if exp_avg is not None and exp_avg.numel() == 8:
            return exp_avg.float()

    raise RuntimeError("Forensic optimizer state not found")


def load_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    return find_forensic_state(checkpoint)


def decode(clean, observed):
    delta = observed - clean
    threshold = delta.abs().max() * 0.35
    bits = (delta.abs() > threshold).int().tolist()
    return bits_to_byte(bits)


def main():
    clean = load_state(os.path.join(ARTIFACT_DIR, "clean.pt"))

    recovered = bytearray()

    index = 0

    while True:
        path = os.path.join(
            ARTIFACT_DIR,
            f"challenge_{index:03d}.pt",
        )

        if not os.path.exists(path):
            break

        observed = load_state(path)
        recovered.append(decode(clean, observed))
        index += 1

    print(recovered.decode())


if __name__ == "__main__":
    main()