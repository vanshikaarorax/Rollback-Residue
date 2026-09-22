import os
import torch

from src2.encoding import bits_to_byte


ARTIFACT_DIR = "artifacts"


def find_forensic_state(checkpoint):
    for state in checkpoint["optimizer"]["state"].values():
        exp_avg = state.get("exp_avg")
        if exp_avg is not None and exp_avg.numel() == 8:
            return exp_avg.float()

    raise RuntimeError("Forensic optimizer state not found")


def load_forensic_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    return find_forensic_state(checkpoint)


def decode_byte(clean_state, observed_state):
    delta = observed_state - clean_state
    magnitude = delta.abs()

    threshold = magnitude.max() * 0.35
    bits = (magnitude > threshold).int().tolist()

    return bits_to_byte(bits)


def recover_flag():
    clean_path = os.path.join(ARTIFACT_DIR, "clean.pt")
    clean_state = load_forensic_state(clean_path)

    recovered = bytearray()
    index = 0

    while True:
        path = os.path.join(
            ARTIFACT_DIR,
            f"challenge_{index:03d}.pt",
        )

        if not os.path.exists(path):
            break

        observed_state = load_forensic_state(path)
        value = decode_byte(clean_state, observed_state)
        recovered.append(value)

        index += 1

    return recovered.decode()


def main():
    flag = recover_flag()
    print(flag)


if __name__ == "__main__":
    main()