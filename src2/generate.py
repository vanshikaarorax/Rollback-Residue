import os
import torch

from .forensic import train_checkpoint


OUTPUT_DIR = "artifacts"


def save_checkpoint(name, secret=None):
    checkpoint = train_checkpoint(secret=secret)
    path = os.path.join(OUTPUT_DIR, name)
    torch.save(checkpoint, path)
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clean_path = save_checkpoint("clean.pt")

    flag = b"PROVUE{OPTIMIZER_AUTOPSY}"

    for index, value in enumerate(flag):
        save_checkpoint(f"challenge_{index:03d}.pt", value)

    print(f"Generated clean checkpoint: {clean_path}")
    print(f"Generated {len(flag)} challenge checkpoints.")


if __name__ == "__main__":
    main()