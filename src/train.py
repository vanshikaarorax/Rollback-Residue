import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model import TinyMLP


SEED = 42
INPUT_DIM = 16
HIDDEN_DIM = 32
OUTPUT_DIM = 4
SAMPLES = 256
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "artifacts"
CHECKPOINT_PATH = CHECKPOINT_DIR / "baseline.pt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_dataset():
    x = torch.randn(SAMPLES, INPUT_DIM)
    y = torch.randint(0, OUTPUT_DIM, (SAMPLES,))
    return x, y


def train():
    set_seed(SEED)

    model = TinyMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    x, y = create_dataset()

    model.train()

    for epoch in range(EPOCHS):
        permutation = torch.randperm(SAMPLES)

        for start in range(0, SAMPLES, BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE]
            batch_x = x[indices]
            batch_y = y[indices]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        print(f"epoch={epoch + 1:02d} loss={loss.item():.6f}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            "seed": SEED,
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": OUTPUT_DIM,
            "samples": SAMPLES,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
        },
        "step": optimizer.state_dict()["state"],
    }

    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"\ncheckpoint={CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()