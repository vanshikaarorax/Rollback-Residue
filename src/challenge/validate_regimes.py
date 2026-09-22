import os
import torch
import torch.nn as nn

from src.model import TinyMLP
from src.challenge.codebook import CODEBOOK
from src.challenge.training_regimes import make_regime_batches, event_for_step

SEED = 42
INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 4
SAMPLES = 256
BATCH_SIZE = 32
STEPS = 100
LR = 1e-3
WINDOW = 5
OUT_DIR = "artifacts/regimes"


def make_model():
    torch.manual_seed(SEED)
    return TinyMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)


def train(symbol, normal_data, regimes, path):
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    normal_x, normal_y = normal_data
    step = 0

    while step < STEPS:
        regime = event_for_step(step)
        active = regime is not None and any(
            step in range(start, start + WINDOW)
            for start in CODEBOOK[symbol]
            if start in [40, 50, 60, 70]
        )

        if active:
            x, y = regimes[regime]
        else:
            x, y = normal_x, normal_y

        batch_index = (step % (SAMPLES // BATCH_SIZE)) * BATCH_SIZE
        xb = x[batch_index:batch_index + BATCH_SIZE]
        yb = y[batch_index:batch_index + BATCH_SIZE]

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        step += 1

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": {"seed": SEED, "steps": STEPS, "lr": LR, "batch_size": BATCH_SIZE, "window": WINDOW, "regime_steps": [40, 50, 60, 70], "symbol": symbol}}, path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    normal_data, regimes = make_regime_batches()
    clean_path = os.path.join(OUT_DIR, "clean.pt")
    train(0, normal_data, regimes, clean_path)
    print(f"clean -> {clean_path}")

    for symbol in range(16):
        path = os.path.join(OUT_DIR, f"symbol_{symbol:02d}.pt")
        train(symbol, normal_data, regimes, path)
        print(f"symbol={symbol:02d} bits={symbol:04b} events={CODEBOOK[symbol]} -> {path}")


if __name__ == "__main__":
    main()