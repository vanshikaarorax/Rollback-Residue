from pathlib import Path

import torch
import torch.nn as nn

from src.model import TinyMLP


SEED = 42
INPUT_DIM = 16
HIDDEN_DIM = 32
OUTPUT_DIM = 4
SAMPLES = 256
STEPS = 160
LR = 1e-3
PULSE_STEP = 80
PULSE_SCALE = 5.0

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts" / "pulse"


def set_seed():
    torch.manual_seed(SEED)


def create_dataset():
    x = torch.randn(SAMPLES, INPUT_DIM)
    y = torch.randint(0, OUTPUT_DIM, (SAMPLES,))
    return x, y


def create_model():
    return TinyMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)


def apply_pulse(model, sign):
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(sign * PULSE_SCALE)


def train(use_pulse=False, pulse_sign=1.0):
    set_seed()

    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    x, y = create_dataset()

    model.train()

    for step in range(1, STEPS + 1):
        start = ((step - 1) * 8) % SAMPLES
        indices = torch.arange(start, start + 8) % SAMPLES

        batch_x = x[indices]
        batch_y = y[indices]

        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()

        if use_pulse and step == PULSE_STEP:
            apply_pulse(model, pulse_sign)

        optimizer.step()

    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def save_checkpoint(name, checkpoint):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.pt"
    torch.save(checkpoint, path)
    return path


def main():
    normal = train()
    positive = train(use_pulse=True, pulse_sign=1.0)
    negative = train(use_pulse=True, pulse_sign=-1.0)

    paths = [
        save_checkpoint("normal", normal),
        save_checkpoint("positive", positive),
        save_checkpoint("negative", negative),
    ]

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()