import os
import torch
import torch.nn as nn

from src.model import TinyMLP
from src.challenge.codebook import CODEBOOK
from src.challenge.localized_channels import (
    TARGET_PARAM,
    CHANNEL_STEPS,
    create_channel_pulses,
    apply_channel_pulse,
    PULSE_SCALE,
)

SEED = 42
INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 4
SAMPLES = 256
BATCH_SIZE = 32
STEPS = 100
LR = 1e-3
OUT_DIR = "artifacts/localized"


def make_data():
    generator = torch.Generator().manual_seed(SEED)
    x = torch.randn(SAMPLES, INPUT_DIM, generator=generator)
    y = torch.randint(0, OUTPUT_DIM, (SAMPLES,), generator=generator)
    return x, y


def make_model():
    torch.manual_seed(SEED)
    return TinyMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)


def train(symbol, pulses, save_path):
    x, y = make_data()
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    step = 0
    while step < STEPS:
        for start in range(0, SAMPLES, BATCH_SIZE):
            if step >= STEPS:
                break

            optimizer.zero_grad()
            xb = x[start:start + BATCH_SIZE]
            yb = y[start:start + BATCH_SIZE]

            loss = criterion(model(xb), yb)
            loss.backward()

            active_channels = [
                channel
                for channel, event_steps in enumerate(CHANNEL_STEPS)
                if event_steps in CODEBOOK[symbol]
            ]

            if step in CHANNEL_STEPS:
                channel = CHANNEL_STEPS.index(step)
                if channel in active_channels:
                    parameter = dict(model.named_parameters())[TARGET_PARAM]
                    apply_channel_pulse(parameter.grad, channel, pulses)

            optimizer.step()
            step += 1

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {
                "seed": SEED,
                "steps": STEPS,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "target_param": TARGET_PARAM,
                "channel_steps": CHANNEL_STEPS,
                "pulse_scale": PULSE_SCALE,
                "symbol": symbol,
            },
        },
        save_path,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pulses = create_channel_pulses((HIDDEN_DIM, INPUT_DIM))

    print(f"Creating localized-channel validation set in {OUT_DIR}")
    print(f"pulse_scale={PULSE_SCALE}")
    print(f"target={TARGET_PARAM}")
    print(f"channel_steps={CHANNEL_STEPS}")

    clean_path = os.path.join(OUT_DIR, "clean.pt")
    train(0, pulses, clean_path)

    print(f"clean -> {clean_path}")

    for symbol in range(16):
        path = os.path.join(OUT_DIR, f"symbol_{symbol:02d}.pt")
        train(symbol, pulses, path)
        bits = f"{symbol:04b}"
        events = CODEBOOK[symbol]
        print(
            f"symbol={symbol:02d} bits={bits} "
            f"events={events} -> {path}"
        )


if __name__ == "__main__":
    main()