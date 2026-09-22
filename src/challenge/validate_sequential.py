import os
import torch
import torch.nn as nn

from src.model import TinyMLP
from src.challenge.sequential_events import make_events, make_normal, EVENT_STEP, WINDOW

SEED = 42
INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 4
SAMPLES = 256
BATCH_SIZE = 32
STEPS = 120
LR = 1e-3
OUT_DIR = "artifacts/sequential"


def make_model():
    torch.manual_seed(SEED)
    return TinyMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)


def train(event_id, normal, events, path):
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    normal_x, normal_y = normal

    for step in range(STEPS):
        if EVENT_STEP <= step < EVENT_STEP + WINDOW and event_id is not None:
            x, y = events[event_id]
        else:
            x, y = normal_x, normal_y

        index = (step % (SAMPLES // BATCH_SIZE)) * BATCH_SIZE
        xb = x[index:index + BATCH_SIZE]
        yb = y[index:index + BATCH_SIZE]

        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            "steps": STEPS,
            "event_step": EVENT_STEP,
            "window": WINDOW,
            "event": event_id,
        },
    }, path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    normal = make_normal()
    events = make_events()

    train(None, normal, events, os.path.join(OUT_DIR, "clean.pt"))
    print("clean -> artifacts/sequential/clean.pt")

    for event_id in range(16):
        path = os.path.join(OUT_DIR, f"event_{event_id:02d}.pt")
        train(event_id, normal, events, path)
        print(f"event={event_id:02d} symbol={event_id:04b} -> {path}")


if __name__ == "__main__":
    main()