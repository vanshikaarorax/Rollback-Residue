from pathlib import Path

import torch
import torch.nn as nn

from src.model import TinyMLP


SEED = 42
INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 4
SAMPLES = 256
STEPS = 120
LR = 1e-3

EVENT_STEP = 40
PULSE_SCALE = 0.5

EVENTS = {
    "A": 0,
    "B": 2,
    "C": 4,
    "D": 5,
}

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts" / "localized"


def set_seed():
    torch.manual_seed(SEED)


def create_dataset():
    x = torch.randn(SAMPLES, INPUT_DIM)
    y = torch.randint(0, OUTPUT_DIM, (SAMPLES,))
    return x, y


def get_batch(x, y, step):
    start = ((step - 1) * 8) % SAMPLES
    indices = torch.arange(start, start + 8) % SAMPLES
    return x[indices], y[indices]


def create_model():
    return TinyMLP(
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
    )


def apply_event(model, event):
    parameter = list(model.parameters())[EVENTS[event]]

    with torch.no_grad():
        parameter.grad.mul_(PULSE_SCALE)


def clone_optimizer_state(optimizer):
    return {
        parameter_id: {
            name: value.detach().clone()
            if torch.is_tensor(value)
            else value
            for name, value in state.items()
        }
        for parameter_id, state
        in optimizer.state_dict()["state"].items()
    }


def train(event=None):
    set_seed()

    model = create_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    criterion = nn.CrossEntropyLoss()
    x, y = create_dataset()

    for step in range(1, STEPS + 1):
        batch_x, batch_y = get_batch(
            x,
            y,
            step,
        )

        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()

        if event is not None and step == EVENT_STEP:
            apply_event(model, event)

        optimizer.step()

    return {
        "model": {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        },
        "optimizer": clone_optimizer_state(
            optimizer
        ),
        "metadata": {
            "event": event,
            "event_step": EVENT_STEP,
            "pulse_scale": PULSE_SCALE,
            "target_parameter": (
                EVENTS[event]
                if event is not None
                else None
            ),
        },
    }


def save_checkpoint(name, checkpoint):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = OUTPUT_DIR / f"{name}.pt"
    torch.save(checkpoint, path)

    return path


def main():
    for event in [None, "A", "B", "C", "D"]:
        name = "normal" if event is None else f"event_{event}"
        path = save_checkpoint(
            name,
            train(event),
        )

        print(
            f"{name:10s} -> {path}"
        )


if __name__ == "__main__":
    main()