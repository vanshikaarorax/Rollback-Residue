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

PULSE_SCALE = 0.05
TARGET_PARAMETER = 0

EVENTS = {
    "A": [40, 41],
    "B": [40, 43],
    "C": [40, 46],
    "D": [40, 50],
}

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts" / "temporal"


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


def apply_pulse(model):
    parameter = list(model.parameters())[TARGET_PARAMETER]

    with torch.no_grad():
        parameter.grad.mul_(PULSE_SCALE)

    return parameter.grad.norm().item()


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

    event_steps = set(
        EVENTS[event]
        if event is not None
        else []
    )

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

        if step in event_steps:
            grad_norm = apply_pulse(model)

            print(
                f"event={event or 'normal'} "
                f"step={step:03d} "
                f"grad_norm={grad_norm:.8f}"
            )

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
            "event_steps": (
                EVENTS[event]
                if event is not None
                else []
            ),
            "pulse_scale": PULSE_SCALE,
            "target_parameter": TARGET_PARAMETER,
        },
    }


def save_checkpoint(name, checkpoint):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = OUTPUT_DIR / f"{name}.pt"

    torch.save(
        checkpoint,
        path,
    )

    return path


def main():
    for event in [None, "A", "B", "C", "D"]:
        name = (
            "normal"
            if event is None
            else f"event_{event}"
        )

        path = save_checkpoint(
            name,
            train(event),
        )

        print(
            f"{name:10s} -> {path}"
        )


if __name__ == "__main__":
    main()