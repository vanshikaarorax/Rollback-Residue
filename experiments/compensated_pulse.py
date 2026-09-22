from pathlib import Path

import torch
import torch.nn as nn

from src.model import TinyMLP


SEED = 42
INPUT_DIM = 16
HIDDEN_DIM = 32
OUTPUT_DIM = 4
SAMPLES = 256
STEPS = 120
LR = 1e-3

EVENT_STEP = 40
COMPENSATION_STEP = 41
PULSE_SCALE = 5.0

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts" / "compensated"


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


def apply_pulse(model, scale):
    total_norm = 0.0

    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
                total_norm += parameter.grad.norm().item() ** 2

    print(
        f"pulse_scale={scale:+.1f} "
        f"grad_norm={total_norm ** 0.5:.8f}"
    )


def clone_state(state):
    return {
        name: value.detach().clone()
        if torch.is_tensor(value)
        else value
        for name, value in state.items()
    }


def clone_optimizer_state(optimizer):
    return {
        parameter_id: clone_state(state)
        for parameter_id, state
        in optimizer.state_dict()["state"].items()
    }


def train(mode):
    set_seed()

    model = TinyMLP(
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    criterion = nn.CrossEntropyLoss()
    x, y = create_dataset()

    event_batch = get_batch(
        x,
        y,
        EVENT_STEP,
    )

    for step in range(1, STEPS + 1):
        if mode == "compensated" and step in {
            EVENT_STEP,
            COMPENSATION_STEP,
        }:
            batch_x, batch_y = event_batch
        else:
            batch_x, batch_y = get_batch(
                x,
                y,
                step,
            )

        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()

        if mode == "positive" and step == EVENT_STEP:
            apply_pulse(
                model,
                PULSE_SCALE,
            )

        if mode == "compensated":
            if step == EVENT_STEP:
                apply_pulse(
                    model,
                    PULSE_SCALE,
                )
            elif step == COMPENSATION_STEP:
                apply_pulse(
                    model,
                    -PULSE_SCALE,
                )

        optimizer.step()

        if step in {
            EVENT_STEP,
            COMPENSATION_STEP,
        }:
            print(
                f"mode={mode:12s} "
                f"step={step:03d} "
                f"loss={loss.item():.6f}"
            )

    return {
        "model": {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        },
        "optimizer": clone_optimizer_state(
            optimizer
        ),
        "metadata": {
            "mode": mode,
            "event_step": EVENT_STEP,
            "compensation_step": (
                COMPENSATION_STEP
                if mode == "compensated"
                else None
            ),
            "pulse_scale": PULSE_SCALE,
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
    for mode in [
        "normal",
        "positive",
        "compensated",
    ]:
        path = save_checkpoint(
            mode,
            train(mode),
        )

        print(f"{mode}: {path}")


if __name__ == "__main__":
    main()