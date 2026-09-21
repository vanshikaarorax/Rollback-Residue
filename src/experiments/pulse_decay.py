from pathlib import Path

import torch
import torch.nn as nn

from src.model import TinyMLP


SEED = 42
INPUT_DIM = 16
HIDDEN_DIM = 32
OUTPUT_DIM = 4
SAMPLES = 256
LR = 1e-3

PULSE_STEP = 40
PULSE_SCALE = 5.0

DELAYS = [0, 1, 2, 5, 10, 20, 40, 80]

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts" / "decay"


def set_seed():
    torch.manual_seed(SEED)


def create_dataset():
    x = torch.randn(SAMPLES, INPUT_DIM)
    y = torch.randint(0, OUTPUT_DIM, (SAMPLES,))
    return x, y


def create_model():
    return TinyMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)


def get_batch(x, y, step):
    start = ((step - 1) * 8) % SAMPLES
    indices = torch.arange(start, start + 8) % SAMPLES

    return x[indices], y[indices]


def apply_pulse(model):
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(PULSE_SCALE)


def clone_model_state(model):
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }


def clone_optimizer_state(optimizer):
    return {
        parameter_id: {
            name: value.detach().clone()
            if torch.is_tensor(value)
            else value
            for name, value in state.items()
        }
        for parameter_id, state in optimizer.state_dict()["state"].items()
    }


def train(use_pulse, observation_step):
    set_seed()

    model = create_model()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )
    criterion = nn.CrossEntropyLoss()

    x, y = create_dataset()

    for step in range(1, observation_step + 1):
        batch_x, batch_y = get_batch(x, y, step)

        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()

        if use_pulse and step == PULSE_STEP:
            apply_pulse(model)

        optimizer.step()

    return {
        "model": clone_model_state(model),
        "optimizer": clone_optimizer_state(optimizer),
        "metadata": {
            "pulse": use_pulse,
            "pulse_step": PULSE_STEP if use_pulse else None,
            "observation_step": observation_step,
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
    for delay in DELAYS:
        observation_step = PULSE_STEP + delay

        normal = train(
            use_pulse=False,
            observation_step=observation_step,
        )

        pulse = train(
            use_pulse=True,
            observation_step=observation_step,
        )

        normal_path = save_checkpoint(
            f"delay_{delay}_normal",
            normal,
        )

        pulse_path = save_checkpoint(
            f"delay_{delay}_pulse",
            pulse,
        )

        print(
            f"delay={delay:02d} "
            f"step={observation_step:03d} "
            f"normal={normal_path} "
            f"pulse={pulse_path}"
        )


if __name__ == "__main__":
    main()