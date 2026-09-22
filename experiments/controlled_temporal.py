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
PULSE_SCALE = 0.01
TARGET_PARAMETER = 0

EVENTS = {"A": [40, 41], "B": [40, 43], "C": [40, 46], "D": [40, 50]}

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts" / "controlled_temporal"


def set_seed():
    torch.manual_seed(SEED)


def create_dataset():
    return torch.randn(SAMPLES, INPUT_DIM), torch.randint(0, OUTPUT_DIM, (SAMPLES,))


def get_batch(x, y, step):
    start = ((step - 1) * 8) % SAMPLES
    indices = torch.arange(start, start + 8) % SAMPLES
    return x[indices], y[indices]


def create_model():
    return TinyMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)


def create_pulse_vector(model):
    parameter = list(model.parameters())[TARGET_PARAMETER]
    generator = torch.Generator().manual_seed(777)
    return torch.randn(parameter.shape, generator=generator) * PULSE_SCALE


def train(event=None):
    set_seed()
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    x, y = create_dataset()
    pulse_vector = create_pulse_vector(model)
    event_steps = set(EVENTS[event] if event else [])

    for step in range(1, STEPS + 1):
        batch_x, batch_y = get_batch(x, y, step)
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()

        if step in event_steps:
            parameter = list(model.parameters())[TARGET_PARAMETER]
            parameter.grad.add_(pulse_vector)
            print(f"event={event} step={step:03d} pulse_norm={pulse_vector.norm().item():.8f}")

        optimizer.step()

    checkpoint = {
        "model": {name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
        "optimizer": {
            parameter_id: {name: value.detach().clone() if torch.is_tensor(value) else value for name, value in state.items()}
            for parameter_id, state in optimizer.state_dict()["state"].items()
        },
        "metadata": {"event": event, "event_steps": EVENTS[event] if event else [], "pulse_scale": PULSE_SCALE, "target_parameter": TARGET_PARAMETER}
    }

    return checkpoint


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for event in [None, "A", "B", "C", "D"]:
        name = "normal" if event is None else f"event_{event}"
        path = OUTPUT_DIR / f"{name}.pt"
        torch.save(train(event), path)
        print(f"{name:10s} -> {path}")


if __name__ == "__main__":
    main()