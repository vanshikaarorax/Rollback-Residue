from pathlib import Path

import torch
import torch.nn as nn

from src.challenge.codebook import CODEBOOK
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

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts" / "codebook"


def create_dataset():
    torch.manual_seed(SEED)
    return torch.randn(SAMPLES, INPUT_DIM), torch.randint(0, OUTPUT_DIM, (SAMPLES,))


def create_pulse_vector(model):
    parameter = list(model.parameters())[TARGET_PARAMETER]
    generator = torch.Generator().manual_seed(777)
    return torch.randn(parameter.shape, generator=generator) * PULSE_SCALE


def get_batch(x, y, step):
    start = ((step - 1) * 8) % SAMPLES
    indices = torch.arange(start, start + 8) % SAMPLES
    return x[indices], y[indices]


def train(event_steps):
    torch.manual_seed(SEED)

    model = TinyMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    x, y = create_dataset()
    pulse_vector = create_pulse_vector(model)

    for step in range(1, STEPS + 1):
        batch_x, batch_y = get_batch(x, y, step)

        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()

        if step in event_steps:
            parameter = list(model.parameters())[TARGET_PARAMETER]
            parameter.grad.add_(pulse_vector)

        optimizer.step()

    return {
        "model": {name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
        "optimizer": {
            parameter_id: {name: value.detach().clone() if torch.is_tensor(value) else value for name, value in state.items()}
            for parameter_id, state in optimizer.state_dict()["state"].items()
        }
    }


def save_checkpoints():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for symbol, event_steps in CODEBOOK.items():
        checkpoint = train(event_steps)
        path = OUTPUT_DIR / f"symbol_{symbol:02d}.pt"
        torch.save(checkpoint, path)
        print(f"symbol={symbol:02d} events={event_steps} -> {path}")


if __name__ == "__main__":
    save_checkpoints()