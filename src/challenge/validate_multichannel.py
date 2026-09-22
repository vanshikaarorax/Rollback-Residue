from pathlib import Path

import torch
import torch.nn as nn

from src.challenge.channels import create_channels
from src.challenge.codebook import CODEBOOK
from src.model import TinyMLP


SEED = 42
INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 4
SAMPLES = 256
STEPS = 100
LR = 1e-3
TARGET_PARAMETER = 0

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts" / "multichannel"


def create_dataset():
    torch.manual_seed(SEED)
    return torch.randn(SAMPLES, INPUT_DIM), torch.randint(0, OUTPUT_DIM, (SAMPLES,))


def get_batch(x, y, step):
    start = ((step - 1) * 8) % SAMPLES
    indices = torch.arange(start, start + 8) % SAMPLES
    return x[indices], y[indices]


def train(symbol):
    torch.manual_seed(SEED)

    model = TinyMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    x, y = create_dataset()
    parameter = list(model.parameters())[TARGET_PARAMETER]
    channels = create_channels(tuple(parameter.shape))

    events = CODEBOOK[symbol]
    step_to_channel = {40: 0, 50: 1, 60: 2, 70: 3}

    for step in range(1, STEPS + 1):
        batch_x, batch_y = get_batch(x, y, step)

        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()

        if step in events:
            channel = step_to_channel[step]
            parameter.grad.add_(channels[channel])

        optimizer.step()

    return {
        "model": {name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
        "optimizer": {
            parameter_id: {name: value.detach().clone() if torch.is_tensor(value) else value for name, value in state.items()}
            for parameter_id, state in optimizer.state_dict()["state"].items()
        },
        "metadata": {"symbol": symbol, "events": events}
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for symbol in CODEBOOK:
        checkpoint = train(symbol)
        path = OUTPUT_DIR / f"symbol_{symbol:02d}.pt"
        torch.save(checkpoint, path)
        print(f"symbol={symbol:02d} bits={symbol:04b} events={CODEBOOK[symbol]} -> {path}")


if __name__ == "__main__":
    main()