import copy
import torch
import torch.nn as nn

from .encoding import byte_to_bits
from .model import TinyMLP


class AuxiliaryState(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.state = nn.Parameter(torch.zeros(size))

    def encode(self, bits):
        return torch.dot(self.state, bits)


def create_models(seed=42):
    torch.manual_seed(seed)
    return TinyMLP(), AuxiliaryState()


def create_data(seed=42, samples=256, input_dim=16, output_dim=4):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(samples, input_dim, generator=generator)
    y = torch.randint(0, output_dim, (samples,), generator=generator)
    return x, y


def train_checkpoint(
    secret=None,
    seed=42,
    steps=100,
    event_step=40,
    learning_rate=1e-3,
):
    x, y = create_data(seed)
    model, auxiliary = create_models(seed)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(auxiliary.parameters()),
        lr=learning_rate,
    )

    loss_fn = nn.CrossEntropyLoss()
    batch_size = 32

    for step in range(steps):
        start = (step % (len(x) // batch_size)) * batch_size
        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]

        optimizer.zero_grad()

        loss = loss_fn(model(xb), yb)
        loss = loss + 0.0 * auxiliary.state.sum()

        if step == event_step and secret is not None:
            bits = byte_to_bits(secret)
            loss = loss + auxiliary.encode(bits)

        loss.backward()

        if step == event_step and secret is not None:
            model_state = copy.deepcopy(model.state_dict())
            auxiliary_state = copy.deepcopy(auxiliary.state_dict())

            optimizer.step()

            model.load_state_dict(model_state)
            auxiliary.load_state_dict(auxiliary_state)
        else:
            optimizer.step()

    return {
        "model": model.state_dict(),
        "auxiliary": auxiliary.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            "seed": seed,
            "steps": steps,
            "event_step": event_step,
            "learning_rate": learning_rate,
        },
    }