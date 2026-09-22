import torch

NUM_EVENTS = 16
SEED = 900
EVENT_STEP = 40
WINDOW = 5


def make_events(samples=256, input_dim=16, output_dim=4):
    events = {}
    for event in range(NUM_EVENTS):
        g = torch.Generator().manual_seed(SEED + event)
        x = torch.randn(samples, input_dim, generator=g)
        shift = (event - 7.5) * 0.35
        x = x + shift
        y = ((x[:, 0] * (event + 1) + x[:, 1]) > 0).long() % output_dim
        events[event] = (x, y)
    return events


def make_normal(samples=256, input_dim=16, output_dim=4):
    g = torch.Generator().manual_seed(SEED - 1)
    x = torch.randn(samples, input_dim, generator=g)
    y = torch.randint(0, output_dim, (samples,), generator=g)
    return x, y