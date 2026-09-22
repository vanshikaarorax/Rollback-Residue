import torch

SEED = 777
NUM_REGIMES = 4
WINDOW = 5
REGIME_STEPS = [40, 50, 60, 70]


def make_regime_batches(samples=256, input_dim=16, output_dim=4):
    g = torch.Generator().manual_seed(SEED)
    normal_x = torch.randn(samples, input_dim, generator=g)
    normal_y = torch.randint(0, output_dim, (samples,), generator=g)

    regimes = {}

    g = torch.Generator().manual_seed(SEED + 1)
    x = torch.randn(samples, input_dim, generator=g)
    regimes[0] = (x, normal_y.clone())

    g = torch.Generator().manual_seed(SEED + 2)
    x = torch.randn(samples, input_dim, generator=g) + 2.0
    regimes[1] = (x, normal_y.clone())

    g = torch.Generator().manual_seed(SEED + 3)
    x = torch.randn(samples, input_dim, generator=g)
    y = torch.randint(0, output_dim, (samples,), generator=g)
    regimes[2] = (x, y)

    g = torch.Generator().manual_seed(SEED + 4)
    x = torch.randn(samples, input_dim, generator=g)
    y = (x[:, 0] > 0).long() % output_dim
    regimes[3] = (x, y)

    return (normal_x, normal_y), regimes


def event_for_step(step):
    for channel, start in enumerate(REGIME_STEPS):
        if start <= step < start + WINDOW:
            return channel
    return None