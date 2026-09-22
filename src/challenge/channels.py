import torch


SEED = 777
PULSE_SCALE = 0.01
NUM_CHANNELS = 4


def create_channels(shape):
    generator = torch.Generator().manual_seed(SEED)
    vectors = torch.randn(NUM_CHANNELS, *shape, generator=generator)
    norms = vectors.flatten(1).norm(dim=1).view(NUM_CHANNELS, 1, 1)
    return vectors / norms * PULSE_SCALE