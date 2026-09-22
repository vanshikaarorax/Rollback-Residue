import torch

TARGET_PARAM = "network.0.weight"
CHANNEL_STEPS = [40, 50, 60, 70]
NUM_CHANNELS = 4
SEED = 777
PULSE_SCALE = 0.05


def channel_bounds(numel):
    if numel % NUM_CHANNELS != 0:
        raise ValueError(f"Parameter size {numel} is not divisible by {NUM_CHANNELS}.")
    width = numel // NUM_CHANNELS
    return [(i * width, (i + 1) * width) for i in range(NUM_CHANNELS)]


def create_channel_pulses(shape, scale=PULSE_SCALE):
    numel = 1
    for dim in shape:
        numel *= dim

    bounds = channel_bounds(numel)
    generator = torch.Generator().manual_seed(SEED)
    pulses = []

    for start, end in bounds:
        vector = torch.randn(end - start, generator=generator)
        vector = vector / vector.norm()
        pulses.append(vector * scale)

    return pulses


def apply_channel_pulse(grad, channel, pulses):
    flat = grad.reshape(-1)
    start, end = channel_bounds(flat.numel())[channel]
    flat[start:end] += pulses[channel]