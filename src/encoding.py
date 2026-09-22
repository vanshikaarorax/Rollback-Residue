import torch


BUFFER_SIZE = 8


def byte_to_bits(value: int) -> torch.Tensor:
    if not 0 <= value <= 255:
        raise ValueError("value must be in range 0..255")

    return torch.tensor(
        [(value >> (7 - i)) & 1 for i in range(BUFFER_SIZE)],
        dtype=torch.float32,
    )


def bits_to_byte(bits) -> int:
    if len(bits) != BUFFER_SIZE:
        raise ValueError("expected exactly 8 bits")

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)

    return value