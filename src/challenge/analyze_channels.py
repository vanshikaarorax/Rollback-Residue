import torch

from src.challenge.channels import create_channels


SHAPE = (64, 16)


def main():
    channels = create_channels(SHAPE)

    print("CHANNEL NORMS")
    print("-" * 40)

    for index, vector in enumerate(channels):
        print(f"channel_{index}: {vector.norm().item():.8f}")

    print("\nCHANNEL COSINE SIMILARITY")
    print("-" * 40)

    flat = channels.flatten(1)
    normalized = flat / flat.norm(dim=1, keepdim=True)
    similarity = normalized @ normalized.T

    for row in similarity:
        print(" ".join(f"{value.item(): .4f}" for value in row))


if __name__ == "__main__":
    main()