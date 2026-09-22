import os
import torch

BASE_DIR = "artifacts/sequential"


def get_avg(path):
    checkpoint = torch.load(path, map_location="cpu")
    tensors = []
    for state in checkpoint["optimizer"]["state"].values():
        if "exp_avg" in state:
            tensors.append(state["exp_avg"].float().flatten())
    return torch.cat(tensors)


def distance(a, b):
    return torch.norm(a - b).item()


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    clean = get_avg(os.path.join(BASE_DIR, "clean.pt"))
    events = [get_avg(os.path.join(BASE_DIR, f"event_{i:02d}.pt")) for i in range(16)]
    deltas = [x - clean for x in events]

    print()
    print("SEQUENTIAL EVENT FORENSICS")
    print("=" * 100)

    print("\nDISTANCE FROM CLEAN")
    print("-" * 100)
    for i, delta in enumerate(deltas):
        print(f"{i:02d} {i:04b} norm={delta.norm().item():.6f}")

    print("\nNEAREST EVENT")
    print("-" * 100)

    correct = 0

    for i, delta in enumerate(deltas):
        distances = [distance(delta, other) if i != j else float("inf") for j, other in enumerate(deltas)]
        nearest = min(range(16), key=lambda j: distances[j])

        if nearest == i:
            correct += 1

        print(f"{i:02d} {i:04b} -> nearest={nearest:02d} distance={distances[nearest]:.6f}")

    print(f"\nSELF-IDENTIFICATION: {correct}/16 = {correct / 16:.1%}")

    print("\nPAIRWISE COSINE")
    print("-" * 100)

    for i in range(16):
        row = []
        for j in range(16):
            if i == j:
                continue
            row.append((cosine(deltas[i], deltas[j]), j))
        row.sort()
        nearest = ", ".join(f"{j:02d}={score:.3f}" for score, j in row[:3])
        print(f"{i:02d}: {nearest}")


if __name__ == "__main__":
    main()