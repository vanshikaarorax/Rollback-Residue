from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "artifacts" / "multichannel"
SYMBOLS = list(range(16))


def load(symbol):
    return torch.load(DATA_DIR / f"symbol_{symbol:02d}.pt", map_location="cpu", weights_only=False)


def difference(a, b, section):
    total = 0.0

    if section == "model":
        for name in a["model"]:
            delta = a["model"][name].float() - b["model"][name].float()
            total += delta.pow(2).sum().item()
    else:
        for parameter_id in a["optimizer"]:
            a_state = a["optimizer"][parameter_id]
            b_state = b["optimizer"][parameter_id]
            delta = a_state[section].float() - b_state[section].float()
            total += delta.pow(2).sum().item()

    return total ** 0.5


def nearest_neighbors(checkpoints, section):
    print(f"\nNEAREST NEIGHBORS — {section.upper()}")
    print("-" * 65)

    for symbol in SYMBOLS:
        distances = []

        for other in SYMBOLS:
            if symbol == other:
                continue

            distance = difference(checkpoints[symbol], checkpoints[other], section)
            distances.append((distance, other))

        distances.sort()

        nearest = ", ".join(f"{other:02d}={distance:.6f}" for distance, other in distances[:5])
        print(f"{symbol:02d} ({symbol:04b}): {nearest}")


def main():
    checkpoints = {symbol: load(symbol) for symbol in SYMBOLS}

    nearest_neighbors(checkpoints, "model")
    nearest_neighbors(checkpoints, "exp_avg")
    nearest_neighbors(checkpoints, "exp_avg_sq")


if __name__ == "__main__":
    main()