from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "artifacts" / "codebook"
SYMBOLS = list(range(16))


def load(symbol):
    return torch.load(DATA_DIR / f"symbol_{symbol:02d}.pt", map_location="cpu", weights_only=False)


def difference(a, b, section, state_name=None):
    total = 0.0

    if section == "model":
        for name in a["model"]:
            delta = a["model"][name].float() - b["model"][name].float()
            total += delta.pow(2).sum().item()
    else:
        for parameter_id in a["optimizer"]:
            a_state = a["optimizer"][parameter_id]
            b_state = b["optimizer"][parameter_id]

            if state_name not in a_state:
                continue

            delta = a_state[state_name].float() - b_state[state_name].float()
            total += delta.pow(2).sum().item()

    return total ** 0.5


def print_matrix(checkpoints, section, state_name=None):
    title = "MODEL" if section == "model" else state_name.upper()
    print(f"\n{title}\n" + "-" * 110)
    print("      " + "".join(f"{symbol:>8d}" for symbol in SYMBOLS))

    for a in SYMBOLS:
        row = f"{a:4d} "

        for b in SYMBOLS:
            value = difference(checkpoints[a], checkpoints[b], section, state_name)
            row += f"{value:8.4f}"

        print(row)


def print_nearest_neighbors(checkpoints, section, state_name=None):
    title = "MODEL" if section == "model" else state_name.upper()
    print(f"\nNEAREST NEIGHBORS — {title}\n" + "-" * 60)

    for symbol in SYMBOLS:
        distances = []

        for other in SYMBOLS:
            if symbol == other:
                continue

            distance = difference(checkpoints[symbol], checkpoints[other], section, state_name)
            distances.append((distance, other))

        distances.sort()

        print(f"{symbol:02d}: " + ", ".join(f"{other:02d}={distance:.6f}" for distance, other in distances[:3]))


def main():
    checkpoints = {symbol: load(symbol) for symbol in SYMBOLS}

    print_matrix(checkpoints, "model")
    print_matrix(checkpoints, "optimizer", "exp_avg")
    print_matrix(checkpoints, "optimizer", "exp_avg_sq")

    print_nearest_neighbors(checkpoints, "model")
    print_nearest_neighbors(checkpoints, "optimizer", "exp_avg")


if __name__ == "__main__":
    main()