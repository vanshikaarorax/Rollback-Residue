from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "artifacts" / "compensated"

MODES = [
    "normal",
    "positive",
    "compensated",
]


def load(name):
    path = DATA_DIR / f"{name}.pt"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def model_difference(a, b):
    total = 0.0

    for name in a["model"]:
        delta = (
            a["model"][name].float()
            - b["model"][name].float()
        )

        total += delta.pow(2).sum().item()

    return total ** 0.5


def optimizer_difference(a, b, state_name):
    total = 0.0

    for parameter_id in a["optimizer"]:
        a_state = a["optimizer"][parameter_id]
        b_state = b["optimizer"][parameter_id]

        if state_name not in a_state:
            continue

        delta = (
            a_state[state_name].float()
            - b_state[state_name].float()
        )

        total += delta.pow(2).sum().item()

    return total ** 0.5


def compare(label, a, b):
    model = model_difference(a, b)
    exp_avg = optimizer_difference(
        a,
        b,
        "exp_avg",
    )
    exp_avg_sq = optimizer_difference(
        a,
        b,
        "exp_avg_sq",
    )

    print(
        f"{label:25s} "
        f"{model:14.8f} "
        f"{exp_avg:14.8f} "
        f"{exp_avg_sq:14.8f}"
    )


def inspect_metadata(checkpoints):
    print("\nCHECKPOINT METADATA")
    print("-" * 50)

    for name, checkpoint in checkpoints.items():
        print(
            f"{name:12s} "
            f"{checkpoint['metadata']}"
        )


def main():
    checkpoints = {
        mode: load(mode)
        for mode in MODES
    }

    inspect_metadata(checkpoints)

    print("\nCOMPARISON")
    print("-" * 72)

    print(
        f"{'comparison':25s} "
        f"{'model':>14s} "
        f"{'exp_avg':>14s} "
        f"{'exp_avg_sq':>14s}"
    )

    print("-" * 72)

    compare(
        "normal vs positive",
        checkpoints["normal"],
        checkpoints["positive"],
    )

    compare(
        "normal vs compensated",
        checkpoints["normal"],
        checkpoints["compensated"],
    )

    compare(
        "positive vs compensated",
        checkpoints["positive"],
        checkpoints["compensated"],
    )


if __name__ == "__main__":
    main()