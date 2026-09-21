from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "artifacts" / "localized"

NAMES = [
    "normal",
    "event_A",
    "event_B",
    "event_C",
    "event_D",
]


def load(name):
    path = DATA_DIR / f"{name}.pt"

    if not path.exists():
        raise FileNotFoundError(path)

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


def print_metadata(checkpoints):
    print("\nEVENT METADATA")
    print("-" * 60)

    for name, checkpoint in checkpoints.items():
        print(
            f"{name:10s} "
            f"{checkpoint['metadata']}"
        )


def print_matrix(checkpoints, names, metric):
    print(f"\n{metric.upper()}")
    print("-" * 72)

    print(
        f"{'':10s}" +
        "".join(
            f"{name:>12s}"
            for name in names
        )
    )

    for name_a in names:
        row = f"{name_a:10s}"

        for name_b in names:
            if metric == "model":
                value = model_difference(
                    checkpoints[name_a],
                    checkpoints[name_b],
                )
            else:
                value = optimizer_difference(
                    checkpoints[name_a],
                    checkpoints[name_b],
                    metric,
                )

            row += f"{value:12.5f}"

        print(row)


def main():
    checkpoints = {
        name: load(name)
        for name in NAMES
    }

    print_metadata(checkpoints)

    print_matrix(
        checkpoints,
        NAMES,
        "model",
    )

    print_matrix(
        checkpoints,
        NAMES,
        "exp_avg",
    )

    print_matrix(
        checkpoints,
        NAMES,
        "exp_avg_sq",
    )


if __name__ == "__main__":
    main()