from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "artifacts" / "temporal"

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
    print("-" * 65)

    for name, checkpoint in checkpoints.items():
        print(
            f"{name:10s} "
            f"{checkpoint['metadata']}"
        )


def print_matrix(checkpoints, metric):
    print(f"\n{metric.upper()}")
    print("-" * 72)

    print(
        f"{'':10s}"
        + "".join(
            f"{name:>12s}"
            for name in NAMES
        )
    )

    for name_a in NAMES:
        row = f"{name_a:10s}"

        for name_b in NAMES:
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

            row += f"{value:12.6f}"

        print(row)


def print_normal_relative(checkpoints):
    print("\nNORMAL-RELATIVE")
    print("-" * 72)

    print(
        f"{'event':10s}"
        f"{'model':>14s}"
        f"{'exp_avg':>14s}"
        f"{'exp_avg_sq':>14s}"
    )

    normal = checkpoints["normal"]

    for event in ["A", "B", "C", "D"]:
        checkpoint = checkpoints[f"event_{event}"]

        model = model_difference(
            normal,
            checkpoint,
        )

        exp_avg = optimizer_difference(
            normal,
            checkpoint,
            "exp_avg",
        )

        exp_avg_sq = optimizer_difference(
            normal,
            checkpoint,
            "exp_avg_sq",
        )

        print(
            f"{event:10s}"
            f"{model:14.8f}"
            f"{exp_avg:14.8f}"
            f"{exp_avg_sq:14.8f}"
        )


def main():
    checkpoints = {
        name: load(name)
        for name in NAMES
    }

    print_metadata(checkpoints)

    print_normal_relative(checkpoints)

    print_matrix(
        checkpoints,
        "model",
    )

    print_matrix(
        checkpoints,
        "exp_avg",
    )

    print_matrix(
        checkpoints,
        "exp_avg_sq",
    )


if __name__ == "__main__":
    main()