from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "artifacts" / "controlled_temporal"
NAMES = ["normal", "event_A", "event_B", "event_C", "event_D"]


def load(name):
    return torch.load(DATA_DIR / f"{name}.pt", map_location="cpu", weights_only=False)


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
    title = section.upper() if section == "model" else state_name.upper()
    print(f"\n{title}\n" + "-" * 72)
    print(f"{'':10s}" + "".join(f"{name:>12s}" for name in NAMES))

    for a in NAMES:
        row = f"{a:10s}"

        for b in NAMES:
            value = difference(checkpoints[a], checkpoints[b], section, state_name)
            row += f"{value:12.6f}"

        print(row)


def print_normal_relative(checkpoints):
    normal = checkpoints["normal"]
    print("\nNORMAL-RELATIVE\n" + "-" * 72)
    print(f"{'event':10s}{'model':>14s}{'exp_avg':>14s}{'exp_avg_sq':>14s}")

    for event in ["A", "B", "C", "D"]:
        checkpoint = checkpoints[f"event_{event}"]
        model = difference(normal, checkpoint, "model")
        exp_avg = difference(normal, checkpoint, "optimizer", "exp_avg")
        exp_avg_sq = difference(normal, checkpoint, "optimizer", "exp_avg_sq")
        print(f"{event:10s}{model:14.8f}{exp_avg:14.8f}{exp_avg_sq:14.8f}")


def main():
    checkpoints = {name: load(name) for name in NAMES}

    print_normal_relative(checkpoints)
    print_matrix(checkpoints, "model")
    print_matrix(checkpoints, "optimizer", "exp_avg")
    print_matrix(checkpoints, "optimizer", "exp_avg_sq")


if __name__ == "__main__":
    main()