from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DECAY_DIR = ROOT / "artifacts" / "decay"

DELAYS = [0, 1, 2, 5, 10, 20, 40, 80]


def load_checkpoint(delay, kind):
    path = DECAY_DIR / f"delay_{delay}_{kind}.pt"

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def tensor_difference(a, b):
    delta = a.float() - b.float()

    return {
        "l2": delta.norm().item(),
        "mean_abs": delta.abs().mean().item(),
        "max_abs": delta.abs().max().item(),
    }


def model_difference(normal, pulse):
    results = []

    for name in normal["model"]:
        result = tensor_difference(
            normal["model"][name],
            pulse["model"][name],
        )

        results.append(result)

    return aggregate(results)


def optimizer_difference(normal, pulse, state_name):
    results = []

    normal_state = normal["optimizer"]
    pulse_state = pulse["optimizer"]

    for parameter_id in normal_state:
        if state_name not in normal_state[parameter_id]:
            continue

        result = tensor_difference(
            normal_state[parameter_id][state_name],
            pulse_state[parameter_id][state_name],
        )

        results.append(result)

    return aggregate(results)


def aggregate(results):
    return {
        "l2": sum(
            result["l2"] ** 2
            for result in results
        ) ** 0.5,
        "mean_abs": sum(
            result["mean_abs"]
            for result in results
        ) / len(results),
        "max_abs": max(
            result["max_abs"]
            for result in results
        ),
    }


def print_result(delay, model, exp_avg, exp_avg_sq):
    print(
        f"{delay:5d} "
        f"{model['l2']:14.8f} "
        f"{exp_avg['l2']:14.8f} "
        f"{exp_avg_sq['l2']:14.8f}"
    )


def main():
    print(
        f"{'delay':>5s} "
        f"{'model_l2':>14s} "
        f"{'exp_avg_l2':>14s} "
        f"{'exp_avg_sq_l2':>14s}"
    )

    print("-" * 55)

    for delay in DELAYS:
        normal = load_checkpoint(
            delay,
            "normal",
        )

        pulse = load_checkpoint(
            delay,
            "pulse",
        )

        model = model_difference(
            normal,
            pulse,
        )

        exp_avg = optimizer_difference(
            normal,
            pulse,
            "exp_avg",
        )

        exp_avg_sq = optimizer_difference(
            normal,
            pulse,
            "exp_avg_sq",
        )

        print_result(
            delay,
            model,
            exp_avg,
            exp_avg_sq,
        )


if __name__ == "__main__":
    main()