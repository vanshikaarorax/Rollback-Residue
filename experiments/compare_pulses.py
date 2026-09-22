from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
PULSE_DIR = ROOT / "artifacts" / "pulse"


def load_checkpoint(name):
    path = PULSE_DIR / f"{name}.pt"

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def flatten_tensors(state_dict):
    tensors = []

    for value in state_dict.values():
        if torch.is_tensor(value):
            tensors.append(value.detach().float().flatten())

    return torch.cat(tensors)


def model_difference(a, b):
    a_weights = flatten_tensors(a["model"])
    b_weights = flatten_tensors(b["model"])

    delta = a_weights - b_weights

    return {
        "l2": delta.norm().item(),
        "mean_abs": delta.abs().mean().item(),
        "max_abs": delta.abs().max().item(),
        "cosine": F.cosine_similarity(
            a_weights.unsqueeze(0),
            b_weights.unsqueeze(0),
        ).item(),
    }


def optimizer_difference(a, b, state_name):
    deltas = []
    a_values = []
    b_values = []

    for parameter_id in a["optimizer"]["state"]:
        a_state = a["optimizer"]["state"][parameter_id]
        b_state = b["optimizer"]["state"][parameter_id]

        if state_name not in a_state:
            continue

        a_tensor = a_state[state_name].float().flatten()
        b_tensor = b_state[state_name].float().flatten()

        a_values.append(a_tensor)
        b_values.append(b_tensor)

        deltas.append(a_tensor - b_tensor)

    a_values = torch.cat(a_values)
    b_values = torch.cat(b_values)
    delta = torch.cat(deltas)

    return {
        "l2": delta.norm().item(),
        "mean_abs": delta.abs().mean().item(),
        "max_abs": delta.abs().max().item(),
        "cosine": F.cosine_similarity(
            a_values.unsqueeze(0),
            b_values.unsqueeze(0),
        ).item(),
    }


def print_result(label, result):
    print(f"\n{label}")
    print("-" * 50)

    for key, value in result.items():
        print(f"{key:12s}: {value:.10f}")


def main():
    normal = load_checkpoint("normal")
    positive = load_checkpoint("positive")
    negative = load_checkpoint("negative")

    print("MODEL DIFFERENCES")

    print_result(
        "normal vs positive",
        model_difference(normal, positive),
    )

    print_result(
        "normal vs negative",
        model_difference(normal, negative),
    )

    print_result(
        "positive vs negative",
        model_difference(positive, negative),
    )

    for state_name in ["exp_avg", "exp_avg_sq"]:
        print(f"\nOPTIMIZER STATE: {state_name}")

        print_result(
            "normal vs positive",
            optimizer_difference(
                normal,
                positive,
                state_name,
            ),
        )

        print_result(
            "normal vs negative",
            optimizer_difference(
                normal,
                negative,
                state_name,
            ),
        )

        print_result(
            "positive vs negative",
            optimizer_difference(
                positive,
                negative,
                state_name,
            ),
        )


if __name__ == "__main__":
    main()