from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "artifacts" / "baseline.pt"


def load_checkpoint():
    return torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )


def tensor_stats(tensor):
    return (
        f"shape={str(tuple(tensor.shape)):15s} "
        f"mean={tensor.mean().item(): .6f} "
        f"std={tensor.std(unbiased=False).item(): .6f} "
        f"min={tensor.min().item(): .6f} "
        f"max={tensor.max().item(): .6f}"
    )


def inspect_model(checkpoint):
    print("\nMODEL")
    print("-" * 50)

    for name, tensor in checkpoint["model"].items():
        print(f"{name:25s} {tensor_stats(tensor)}")


def inspect_optimizer(checkpoint):
    optimizer_state = checkpoint["optimizer"]

    print("\nOPTIMIZER")
    print("-" * 50)
    print(f"param_groups: {len(optimizer_state['param_groups'])}")
    print(f"state_entries: {len(optimizer_state['state'])}")

    for parameter_id, state in optimizer_state["state"].items():
        print(f"\nparameter_id={parameter_id}")

        for name, value in state.items():
            if torch.is_tensor(value):
                if value.numel() == 1:
                    print(
                        f"  {name:12s} "
                        f"value={value.item():.6f}"
                    )
                else:
                    print(
                        f"  {name:12s} "
                        f"{tensor_stats(value)}"
                    )
            else:
                print(f"  {name:12s} value={value}")


def inspect_config(checkpoint):
    print("\nCONFIG")
    print("-" * 50)

    for key, value in checkpoint["config"].items():
        print(f"{key:15s} {value}")


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run `python -m src.train` first."
        )

    checkpoint = load_checkpoint()

    print(f"checkpoint={CHECKPOINT_PATH}")

    inspect_model(checkpoint)
    inspect_optimizer(checkpoint)
    inspect_config(checkpoint)


if __name__ == "__main__":
    main()