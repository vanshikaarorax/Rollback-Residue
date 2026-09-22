import torch

from src.forensic import train_checkpoint
from src.encoding import byte_to_bits


def get_buffer_state(checkpoint):
    for state in checkpoint["optimizer"]["state"].values():
        if state.get("exp_avg") is not None and state["exp_avg"].numel() == 8:
            return state["exp_avg"]

    raise AssertionError("forensic state not found")


def test_deterministic():
    a = train_checkpoint(secret=65)
    b = train_checkpoint(secret=65)

    state_a = get_buffer_state(a)
    state_b = get_buffer_state(b)

    assert torch.equal(state_a, state_b)


def test_secret_changes_optimizer_state():
    clean = train_checkpoint()
    encoded = train_checkpoint(secret=65)

    clean_state = get_buffer_state(clean)
    encoded_state = get_buffer_state(encoded)

    assert not torch.equal(clean_state, encoded_state)