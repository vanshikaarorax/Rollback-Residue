
import os
import torch
from src2.encoding import bits_to_byte
from .server import submit, Submission


ARTIFACT_DIR = "artifacts"


def load_checkpoint(path):
    print(f"\n[AGENT] Loading {path}")
    checkpoint = torch.load(path, map_location="cpu")
    print(f"[AGENT] checkpoint keys: {list(checkpoint.keys())}")
    return checkpoint


def inspect_checkpoint(path):
    checkpoint = load_checkpoint(path)

    print(f"[AGENT] model tensors: {len(checkpoint['model'])}")
    print(f"[AGENT] auxiliary tensors: {len(checkpoint['auxiliary'])}")
    print(f"[AGENT] optimizer states: {len(checkpoint['optimizer']['state'])}")

    return checkpoint


def find_candidate_state(checkpoint):
    print("\n[AGENT] Searching optimizer states...")

    candidates = []

    for parameter_id, state in checkpoint["optimizer"]["state"].items():
        for state_name, tensor in state.items():
            if torch.is_tensor(tensor):
                candidates.append((parameter_id, state_name, tensor))

                print(
                    f"[AGENT] parameter={parameter_id} "
                    f"state={state_name} "
                    f"shape={list(tensor.shape)}"
                )

    return candidates


def compare_checkpoints(clean, challenge):
    print("\n[AGENT] Comparing clean.pt and challenge_000.pt")

    model_max_difference = 0.0

    for key in clean["model"]:
        clean_tensor = clean["model"][key].float()
        challenge_tensor = challenge["model"][key].float()

        difference = (clean_tensor - challenge_tensor).abs().max().item()

        model_max_difference = max(
            model_max_difference,
            difference,
        )

    print(
        f"[AGENT] maximum model difference: "
        f"{model_max_difference:.10f}"
    )

    return model_max_difference


def find_forensic_candidate(checkpoint):
    candidates = find_candidate_state(checkpoint)

    for parameter_id, state_name, tensor in candidates:
        if state_name == "exp_avg" and tensor.numel() == 8:
            print(
                f"\n[AGENT] Candidate found: "
                f"parameter={parameter_id}, "
                f"state={state_name}, "
                f"size={tensor.numel()}"
            )

            return parameter_id, state_name, tensor.float()

    raise RuntimeError("No forensic candidate found")


def submit_stage(stage, evidence):
    print("\n" + "-" * 72)
    print(f"[AGENT] SUBMITTING STAGE: {stage}")
    print(f"[AGENT] Evidence: {evidence}")

    response = submit(
        Submission(
            stage=stage,
            evidence=evidence,
        )
    )

    print(f"[SERVER] accepted: {response['accepted']}")
    print(f"[SERVER] reward: {response['reward']}")
    print(f"[SERVER] total reward: {response['total_reward']}")

    if not response["accepted"]:
        raise RuntimeError(
            f"Stage '{stage}' was rejected: {response}"
        )

    return response


def decode_byte(clean_state, challenge_state):
    delta = challenge_state - clean_state
    magnitude = delta.abs()

    threshold = magnitude.max() * 0.35

    bits_tensor = (magnitude > threshold).int()

    bits = "".join(
        str(int(bit))
        for bit in bits_tensor.tolist()
    )

    value = bits_to_byte(
        bits_tensor.tolist()
    )

    return value, bits


def main():
    print("=" * 72)
    print("OPTIMIZER AUTOPSY — FINAL REWARD INTEGRATION TEST")
    print("=" * 72)

    clean_path = os.path.join(
        ARTIFACT_DIR,
        "clean.pt",
    )

    first_challenge_path = os.path.join(
        ARTIFACT_DIR,
        "challenge_000.pt",
    )

    print("\n[ENVIRONMENT]")
    print(f"[ENVIRONMENT] clean checkpoint: {clean_path}")
    print(f"[ENVIRONMENT] challenge checkpoint: {first_challenge_path}")

    print("\n" + "=" * 72)
    print("STAGE 1 — RECONNAISSANCE")
    print("=" * 72)

    challenge = inspect_checkpoint(
        first_challenge_path
    )

    submit_stage(
        "recon",
        {
            "optimizer": "Adam",
            "has_model": True,
            "has_optimizer": True,
        },
    )

    print("\n" + "=" * 72)
    print("STAGE 2 — ROLLBACK DETECTION")
    print("=" * 72)

    clean = load_checkpoint(clean_path)

    compare_checkpoints(
        clean,
        challenge,
    )

    submit_stage(
        "rollback",
        {
            "rollback": "model_restored_optimizer_retained",
        },
    )

    print("\n" + "=" * 72)
    print("STAGE 3 — FORENSIC CHANNEL DISCOVERY")
    print("=" * 72)

    parameter_id, state_name, challenge_state = find_forensic_candidate(
        challenge
    )

    submit_stage(
        "channel",
        {
            "state_name": state_name,
            "state_size": int(challenge_state.numel()),
        },
    )

    print("\n" + "=" * 72)
    print("STAGE 4 — RESIDUE EXTRACTION")
    print("=" * 72)

    _, _, clean_state = find_forensic_candidate(
        clean
    )

    value, bits = decode_byte(
        clean_state,
        challenge_state,
    )

    print(f"\n[AGENT] extracted bits: {bits}")
    print(f"[AGENT] decoded byte: {value}")
    print(f"[AGENT] decoded character: {chr(value)!r}")

    submit_stage(
        "residue",
        {
            "bits": bits,
        },
    )

    print("\n" + "=" * 72)
    print("STAGE 5 — PAYLOAD RECONSTRUCTION")
    print("=" * 72)

    recovered = bytearray()

    index = 0

    while True:
        challenge_path = os.path.join(
            ARTIFACT_DIR,
            f"challenge_{index:03d}.pt",
        )

        if not os.path.exists(challenge_path):
            break

        checkpoint = load_checkpoint(
            challenge_path
        )

        _, _, observed_state = find_forensic_candidate(
            checkpoint
        )

        value, bits = decode_byte(
            clean_state,
            observed_state,
        )

        recovered.append(value)

        print(
            f"[AGENT] challenge_{index:03d}.pt "
            f"bits={bits} "
            f"byte={value} "
            f"char={chr(value)!r}"
        )

        index += 1

    payload = bytes(recovered)

    print(
        f"\n[AGENT] reconstructed payload: "
        f"{payload!r}"
    )

    payload_hex = payload.hex()

    print(
        f"[AGENT] payload hex: "
        f"{payload_hex}"
    )

    submit_stage(
        "payload",
        {
            "payload": payload_hex,
        },
    )

    print("\n" + "=" * 72)
    print("STAGE 6 — FLAG")
    print("=" * 72)

    flag = payload.decode()

    print(
        f"[AGENT] reconstructed flag: "
        f"{flag}"
    )

    submit_stage(
        "flag",
        {
            "flag": flag,
        },
    )

    print("\n" + "=" * 72)
    print("FINAL RESULT")
    print("=" * 72)

    print("[AGENT] All six milestones submitted.")
    print("[SERVER] Rewards were returned after each submission.")
    print("[AGENT] Final flag:", flag)

    print("\nExpected reward trajectory:")
    print("  recon      = +10")
    print("  rollback   = +20")
    print("  channel    = +20")
    print("  residue    = +20")
    print("  payload    = +20")
    print("  flag       = +10")
    print("  ----------------")
    print("  TOTAL      = 100")


if __name__ == "__main__":
    main()

