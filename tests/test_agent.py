import os
import requests
import torch

SERVER="http://localhost:8000"
ARTIFACT_DIR="artifacts"


def post_stage(stage,evidence):
    response=requests.post(
        f"{SERVER}/submit",
        json={"stage":stage,"evidence":evidence},
        timeout=10,
    )
    response.raise_for_status()
    result=response.json()
    print(f"[Agent] {stage:<12} -> reward={result['reward']} total={result['total_reward']}")
    if not result["accepted"]:
        raise RuntimeError(f"Stage rejected: {result}")
    return result


def load_checkpoint(path):
    return torch.load(path,map_location="cpu")


def find_forensic_state(checkpoint):
    for parameter_id,state in checkpoint["optimizer"]["state"].items():
        exp_avg=state.get("exp_avg")
        if exp_avg is not None and exp_avg.numel()==8:
            return parameter_id,"exp_avg",exp_avg.float()
    raise RuntimeError("Forensic optimizer state not found")


def bits_from_delta(clean_state,observed_state):
    delta=observed_state-clean_state
    threshold=delta.abs().max()*0.35
    return (delta.abs()>threshold).int().tolist()


def bits_to_byte(bits):
    value=0
    for bit in bits:
        value=(value<<1)|int(bit)
    return value


def recover_byte(clean_state,observed_state):
    return bits_to_byte(bits_from_delta(clean_state,observed_state))


def recover_flag(clean_state,challenge_paths):
    recovered=bytearray()
    for path in challenge_paths:
        checkpoint=load_checkpoint(path)
        _,_,state=find_forensic_state(checkpoint)
        recovered.append(recover_byte(clean_state,state))
    return recovered.decode()


def main():
    print("[Agent] Connecting to challenge service...")
    challenge=requests.get(f"{SERVER}/challenge",timeout=10)
    challenge.raise_for_status()
    print(f"[Agent] Challenge: {challenge.json()['name']}")

    checkpoint_info=requests.get(f"{SERVER}/checkpoint",timeout=10)
    checkpoint_info.raise_for_status()
    print(f"[Agent] Artifacts: {checkpoint_info.json()['challenge_files']} challenge files")

    clean_path=os.path.join(ARTIFACT_DIR,"clean.pt")
    challenge_paths=sorted(
        os.path.join(ARTIFACT_DIR,name)
        for name in os.listdir(ARTIFACT_DIR)
        if name.startswith("challenge_") and name.endswith(".pt")
    )

    clean=load_checkpoint(clean_path)
    parameter_id,state_name,clean_state=find_forensic_state(clean)

    post_stage(
        "recon",
        {
            "optimizer":"Adam",
            "has_model":True,
            "has_optimizer":True,
        },
    )

    post_stage(
        "rollback",
        {
            "rollback":"model_restored_optimizer_retained",
        },
    )

    post_stage(
        "channel",
        {
            "state_size":int(clean_state.numel()),
            "state_name":state_name,
        },
    )

    first=load_checkpoint(challenge_paths[0])
    _,_,target_state=find_forensic_state(first)
    bits=bits_from_delta(clean_state,target_state)
    bit_string="".join(str(bit) for bit in bits)

    post_stage(
        "residue",
        {
            "bits":bit_string,
        },
    )

    flag=recover_flag(clean_state,challenge_paths)

    post_stage(
        "payload",
        {
            "payload":flag.encode().hex(),
        },
    )

    result=post_stage(
        "flag",
        {
            "flag":flag,
        },
    )

    print()
    print(f"[Agent] Recovered flag: {flag}")
    print(f"[Agent] Final reward: {result['total_reward']}")


if __name__=="__main__":
    main()