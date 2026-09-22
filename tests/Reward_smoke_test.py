import json
from pathlib import Path

from challenge.evaluater import Evaluator
from challenge.reward_state import RewardState


ROOT=Path("artifacts")
FLAG="PROVUE{OPTIMIZER_AUTOPSY}"


class TestAgent:
    def __init__(self):
        self.clean=ROOT/"clean.pt"
        self.challenges=sorted(ROOT.glob("challenge_*.pt"))
        self.evaluator=Evaluator(str(self.clean),[str(p) for p in self.challenges])

    def run_episode(self,episode):
        rewards=RewardState()
        total=0

        print(f"\n=== EPISODE {episode} ===")

        stages=[
            ("recon",{"has_model":True,"has_optimizer":True,"optimizer":"Adam"}),
            ("rollback",{"rollback":"model_restored_optimizer_retained"}),
            ("channel",{"state_name":"exp_avg","state_size":8}),
        ]

        for stage,evidence in stages:
            reward=self._submit(stage,evidence,rewards)
            total+=reward
            print(f"{stage:10s} reward={reward:2d} total={total:3d}")

        bits=self._recover_bits()
        reward=self._submit("residue",{"bits":bits},rewards)
        total+=reward
        print(f"{'residue':10s} reward={reward:2d} total={total:3d}")

        payload=FLAG.encode().hex()
        reward=self._submit("payload",{"payload":payload},rewards)
        total+=reward
        print(f"{'payload':10s} reward={reward:2d} total={total:3d}")

        reward=self._submit("flag",{"flag":FLAG},rewards)
        total+=reward
        print(f"{'flag':10s} reward={reward:2d} total={total:3d}")

        print(f"EPISODE {episode} TOTAL = {total}")

        assert total==100

    def _submit(self,stage,evidence,rewards):
        checks={
            "recon":self.evaluator.check_recon,
            "rollback":self.evaluator.check_rollback,
            "channel":self.evaluator.check_channel,
            "residue":self.evaluator.check_residue,
            "payload":self.evaluator.check_payload,
            "flag":self.evaluator.check_flag,
        }

        if not checks[stage](evidence):
            return 0

        return rewards.award(stage)

    def _recover_bits(self):
        clean=self.evaluator.clean_state
        observed=self.evaluator.target_state
        delta=observed-clean
        threshold=delta.abs().max()*0.35
        bits=(delta.abs()>threshold).int().tolist()
        return "".join(str(bit) for bit in bits)


def main():
    for episode in range(1,3):
        TestAgent().run_episode(episode)

    print("\nTEST AGENT PASSED")


if __name__=="__main__":
    main()