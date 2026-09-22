REWARDS={"recon":10,"rollback":20,"channel":20,"residue":20,"payload":20,"flag":10}

class RewardState:
    def __init__(self):
        self.reached=set()
        self.total=0

    def award(self,stage):
        if stage in self.reached:return 0
        self.reached.add(stage)
        reward=REWARDS[stage]
        self.total+=reward
        return reward

    def result(self):
        return {"reward":self.total,"milestones":sorted(self.reached)}