import torch

FLAG="PROVUE{OPTIMIZER_AUTOPSY}"

class Evaluator:
    def __init__(self,clean_path,challenge_paths):
        self.clean=torch.load(clean_path,map_location="cpu")
        self.challenges=[torch.load(p,map_location="cpu") for p in challenge_paths]
        self.clean_state=self._find_state(self.clean)
        self.target_state=self._find_state(self.challenges[0])

    def _find_state(self,checkpoint):
        for state in checkpoint["optimizer"]["state"].values():
            exp_avg=state.get("exp_avg")
            if exp_avg is not None and exp_avg.numel()==8:return exp_avg.float()
        raise RuntimeError("forensic state not found")

    def check_recon(self,data):
        return data.get("optimizer")=="Adam" and data.get("has_model") is True and data.get("has_optimizer") is True

    def check_rollback(self,data):
        return data.get("rollback")=="model_restored_optimizer_retained"

    def check_channel(self,data):
        return data.get("state_size")==8 and data.get("state_name")=="exp_avg"

    def check_residue(self,data):
        if not isinstance(data.get("bits"),str) or len(data["bits"])!=8:return False
        delta=self.target_state-self.clean_state
        threshold=delta.abs().max()*0.35
        expected="".join(str(int(x)) for x in (delta.abs()>threshold).int().tolist())
        return data["bits"]==expected

    def check_payload(self,data):
        return data.get("payload")==FLAG.encode().hex()

    def check_flag(self,data):
        return data.get("flag")==FLAG