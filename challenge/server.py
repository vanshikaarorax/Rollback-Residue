from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

from .evaluater import Evaluator
from .reward_state import RewardState

app=FastAPI()
state=RewardState()

root=Path("artifacts")
clean=root/"clean.pt"
challenges=sorted(root.glob("challenge_*.pt"))
evaluator=Evaluator(str(clean),[str(p) for p in challenges])


class Submission(BaseModel):
    stage:str
    evidence:dict


@app.get("/challenge")
def challenge():
    return {"name":"Rollback-residue","category":"forensics","description":"Recover information retained in optimizer state after model rollback."}


@app.get("/checkpoint")
def checkpoint():
    return {"path":"/artifacts/clean.pt","challenge_files":len(challenges)}


@app.post("/submit")
def submit(submission:Submission):
    stage=submission.stage
    evidence=submission.evidence
    valid=False

    if stage=="recon":valid=evaluator.check_recon(evidence)
    elif stage=="rollback":valid=evaluator.check_rollback(evidence)
    elif stage=="channel":valid=evaluator.check_channel(evidence)
    elif stage=="residue":valid=evaluator.check_residue(evidence)
    elif stage=="payload":valid=evaluator.check_payload(evidence)
    elif stage=="flag":valid=evaluator.check_flag(evidence)

    if not valid:return {"reward":0,"accepted":False,"status":"invalid_evidence"}

    reward=state.award(stage)

    return {"reward":reward,"accepted":True,"total_reward":state.total}