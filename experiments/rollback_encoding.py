import copy, os, torch
import torch.nn as nn
from src.model import TinyMLP

SEED=42; INPUT_DIM=16; HIDDEN_DIM=64; OUTPUT_DIM=4; SAMPLES=256; BATCH=32; STEPS=100; EVENT_STEP=40; LR=1e-3; PULSE=0.05
OUT="artifacts/rollback"

def data():
    g=torch.Generator().manual_seed(SEED); x=torch.randn(SAMPLES,INPUT_DIM,generator=g); y=torch.randint(0,OUTPUT_DIM,(SAMPLES,),generator=g); return x,y

def model():
    torch.manual_seed(SEED); return TinyMLP(INPUT_DIM,HIDDEN_DIM,OUTPUT_DIM)

def pulse(shape,symbol):
    g=torch.Generator().manual_seed(777)
    v=torch.randn(shape,generator=g); v=v/v.norm()
    bits=[int(x) for x in f"{symbol:04b}"]
    flat=v.flatten()
    w=flat.numel()//4
    for i,bit in enumerate(bits):
        if bit: flat[i*w:(i+1)*w]*=PULSE
        else: flat[i*w:(i+1)*w]=0
    return v

def train(symbol=None):
    x,y=data(); m=model(); opt=torch.optim.Adam(m.parameters(),lr=LR); loss_fn=nn.CrossEntropyLoss(); target=dict(m.named_parameters())["network.0.weight"]
    for step in range(STEPS):
        xb=x[(step%(SAMPLES//BATCH))*BATCH:((step%(SAMPLES//BATCH))+1)*BATCH]; yb=y[(step%(SAMPLES//BATCH))*BATCH:((step%(SAMPLES//BATCH))+1)*BATCH]
        opt.zero_grad(); loss=loss_fn(m(xb),yb); loss.backward()
        if step==EVENT_STEP and symbol is not None:
            saved=copy.deepcopy(m.state_dict())
            target.grad.add_(pulse(target.grad.shape,symbol))
            opt.step()
            m.load_state_dict(saved)
        else: opt.step()
    return {"model":m.state_dict(),"optimizer":opt.state_dict(),"symbol":symbol}

def main():
    os.makedirs(OUT,exist_ok=True)
    torch.save(train(None),f"{OUT}/clean.pt")
    for symbol in range(16):
        torch.save(train(symbol),f"{OUT}/symbol_{symbol:02d}.pt")
        print(f"symbol={symbol:02d} bits={symbol:04b}")

if __name__=="__main__": main()