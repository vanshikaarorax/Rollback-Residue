import copy,os,torch
import torch.nn as nn
from src.model import TinyMLP
from src.challenge.forensic_buffer import ForensicBuffer,bits_from_byte

SEED=42; INPUT_DIM=16; HIDDEN_DIM=64; OUTPUT_DIM=4; SAMPLES=256; BATCH=32; STEPS=100; EVENT_STEP=40; LR=1e-3
OUT="artifacts/forensic_buffer"

def data():
    g=torch.Generator().manual_seed(SEED)
    x=torch.randn(SAMPLES,INPUT_DIM,generator=g)
    y=torch.randint(0,OUTPUT_DIM,(SAMPLES,),generator=g)
    return x,y

def make_models():
    torch.manual_seed(SEED)
    return TinyMLP(INPUT_DIM,HIDDEN_DIM,OUTPUT_DIM),ForensicBuffer()

def train(value=None):
    x,y=data()
    model,buffer=make_models()
    opt=torch.optim.Adam(list(model.parameters())+list(buffer.parameters()),lr=LR)
    loss_fn=nn.CrossEntropyLoss()

    for step in range(STEPS):
        idx=(step%(SAMPLES//BATCH))*BATCH
        xb=x[idx:idx+BATCH]
        yb=y[idx:idx+BATCH]

        opt.zero_grad()
        loss=loss_fn(model(xb),yb)
        loss=loss+0.0*buffer.residue.sum()

        if step==EVENT_STEP and value is not None:
            bits=bits_from_byte(value)
            loss=loss+buffer(bits)

        loss.backward()

        if step==EVENT_STEP and value is not None:
            saved_model=copy.deepcopy(model.state_dict())
            saved_buffer=copy.deepcopy(buffer.state_dict())
            opt.step()
            model.load_state_dict(saved_model)
            buffer.load_state_dict(saved_buffer)
        else:
            opt.step()

    return {"model":model.state_dict(),"buffer":buffer.state_dict(),"optimizer":opt.state_dict(),"config":{"event_step":EVENT_STEP,"value":value}}

def main():
    os.makedirs(OUT,exist_ok=True)
    torch.save(train(),f"{OUT}/clean.pt")
    print("clean ->",f"{OUT}/clean.pt")

    for value in range(256):
        torch.save(train(value),f"{OUT}/byte_{value:03d}.pt")
        if value%16==0: print(f"generated through {value}")

if __name__=="__main__": main()