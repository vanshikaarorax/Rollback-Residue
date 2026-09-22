import os, torch

BASE="artifacts/rollback"
TARGET="network.0.weight"

def state(path):
    c=torch.load(path,map_location="cpu")
    for s in c["optimizer"]["state"].values():
        if "exp_avg" in s:return s["exp_avg"].float()
    raise RuntimeError("exp_avg not found")

def model(path):
    c=torch.load(path,map_location="cpu")
    return c["model"][TARGET].float()

def scores(delta):
    f=delta.flatten(); w=f.numel()//4
    return [f[i*w:(i+1)*w].norm().item() for i in range(4)]

def main():
    clean_avg=state(f"{BASE}/clean.pt"); clean_model=model(f"{BASE}/clean.pt")
    print("\nROLLBACK OPTIMIZER FORENSICS")
    print("="*100)
    for symbol in range(16):
        avg=state(f"{BASE}/symbol_{symbol:02d}.pt"); mdl=model(f"{BASE}/symbol_{symbol:02d}.pt")
        ds=avg-clean_avg; dm=mdl-clean_model; sc=scores(ds)
        bits=[int(x) for x in f"{symbol:04b}"]
        print(f"{symbol:02d} expected={symbol:04b} model_delta={dm.norm().item():.8f} scores={[round(x,6) for x in sc]} bits={bits}")

if __name__=="__main__": main()