import os,torch

BASE="artifacts/forensic_buffer"

def load(path):
    c=torch.load(path,map_location="cpu")
    for state in c["optimizer"]["state"].values():
        if "exp_avg" in state and state["exp_avg"].numel()==8:
            return state["exp_avg"].float()
    raise RuntimeError("8-element forensic optimizer state not found")

def main():
    clean=load(f"{BASE}/clean.pt")
    print("\nFORENSIC BUFFER")
    print("="*100)

    correct=0

    for value in range(256):
        state=load(f"{BASE}/byte_{value:03d}.pt")
        delta=state-clean
        scores=delta.abs()
        predicted=(scores>scores.max()*0.35).int()
        decoded=sum(int(predicted[i])<<(7-i) for i in range(8))

        if decoded==value: correct+=1

        if value<32 or value%16==0:
            print(f"{value:03d} {value:08b} delta={[round(x,5) for x in delta.tolist()]} decoded={decoded:03d} bits={''.join(map(str,predicted.tolist()))}")

    print(f"\nBYTE DECODING: {correct}/256 = {correct/256:.1%}")

if __name__=="__main__": main()