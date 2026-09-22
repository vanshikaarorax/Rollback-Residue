import torch
import torch.nn as nn

BUFFER_SIZE=8

class ForensicBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.residue=nn.Parameter(torch.zeros(BUFFER_SIZE))

    def forward(self,bits):
        return torch.dot(self.residue,bits.float())

def bits_from_byte(value):
    return torch.tensor([(value >> (7-i)) & 1 for i in range(8)],dtype=torch.float32)