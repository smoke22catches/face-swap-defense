import torch

class DifferentiableQuantize(torch.nn.Module):
    def __init__(self, levels=256):
        super().__init__()
        self.levels = levels - 1  # 255 for uint8

    def forward(self, x):
        # Scale up, round, scale back
        quantized = torch.round(x * self.levels) / self.levels
        # STE: forward uses quantized, backward sees x
        return x + (quantized - x).detach()