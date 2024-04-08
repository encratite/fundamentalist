import torch

class TransformOutput(torch.nn.Module):
    def forward(self, x):
        return x[0]
