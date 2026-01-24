import torch
import math 
from einops import einsum, rearrange, reduce


class Linear(torch.nn.Module):

    def __init__(self, in_feature: int, out_feature: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        mean = 0
        std = math.sqrt(2 / (out_feature + in_feature))
        lower = -3 * std
        upper = 3 * std

        w = torch.empty((out_feature, in_feature), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, mean, std, lower, upper)

        self.weight = torch.nn.Parameter(w)

    def forward(self, x: torch.Tensor):
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
    
    