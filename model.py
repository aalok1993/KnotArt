import torch
from torch import nn
import math

def quaternions_to_rotation_matrices(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


class CouplingLayer(nn.Module):
    def __init__(self, map_s, map_t, mask):
        super().__init__()
        self.map_s = map_s
        self.map_t = map_t
        self.register_buffer("mask", mask)

    def forward(self, y):
        y1 = y * self.mask
        s = self.map_s(y1)
        t = self.map_t(y1)
        x = y1 + (1-self.mask) * ((y - t) * torch.exp(-s))
        return x

    def inverse(self, x):
        x1 = x * self.mask
        s = self.map_s(x1)
        t = self.map_t(x1)
        y = x1 + (1-self.mask) * (x * torch.exp(s) + t)
        return y


class SimpleNVP(nn.Module):
    def __init__(self, n_layers, hidden_size, device):
        super().__init__()

        input_dims = 3
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            mask = torch.zeros(input_dims)
            mask[torch.randperm(input_dims)[:2]] = 1

            map_s = nn.Sequential(
                nn.Linear(input_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dims),
                nn.Hardtanh(min_val=-10, max_val=10)
            )
            map_t = nn.Sequential(
                nn.Linear(input_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dims)
            )
            self.layers.append(CouplingLayer(
                map_s,
                map_t,
                mask[None]
            ))

    def forward(self, x):
        y = x
        for i,l in enumerate(self.layers):
            y = l(y)
        return y

    def inverse(self, y):
        x = y
        for i,l in enumerate(reversed(self.layers)):
            x = l.inverse(x)
        return x


class RST_NVP(nn.Module):
    def __init__(self, n_layers, hidden_size, device, w_init = 1e-8):
        super().__init__()
        
        input_dims = 3
        self.quaternions = torch.nn.Parameter(torch.cat([torch.ones((n_layers+1,1)), w_init*(-1+2*torch.randn((n_layers+1,3)))],-1).to(device), requires_grad=True)
        self.translations = torch.nn.Parameter(w_init*(-1+2*torch.randn((n_layers+1,3)).to(device)), requires_grad=True)
        self.scalings = torch.nn.Parameter(w_init*(-1+2*torch.randn((n_layers+1,3)).to(device)), requires_grad=True)
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            mask = torch.zeros(input_dims)
            mask[torch.randperm(input_dims)[:2]] = 1

            map_s = nn.Sequential(
                nn.Linear(input_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dims),
                nn.Hardtanh(min_val=-10, max_val=10)
            )
            map_t = nn.Sequential(
                nn.Linear(input_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dims)
            )
            self.layers.append(CouplingLayer(
                map_s,
                map_t,
                mask[None]
            ))

    def forward(self, x):
        
        S = (torch.nn.functional.elu(self.scalings) + 1).unsqueeze(1)
        R = quaternions_to_rotation_matrices(torch.nn.functional.normalize(self.quaternions,dim=-1)).unsqueeze(1)
        T = self.translations.unsqueeze(1)
        y = x
        y = torch.matmul((y * S[0]).unsqueeze(-2), R[0]).squeeze(-2) + T[0]
        for i,l in enumerate(self.layers):
            y = l(y)
            y = torch.matmul((y * S[i+1]).unsqueeze(-2), R[i+1]).squeeze(-2) + T[i+1]
        return y

    def inverse(self, y):

        S = (torch.nn.functional.elu(self.scalings) + 1).unsqueeze(1)
        R = quaternions_to_rotation_matrices(torch.nn.functional.normalize(self.quaternions,dim=-1)).unsqueeze(1)
        T = self.translations.unsqueeze(1)
        x = y
        for i,l in enumerate(reversed(self.layers)):
            x = torch.matmul((x - T[n_layers-i]).unsqueeze(-2), R[n_layers-i].transpose(-1,-2)).squeeze(-2) / S[n_layers-i]
            x = l.inverse(x)
        x = torch.matmul((x - T[0]).unsqueeze(-2), R[0].transpose(-1,-2)).squeeze(-2) / S[0]
        return x


class T_NVP(nn.Module):
    def __init__(self, n_layers, hidden_size, device, w_init=1e-8):
        super().__init__()
        
        input_dims = 3
        self.translations = torch.nn.Parameter(w_init*(-1+2*torch.randn((n_layers,3)).to(device)), requires_grad=True)
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            mask = torch.zeros(input_dims)
            mask[torch.randperm(input_dims)[:2]] = 1

            map_s = nn.Sequential(
                nn.Linear(input_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dims),
                nn.Hardtanh(min_val=-10, max_val=10)
            )
            map_t = nn.Sequential(
                nn.Linear(input_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dims)
            )
            self.layers.append(CouplingLayer(
                map_s,
                map_t,
                mask[None]
            ))


    def forward(self, x):
        T = self.translations.unsqueeze(1)
        y = x
        y = y  + T[0]
        for i,l in enumerate(self.layers):
            y = l(y)
            y = y  + T[i]
        return y

    def inverse(self, y):
        T = self.translations.unsqueeze(1)
        x = y
        for i,l in enumerate(reversed(self.layers)):
            x = x - T[n_layers-i-1]
            x = l.inverse(x)
        x = x - T[0]
        return x

