import torch
import torch.nn as nn


class PINN(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32,1, bias=False)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        value = torch.cat((x,t), dim=1)
        u = self.mlp(value)

        return u