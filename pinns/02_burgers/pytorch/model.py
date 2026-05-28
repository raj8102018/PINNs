import torch
import torch.nn as nn


class PINN(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2, 50, bias=False),
            nn.Tanh(),
            nn.Linear(50, 50, bias=False),
            nn.Tanh(),
            nn.Linear(50, 50, bias=False),
            nn.Tanh(),
            nn.Linear(50, 50, bias=False),
            nn.Tanh(),
            nn.Linear(50, 50, bias=False),
            nn.Tanh(),
            nn.Linear(50,1, bias=False)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        value = torch.cat((x,t), dim=1)
        u = self.mlp(value)

        return u