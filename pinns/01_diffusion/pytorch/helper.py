import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pde_residual(model: nn.Module, x: torch.Tensor, t: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    
    x.requires_grad=True
    t.requires_grad=True
    
    u = model(x,t)
    
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    residual = u_t - (alpha * u_xx)

    return residual

def pinn_loss(model: nn.Module, x_col: torch.Tensor, t_col: torch.Tensor, x_ic: torch.Tensor, t_ic: torch.Tensor, u_ic: torch.Tensor, x_bc: torch.Tensor, t_bc: torch.Tensor, u_bc: torch.Tensor) -> torch.Tensor:

    residual = compute_pde_residual(model, x_col, t_col)

    mse_1 = F.mse_loss(residual, torch.zeros_like(residual))
    
    mse_2 = F.mse_loss(model(x_ic, t_ic), u_ic)

    mse_3 = F.mse_loss(model(x_bc, t_bc), u_bc)

    return mse_1 + mse_2 + mse_3
