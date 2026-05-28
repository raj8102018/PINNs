import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pde_residual(model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    
    x.requires_grad=True
    t.requires_grad=True
    
    u = model(x,t)
    
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]

    return u_tt - u_xx

def pinn_loss(model, x_col, t_col, x_bc, t_bc, u_bc_target, x_ic, t_ic, u_ic_target):

    residual = compute_pde_residual(model, x_col, t_col)

    pde_loss = F.mse_loss(residual, torch.zeros_like(residual))

    bc_loss = F.mse_loss(model(x_bc, t_bc), u_bc_target)

    u_ic = model(x_ic, t_ic)

    ic1_loss = F.mse_loss(u_ic, u_ic_target)

    velocity = torch.autograd.grad(u_ic, t_ic, grad_outputs=torch.ones_like(u_ic), create_graph=True)[0]

    ic2_loss = F.mse_loss(velocity, torch.zeros_like(velocity))

    return pde_loss + bc_loss + ic1_loss + ic2_loss