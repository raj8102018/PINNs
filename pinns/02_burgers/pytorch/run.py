import torch
import math
from model import PINN
from helper import compute_pde_residual

print("--- RUNNING TEST: PyTorch Burgers' Equation Autograd ---")

# 1. Initialize the model
model = PINN()

# 2. Create dummy collocation points (x, t)
x_col = torch.linspace(-1, 1, 10).view(-1, 1).requires_grad_(True)
t_col = torch.linspace(0, 1, 10).view(-1, 1).requires_grad_(True)

# 3. Compute the Burgers' PDE residual
nu_val = 0.01 / math.pi
residual = compute_pde_residual(model, x_col, t_col, nu=nu_val)

# Assertions
assert residual.shape == (10, 1), f"FATAL: Residual shape mismatch. Expected (10, 1), got {residual.shape}"
assert residual.requires_grad, "FATAL: The residual lost its gradient graph!"

# Backpropagate to ensure the computational graph handles the u * u_x term correctly
loss = torch.mean(residual ** 2)
loss.backward()

has_grads = any(p.grad is not None for p in model.parameters())
assert has_grads, "FATAL: Gradients failed to flow through the non-linear PDE terms!"

print("SUCCESS: The non-linear physics constraints are differentiable. Burgers' PINN is operational.")