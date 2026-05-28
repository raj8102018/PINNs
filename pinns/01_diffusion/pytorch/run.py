import torch
from model import PINN
from helper import compute_pde_residual

print("--- RUNNING TEST: PyTorch PINN Autograd Engine ---")

# 1. Initialize the model
model = PINN()

# 2. Create dummy collocation points (x, t)
# We must set requires_grad=True so autograd can track the input variables
x_col = torch.linspace(-1, 1, 10).view(-1, 1).requires_grad_(True)
t_col = torch.linspace(0, 1, 10).view(-1, 1).requires_grad_(True)

# 3. Compute the PDE residual
alpha_val = 0.1
residual = compute_pde_residual(model, x_col, t_col, alpha=alpha_val)

# Assertions
assert residual.shape == (10, 1), f"FATAL: Residual shape mismatch. Expected (10, 1), got {residual.shape}"
assert residual.requires_grad, "FATAL: The residual lost its gradient graph! You forgot create_graph=True."

# Ensure the network weights receive gradients when we backprop the physical loss
loss = torch.mean(residual ** 2)
loss.backward()

# Check the first layer's weights to ensure gradients flowed from the PDE math back to the params
has_grads = any(p.grad is not None for p in model.parameters())
assert has_grads, "FATAL: Gradients did not flow from the PDE residual back to the neural network weights!"

print("SUCCESS: The physical constraints are fully differentiable. The PINN is operational.")