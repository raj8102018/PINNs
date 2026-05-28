import torch
import torch.optim as optim
import torch.nn.functional as F
import math
from model import PINN
from helper import pinn_loss

print("--- RUNNING TEST: PyTorch 1D Wave Equation ---")

model = PINN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 1. Collocation Points
num_col = 2000
x_col = torch.rand(num_col, 1).requires_grad_(True)
t_col = torch.rand(num_col, 1).requires_grad_(True)

# 2. Boundary Points (x=0 and x=1)
num_bc = 100
x_bc = torch.cat([torch.zeros(num_bc, 1), torch.ones(num_bc, 1)], dim=0)
t_bc = torch.rand(num_bc * 2, 1)
u_bc_target = torch.zeros_like(x_bc)

# 3. Initial Condition Points (t=0)
num_ic = 100
x_ic = torch.rand(num_ic, 1).requires_grad_(True) # Needed for sin(pi*x)
# CRITICAL: t_ic needs requires_grad to compute initial velocity u_t
t_ic = torch.zeros(num_ic, 1).requires_grad_(True) 
u_ic_target = torch.sin(math.pi * x_ic).detach()

# 4. Training Loop
for step in range(2001):
    optimizer.zero_grad()

    loss = pinn_loss(model, x_col, t_col, x_bc, t_bc, u_bc_target, x_ic, t_ic, u_ic_target)
    
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step:4d} | Total Loss: {loss.item():.5f}")

print("SUCCESS: 1D Wave Equation trained successfully.")