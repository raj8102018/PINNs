import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from model import PINN
from helper import pde_residual_batch

import matplotlib.pyplot as plt

model = PINN(rngs=nnx.Rngs(0))

optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

key = jax.random.PRNGKey(42)

key, *subkeys = jax.random.split(key, 5)
num_col = 2000
z_col = jax.random.uniform(subkeys[0], (num_col, 1), minval=0.0, maxval=1.0)
t_col = jax.random.uniform(subkeys[1], (num_col, 1), minval=0.0, maxval=1.0)
  
num_bc = 100
z_bc = jnp.concatenate([jnp.zeros((num_bc, 1)), jnp.ones((num_bc, 1))], axis=0)
t_bc = jax.random.uniform(subkeys[2], (num_bc * 2, 1), minval=0.0, maxval=1.0)
u_bc_target = jnp.zeros_like(z_bc)

num_ic = 100
z_ic = jax.random.uniform(subkeys[3], (num_ic, 1), minval=0.0, maxval=1.0)
t_ic = jnp.zeros((num_ic, 1))
u_ic_target = jnp.ones_like(z_ic)

z_test = jnp.linspace(0, 1, 100).reshape(-1, 1)
time_steps = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]


def loss_fn(model, z_col, t_col, z_bc, t_bc, z_ic, t_ic, u_ic_target, u_bc_target):
    
    residual = pde_residual_batch(model, z_col.flatten(), t_col.flatten())

    pde_loss = jnp.mean(residual**2)

    u_bc_pred = model(z_bc, t_bc)

    u_ic_pred = model(z_ic, t_ic)

    bc_loss = jnp.mean((u_bc_pred - u_bc_target)**2)

    ic_loss = jnp.mean((u_ic_pred - u_ic_target)**2)

    return pde_loss + (1 * ic_loss) + (1 * bc_loss)


@nnx.jit
def step_fn(model, optimizer, z_col, t_col, z_bc, t_bc, z_ic, t_ic, u_ic_target, u_bc_target):
    
    loss, grads = nnx.value_and_grad(loss_fn)(model, z_col, t_col, z_bc, t_bc, z_ic, t_ic, u_ic_target, u_bc_target)
    optimizer.update(model, grads)

    return loss

for step in range(2001):
    loss = step_fn(model, optimizer, z_col, t_col, z_bc, t_bc, z_ic, t_ic, u_ic_target, u_bc_target)
    if step % 200 == 0:
        print(f"Step {step:4d} | Total Loss: {loss:.5f}")

print("SUCCESS: NNX 1D Terzaghi Equation trained successfully.")

plt.figure(figsize=(8, 6))

for t_val in time_steps:

    t_test = jnp.full_like(z_test, t_val)
    
    u_pred = model(z_test, t_test)
    
    plt.plot(z_test, u_pred, label=f"$T_v$ = {t_val}")

plt.ylim(-0.1, 1.1)
plt.xlabel("Normalized Depth (z/H)")
plt.ylabel("Excess Pore Pressure (u)")
plt.title("Terzaghi Consolidation Isochrones")
plt.legend()
plt.grid(True)

plt.savefig("terzaghi_isochrones.png")
plt.show()