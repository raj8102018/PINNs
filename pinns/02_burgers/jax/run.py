import jax
import jax.numpy as jnp
from model import PINN
from helper import pde_residual_batch
import math

print("--- RUNNING TEST: JAX/Flax Burgers' Equation ---")

model = PINN()
key = jax.random.PRNGKey(42) 

dummy_x = jnp.ones((1, 1))
dummy_t = jnp.ones((1, 1))
params = model.init(key, dummy_x, dummy_t)

# Dummy Collocation points
x_col = jnp.linspace(-1, 1, 10)
t_col = jnp.linspace(0, 1, 10)
nu_val = 0.01 / math.pi

# Compute batch PDE residual
residual = pde_residual_batch(params, x_col, t_col, nu_val)
assert residual.shape == (10,), f"FATAL: Residual shape mismatch. Expected (10,), got {residual.shape}"

# Gradient flow check
def loss_fn(p, x, t, nu):
    res = pde_residual_batch(p, x, t, nu)
    return jnp.mean(res ** 2)

grad_loss_fn = jax.grad(loss_fn, argnums=0)
grads = grad_loss_fn(params, x_col, t_col, nu_val)

print("SUCCESS: JAX successfully compiled the non-linear Burgers' PDE with correct tensor shapes.")