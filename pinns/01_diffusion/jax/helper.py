import jax
import flax.linen as nn
import jax.numpy as jnp
from model import PINN

def u_fn(params, x, t):

    x = jnp.asarray(x).reshape(())
    t = jnp.asarray(t).reshape(())

    x_array = jnp.array([[x]])
    t_array = jnp.array([[t]])

    u_array  = PINN().apply(params, x_array, t_array)

    return u_array.squeeze()

u_t_fn = jax.grad(u_fn, argnums = 2)
u_x_fn = jax.grad(u_fn, argnums = 1)

u_xx_fn = jax.grad(u_x_fn, argnums = 1)

def pde_residual_single(params, x, t, alpha=0.1):

    u_t = u_t_fn(params, x, t)
    diffusion_term = alpha * u_xx_fn(params, x, t)

    return u_t - diffusion_term

pde_residual_batch = jax.vmap(pde_residual_single, in_axes=(None, 0, 0, None))