import jax
import jax.numpy as jnp
from model import PINN
from helper import pde_residual_batch

print("--- RUNNING TEST: JAX/Flax Functional PINN Engine ---")

# 1. Initialize the model and get the parameters
model = PINN()
key = jax.random.PRNGKey(0) # JAX requires explicit random states

# Dummy inputs to tell Flax what the shapes look like
dummy_x = jnp.ones((1, 1))
dummy_t = jnp.ones((1, 1))
params = model.init(key, dummy_x, dummy_t)

# 2. Create batch collocation points (Notice: No requires_grad!)
# Shapes: [10, 1]
x_col = jnp.linspace(-1, 1, 10)
t_col = jnp.linspace(0, 1, 10)

# 3. Compute the batch PDE residual
alpha_val = 0.1
residual = pde_residual_batch(params, x_col, t_col, alpha_val)

# Assertions
assert residual.shape == (10,), f"FATAL: Residual shape mismatch. Expected (10, 1), got {residual.shape}"

# 4. Prove we can get the gradient of the loss with respect to the parameters
def loss_fn(p, x, t, alpha):
    res = pde_residual_batch(p, x, t, alpha)
    return jnp.mean(res ** 2)

# JAX magic: get a function that computes the gradient of the loss w.r.t the params (argnums=0)
grad_loss_fn = jax.grad(loss_fn, argnums=0)
grads = grad_loss_fn(params, x_col, t_col, alpha_val)

# If it didn't crash, the gradients successfully flowed backward through the functional physics engine
print("SUCCESS: The physical constraints are fully differentiable in JAX. The functional PINN is operational.")