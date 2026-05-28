import jax
import jax.numpy as jnp
import optax  # JAX's standard optimizer library (pip install optax)
from model import PINN
from helper import pde_residual_batch

print("--- RUNNING TEST: JAX Inverse PINN ---")

# 1. Initialize the Master Parameters Dictionary
model = PINN()
key = jax.random.PRNGKey(42)
dummy_x, dummy_t = jnp.ones((1, 1)), jnp.ones((1, 1))

params = {
    'mlp': model.init(key, dummy_x, dummy_t),
    'alpha': jnp.array([0.5])  # Our initial wrong guess
}

# 2. Generate Fake "Sensor Data" (True alpha is 0.1)
x_data = jax.random.uniform(jax.random.PRNGKey(1), (100, 1), minval=-1.0, maxval=1.0)
t_data = jax.random.uniform(jax.random.PRNGKey(2), (100, 1), minval=0.0, maxval=1.0)
u_true = jnp.exp(-0.1 * (jnp.pi**2) * t_data) * jnp.sin(jnp.pi * x_data)

# Collocation points
x_col = jnp.linspace(-1, 1, 2000).reshape(-1, 1)
t_col = jnp.linspace(0, 1, 2000).reshape(-1, 1)

# 3. Define the Compound Loss Function
def loss_fn(p, x_d, t_d, u_d, x_c, t_c):
    # Data Loss
    u_pred = jax.vmap(lambda x, t: PINN().apply(p['mlp'], jnp.array([[x]]), jnp.array([[t]])).squeeze())(x_d.flatten(), t_d.flatten())
    loss_data = jnp.mean((u_pred - u_d.flatten()) ** 2)
    
    # Physics Loss
    residual = pde_residual_batch(p, x_c.flatten(), t_c.flatten())
    loss_pde = jnp.mean(residual ** 2)
    
    return loss_data + loss_pde

# 4. Setup Optax Optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

# Define a single compiled training step
@jax.jit
def step_fn(params, opt_state, x_d, t_d, u_d, x_c, t_c):
    loss, grads = jax.value_and_grad(loss_fn)(params, x_d, t_d, u_d, x_c, t_c)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# 5. Training Loop
for step in range(1001):
    params, opt_state, loss = step_fn(params, opt_state, x_data, t_data, u_true, x_col, t_col)
    if step % 200 == 0:
        print(f"Step {step} | Loss: {loss:.4f} | Discovered Alpha: {params['alpha'][0]:.4f}")

print(f"SUCCESS: Final Discovered Alpha: {params['alpha'][0]:.4f} (Target: 0.1000)")