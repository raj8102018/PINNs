import jax
import flax.linen as nn
import jax.numpy as jnp

class PINN(nn.Module):

    @nn.compact

    def __call__(self, x, t):

        inputs = jnp.concatenate([x,t], axis =-1)

        z = nn.Dense(32)(inputs)
        z = nn.tanh(z)
        z = nn.Dense(32)(z)
        z = nn.tanh(z)
        z = nn.Dense(32)(z)
        z = nn.tanh(z)

        return nn.Dense(1)(z)
