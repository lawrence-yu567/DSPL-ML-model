import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import corner
import matplotlib.pyplot as plt

def read_txt(filename, column):
    data = []
    with open(filename, "r") as f:
        for line in f:
            values = line.strip().split()
            data.append(values[column-1])
    return np.array(data[1:], dtype=float)

w_desi = read_txt("w.chain.2.txt", 3)
OmM_desi = read_txt("w.chain.2.txt", 5)


print(w_desi, OmM_desi)


desi_dist = dist.MultivariateNormal(
    jnp.array([jnp.mean(OmM_desi), jnp.mean(w_desi)]),
    jnp.array(jnp.cov(jnp.vstack([OmM_desi, w_desi])))
)



test_samples = desi_dist.sample(jax.random.PRNGKey(0), (10000,))
corner.hist2d(np.array(test_samples[:,0]), np.array(test_samples[:,1]), color='red')
plt.show()