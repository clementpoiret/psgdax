# psgdax: Preconditioned Stochastic Gradient Descent in JAX

`psgdax` provides a JAX implementation of the Preconditioned Stochastic Gradient Descent (PSGD) optimizer, compatible
with the [Optax](https://github.com/google-deepmind/optax) ecosystem. This library implements the Kronecker-product
preconditioner (Kron) based on the theory of Hessian fitting in Lie groups.

This implementation translates the [PyTorch reference implementation](https://github.com/lixilinx/psgd_torch) by
Xi-Lin Li.

## Mathematical Background

PSGD reformulates preconditioner estimation as a strongly convex optimization problem on Lie groups. Unlike standard
quasi-Newton methods (e.g., BFGS, KFAC) that operate in Euclidean space or the manifold of SPD matrices, PSGD updates
the preconditioner $Q$ (where $P = Q^T Q$) using multiplicative updates that avoid explicit matrix inversion.

The update rule minimizes the criterion $E[\\delta g^T P \\delta g + \\delta \\theta^T P^{-1} \\delta \\theta]$,
ensuring the preconditioner approximates the Hessian or the inverse covariance of gradients.

## Installation

```bash
pip install psgdax
```

## Usage

### Basic Usage with Optax

`psgdax` follows the `optax.GradientTransformation` interface. It can be chained with other transformations, though the
provided `kron` alias handles standard scheduling, weight decay, and scale-by-learning-rate chains automatically.

```python
import jax
import jax.numpy as jnp
from psgdax import kron

# Define parameters
params = {
    'w': jnp.zeros((128, 128)),
    'b': jnp.zeros((128,))
}

# Initialize optimizer
# The default mode is Q0.5EQ1.5 (Procrustes-regularized update)
optimizer = kron(
    learning_rate=1e-3,
    b1=0.9,                 # Momentum
    preconditioner_lr=0.1,  # Learning rate for the preconditioner Q
    whiten_grad=True        # Whiten gradients (True) or Momentum (False)
)

opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, grads):
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = jax.tree.map(lambda p, u: p + u, params, updates)
    return new_params, new_opt_state
```

### Advanced Usage: Scanned Layers

For deep architectures (e.g., Transformers) implemented via `jax.lax.scan`, `psgdax` supports explicit handling of
scanned layers to prevent unrolling computation graphs. This significantly improves compilation time and memory
efficiency.

```python
import jax
from psgdax import kron

# Assume a boolean pytree mask where True indicates a scanned parameter
# matching the structure of 'params'
scanned_layers_mask = ... 

optimizer = kron(
    learning_rate=3e-4,
    scanned_layers=scanned_layers_mask,
    lax_map_scanned_layers=True, # Use lax.map for preconditioner updates
    lax_map_batch_size=8
)
```

## Configuration

### Preconditioner Modes

The geometry of the preconditioner update $dQ$ is controlled via `preconditioner_mode`.

| Mode        | Formula                             | Description                                                                                                                       |
| :---------- | :---------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| `Q0.5EQ1.5` | $dQ = Q^{0.5} \\mathcal{E} Q^{1.5}$ | **Recommended**. Uses an online orthogonal Procrustes solver to keep $Q$ approximately SPD. Numerically stable for low precision. |
| `EQ`        | $dQ = \\mathcal{E} Q$               | The original PSGD update (triangular group). Requires triangular solves.                                                          |
| `QUAD`      | Quadratic Form                      | Ensures $Q$ remains symmetric positive definite via quadratic form updates.                                                       |

### Hyperparameters

- **`preconditioner_lr`**: The learning rate for $Q$. Recommended range $[0.01, 0.1]$.
- **`preconditioner_update_probability`**: Probability of updating $Q$ at each step. Can be a float or a schedule
  callable. Annealing this probability can reduce overhead.
- **`max_size_triangular`**: Dimensions larger than this will default to diagonal preconditioners to save memory.
- **`memory_save_mode`**:
  - `None`: Standard behavior.
  - `'one_diag'`: Forces the largest dimension of a tensor to be diagonal.
  - `'all_diag'`: Forces all dimensions to be diagonal (similar to Shampoo without blocks).
- **`whiten_grad`**:
  - `True`: The preconditioner whitens the raw gradient.
  - `False`: The preconditioner whitens the momentum vector. Requires `b1 > 0`. *Note: If `False`, the learning rate
    typically needs to be reduced by a factor of $\\sqrt{\\frac{1+\\beta}{1-\\beta}}$.*

### Precision

JAX defaults to `bfloat16` on TPUs or `float32` depending on configuration. PSGD is sensitive to precision during the
preconditioner update.

- `precond_update_precision`: Defaults to `"tensorfloat32"`.
- `precond_grads_precision`: Precision for the application of the preconditioner to the gradient.

## Implementation Details

### Kronecker Decomposition

For a tensor parameter of shape $(n_1, n_2, \\dots)$, PSGD approximates the Hessian inverse as a Kronecker product of
smaller matrices $Q = Q_1 \\otimes Q_2 \\dots$.

- Dimensions where $n_i > \\text{max_size}$ or $n_i^2 > \\text{max_skew} \\cdot \\text{numel}$ are approximated via
  diagonal matrices.
- Dimensions fitting the criteria utilize full dense matrices.

### Eigenvalue Bounds

To ensure numerical stability without expensive eigenvalue decompositions, this implementation utilizes randomized
lower-bound estimators for spectral norms (`_norm_lower_bound_spd` and `_norm_lower_bound_skh`) during the update of
Lipschitz constants.

## Citations

This library is a translation of the work by Xi-Lin Li. If you use this optimizer, please cite the original papers:

```bibtex
@article{li2015preconditioned,
  title={Preconditioned Stochastic Gradient Descent},
  author={Li, Xi-Lin},
  journal={arXiv preprint arXiv:1512.04202},
  year={2015}
}

@article{li2018preconditioner,
  title={Preconditioner on Matrix Lie Group for SGD},
  author={Li, Xi-Lin},
  journal={arXiv preprint arXiv:1809.10232},
  year={2018}
}

@article{li2024stochastic,
  title={Stochastic Hessian Fittings with Lie Groups},
  author={Li, Xi-Lin},
  journal={arXiv preprint arXiv:2402.11858},
  year={2024}
}
```

## Acknowledgments

- **Xi-Lin Li**: Author of the original [PSGD algorithm and PyTorch implementation](https://github.com/lixilinx/psgd_torch).
- **Evanatyourservice**: Author of the [preliminary JAX port](https://github.com/evanatyourservice/psgd_jax) upon which this library improves.
