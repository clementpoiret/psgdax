import string
from enum import Enum
from functools import partial
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from optax import tree_utils as otu
from optax._src import base, numerics, transform
from optax._src.combine import chain
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype


class PreconditionerMode(str, Enum):
    """Supported preconditioner update geometries."""

    EQ = "EQ"  # dQ = E * Q (triangular, requires triangular solves)
    Q0P5EQ1P5 = "Q0.5EQ1.5"  # dQ = Q^0.5 * E * Q^1.5 (recommended, uses Procrustes)
    QUAD = "QUAD"  # Quadratic form update (ensures Q stays SPD)


class KronState(NamedTuple):
    """State for Kronecker preconditioner."""

    count: jax.Array
    mu: Optional[Any]
    Qs_preconditioners: Any
    Ls_lipschitz: Optional[Any]  # Lipschitz smoothness constants per Q factor
    update_counter: jax.Array
    key: jax.Array


def init_scale_from_grads(
    grads,
    *,
    noise_scale: float = 1e-9,
    per_leaf: bool = True,
    dtype=jnp.float32,
) -> Union[Any, jnp.ndarray]:
    """
    Compute whitening init scale(s) from the first batch via:
        scale = (mean(|g|^2) + noise_scale^2)^(-1/4)
    """

    def to_array_or_none(g):
        if g is None:
            return None
        return jnp.asarray(g, dtype=dtype)

    grads = jax.tree_util.tree_map(to_array_or_none, grads)

    def leaf_scale(g):
        if g is None:
            return jnp.asarray(1.0, dtype=dtype)
        mean_sq = jnp.mean(jnp.square(g))
        mean_sq = jnp.nan_to_num(mean_sq, nan=0.0, posinf=0.0, neginf=0.0)
        return jnp.power(mean_sq + (noise_scale**2), -0.25)

    if per_leaf:
        return jax.tree_util.tree_map(leaf_scale, grads)

    leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
    if not leaves:
        return jnp.asarray(1.0, dtype=dtype)

    sum_sqs = jnp.sum(jnp.stack([jnp.sum(jnp.square(g)) for g in leaves], axis=0))
    counts = jnp.sum(
        jnp.stack([jnp.asarray(g.size, dtype=jnp.int32) for g in leaves], axis=0)
    )

    mean_sq_global = jnp.where(
        counts > 0,
        sum_sqs / counts.astype(sum_sqs.dtype),
        jnp.asarray(0.0, dtype=dtype),
    )
    mean_sq_global = jnp.nan_to_num(mean_sq_global, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.power(mean_sq_global + (noise_scale**2), -0.25).astype(dtype)


def precond_update_prob_schedule(
    max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500
):
    """Anneal preconditioner update probability during beginning of training."""

    def _schedule(n):
        return jnp.minimum(
            jnp.maximum(max_prob * jnp.exp(-decay * (n - flat_start)), min_prob),
            max_prob,
        )

    return _schedule


def _add_tiny(x):
    """Add smallest normal number to avoid division by zero."""
    return x + jnp.finfo(x.dtype).tiny


def _norm_lower_bound(A: jax.Array) -> jax.Array:
    """
    Returns a cheap lower bound for the spectral norm of A (general matrix).
    Used in EQ mode for normalizing updates.
    """
    max_abs = jnp.max(jnp.abs(A))

    def calc(A):
        A = A / max_abs
        A_conj = A.conj()
        aa = jnp.real(A * A_conj)
        aa_sum0 = jnp.sum(aa, axis=0)
        aa_sum1 = jnp.sum(aa, axis=1)
        i = jnp.argmax(aa_sum0, 0)
        j = jnp.argmax(aa_sum1, 0)
        value0 = jax.lax.dynamic_index_in_dim(aa_sum0, i, 0, keepdims=False)
        value1 = jax.lax.dynamic_index_in_dim(aa_sum1, j, 0, keepdims=False)

        def gt_branch():
            x = jax.lax.dynamic_index_in_dim(A, i, 1, keepdims=False)
            x = x.conj() @ A
            return max_abs * jnp.linalg.norm((x / jnp.linalg.norm(x)) @ A_conj.T)

        def le_branch():
            x = jax.lax.dynamic_index_in_dim(A, j, 0, keepdims=False)
            x = A @ x.conj()
            return max_abs * jnp.linalg.norm(A_conj.T @ (x / jnp.linalg.norm(x)))

        return jax.lax.cond(value0 > value1, gt_branch, le_branch)

    return jax.lax.cond(max_abs > 0, calc, lambda _: max_abs, A)


def _norm_lower_bound_spd(
    A: jax.Array, key: jax.Array, k: int = 32, half_iters: int = 2
) -> jax.Array:
    """
    Returns a cheap lower bound for the spectral norm of a symmetric positive
    definite matrix A using subspace iteration.

    Args:
        A: SPD matrix
        k: dimension of subspace (32 for float32, 128 for bfloat16)
        half_iters: half of the number of subspace iterations
    """
    smallest_normal = jnp.finfo(A.dtype).smallest_normal
    normalizing_factor = jnp.max(jnp.real(jnp.diag(A))) + smallest_normal
    A = A / normalizing_factor

    # Find row with largest norm for initial alignment
    row_norms = jnp.linalg.norm(A, axis=1)
    j = jnp.argmax(row_norms)
    Aj = A[j]

    # Initialize random subspace
    V = jax.random.normal(key, shape=(k, A.shape[1]), dtype=A.dtype)

    # Align with largest row
    dots = jnp.sum(Aj * jnp.conj(V), axis=1, keepdims=True)
    V = Aj + jnp.sign(jnp.real(dots)) * V

    # Subspace iteration
    def iteration_body(V, _):
        V = V @ A
        V = V / (jnp.linalg.norm(V, axis=1, keepdims=True) + smallest_normal)
        V = V @ A
        return V, None

    V, _ = jax.lax.scan(iteration_body, V, None, length=half_iters)

    return normalizing_factor * jnp.max(jnp.linalg.norm(V, axis=1))


def _norm_lower_bound_skh(
    A: jax.Array, key: jax.Array, k: int = 32, half_iters: int = 2
) -> jax.Array:
    """
    Returns a cheap lower bound for the spectral norm of a skew-Hermitian matrix A.
    Used in Procrustes step for normalizing the generator R.
    """
    smallest_normal = jnp.finfo(A.dtype).smallest_normal
    normalizing_factor = jnp.max(jnp.abs(A)) + smallest_normal
    A = A / normalizing_factor

    row_norms = jnp.linalg.norm(A, axis=1)
    j = jnp.argmax(row_norms)
    Aj = A[j]

    V = jax.random.normal(key, shape=(k, A.shape[1]), dtype=A.dtype)

    dots = jnp.sum(Aj * jnp.conj(V), axis=1, keepdims=True)
    V = Aj + jnp.sign(jnp.real(dots)) * V

    def iteration_body(V, _):
        V = V @ A
        V = V / (jnp.linalg.norm(V, axis=1, keepdims=True) + smallest_normal)
        V = V @ A
        return V, None

    V, _ = jax.lax.scan(iteration_body, V, None, length=half_iters)

    return normalizing_factor * jnp.max(jnp.linalg.norm(V, axis=1))


def _procrustes_step2(
    Q: jax.Array, key: jax.Array, max_step_size: float = 1 / 8
) -> jax.Array:
    """
    Online solver for the orthogonal Procrustes problem:
        min_U || U Q - I ||_F,   s.t. U^H U = I
    by rotating Q as exp(a R) Q, where R = Q^H - Q is the generator.

    Expands exp(a R) to 2nd order: I + aR + (aR)^2/2
    Truncation error bounded by ||a R||^4/4. Set max_step_size <= 1/4.

    Args:
        Q: Matrix to be made SPD
        max_step_size: Maximum step size (default 1/8)

    Returns:
        Updated Q closer to being SPD
    """
    R = jnp.conj(Q.T) - Q  # Skew-Hermitian generator
    R_norm = _norm_lower_bound_skh(R, key) + jnp.finfo(R.dtype).smallest_normal
    R = R / R_norm  # Normalize R

    RQ = R @ Q
    RRQ = R @ RQ

    tr_RQ = jnp.real(jnp.sum(jnp.diag(RQ)))  # tr(RQ) >= 0 by theory
    tr_RRQ = jnp.real(jnp.sum(jnp.diag(RRQ)))

    # Line search if tr_RRQ < 0
    a = jnp.where(
        tr_RRQ < 0, jnp.minimum(-tr_RQ / tr_RRQ, max_step_size), max_step_size
    )

    return Q + a * (RQ + 0.5 * a * RRQ)


def _init_Q_exprs(
    t,
    scale,
    max_size,
    max_skew,
    min_ndim_triangular,
    memory_save_mode,
    dtype,
    existing_Q=None,
) -> Tuple[List[jax.Array], Tuple[str, Tuple[str, ...], str]]:
    """
    Initialize preconditioner Q and einsum expressions for a tensor t.

    Returns:
        [Q, (exprA, exprGs, exprP)] where:
        - Q: List of preconditioner factors (diagonal vectors or triangular matrices)
        - exprA: Expression for applying all Q factors
        - exprGs: Expressions for gradient contractions (one per dim)
        - exprP: Expression for full preconditioning P = Q^H Q
    """
    letters = string.ascii_lowercase + string.ascii_uppercase

    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = (
            [scale * jnp.ones_like(t, dtype=dtype)]
            if existing_Q is None
            else existing_Q
        )
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:  # tensor
        if len(shape) > 13:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!"
            )

        scale = scale ** (1 / len(shape))
        total_numel = t.size  # Total number of elements

        if memory_save_mode is None:
            dim_diag = [False for _ in shape]
        elif memory_save_mode == "one_diag":
            rev_sorted_dims = np.argsort(shape)[::-1]
            dim_diag = [False for _ in shape]
            dim_diag[rev_sorted_dims[0]] = True
        elif memory_save_mode == "all_diag":
            dim_diag = [True for _ in shape]
        else:
            raise ValueError(
                f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
                "[None, 'one_diag', 'all_diag']"
            )

        Q = [] if existing_Q is None else existing_Q
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")

        for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
            # Match PyTorch logic: size <= 1 or size > max_size or size**2 > max_skew * numel
            is_diagonal = (
                size <= 1
                or size > max_size
                or size**2 > max_skew * total_numel  # NEW CONDITION
                or len(shape) < min_ndim_triangular
                or dim_d
            )

            if is_diagonal:
                # Use diagonal matrix as preconditioner for this dim
                if existing_Q is None:
                    Q.append(scale * jnp.ones(size, dtype=dtype))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                exprGs.append(piece1 + "," + piece1 + "->" + letters[i + 13])

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # Use triangular matrix as preconditioner for this dim
                if existing_Q is None:
                    Q.append(scale * jnp.eye(size, dtype=dtype))

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                piece2 = "".join(
                    [
                        (letters[i + 26] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                exprGs.append(
                    piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                )

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = (
            ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        )

    exprGs = tuple(exprGs)
    if existing_Q is not None:
        return exprA, exprGs, exprP
    return [Q, (exprA, exprGs, exprP)]


def _solve_triangular_right(X: jax.Array, A: jax.Array) -> jax.Array:
    """Compute X @ inv(A) via triangular solve."""
    X_ndim = X.ndim
    if X_ndim < 2:
        X = X[None, :]

    dtype_in = jnp.promote_types(A.dtype, X.dtype)
    A, X = A.astype(dtype_in), X.astype(dtype_in)

    leading_dims = max(0, X.ndim - 2)
    solve_fn = partial(jax.lax.linalg.triangular_solve, left_side=False, lower=False)
    for _ in range(leading_dims):
        solve_fn = vmap(solve_fn, in_axes=(None, 0))
    solution = solve_fn(A, X)

    if X_ndim < 2:
        return solution[0]
    return solution


def _conjB(Q: List[jax.Array], G: jax.Array, V: jax.Array) -> jax.Array:
    """
    Compute conjB = V @ inv(Q) for EQ mode.
    This is the key operation that differentiates EQ from other modes.
    """
    order = G.ndim
    p = list(range(order))
    conjB = jnp.transpose(V.conj(), p[1:] + p[:1])

    for i, q in enumerate(Q):
        if q.ndim < 2:
            conjB = conjB / q
        else:
            conjB = _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = jnp.swapaxes(conjB, i, order - 1)

    return conjB


def _update_precond_eq(
    Q: List[jax.Array],
    L: Optional[List[jax.Array]],
    G: jax.Array,
    conjB: jax.Array,
    exprs: Tuple[str, Tuple[str, ...], str],
    precond_lr: float,
    key: jax.Array,
    beta_l: float = 0.9,
) -> Tuple[List[jax.Array], Optional[List[jax.Array]]]:
    """
    Update preconditioner Q using EQ mode: dQ = E * Q.
    """
    exprA, exprGs, _ = exprs
    A = jnp.einsum(exprA, *Q, G)
    A_conj = A.conj()
    conjB_conj = conjB.conj()

    new_Qs = []
    new_Ls = [] if L is not None else None

    keys = jax.random.split(key, len(Q))

    for i, q in enumerate(Q):
        l = L[i] if L is not None else None
        term1 = jnp.einsum(exprGs[i], A, A_conj)
        term2 = jnp.einsum(exprGs[i], conjB_conj, conjB)

        if q.ndim < 2:  # Diagonal preconditioner
            tmp = (term1 - term2) * precond_lr

            if l is not None:
                ell = jnp.max(jnp.real(term1 + term2))
                new_l = jnp.maximum(beta_l * l + (1 - beta_l) * ell, ell)
                tmp = tmp / new_l
            else:
                tmp = tmp / _add_tiny(jnp.max(jnp.abs(term1 + term2)))
                new_l = None

            new_q = q - tmp * q
        else:  # Matrix preconditioner
            tmp = jnp.triu(term1 - term2) * precond_lr

            if l is not None:
                ell = _norm_lower_bound_spd(term1 + term2, keys[i])
                new_l = jnp.maximum(beta_l * l + (1 - beta_l) * ell, ell)
                tmp = tmp / new_l
            else:
                tmp = tmp / _add_tiny(_norm_lower_bound_spd(term1 + term2, key))
                new_l = None

            new_q = q - tmp @ q

        new_Qs.append(new_q)
        if new_Ls is not None:
            new_Ls.append(new_l)

    return new_Qs, new_Ls


def _update_precond_q0p5eq1p5(
    Q: List[jax.Array],
    L: Optional[List[jax.Array]],
    Pg: jax.Array,
    total_numel: int,
    exprs: Tuple[str, Tuple[str, ...], str],
    precond_lr: float,
    key: jax.Array,
    beta_l: float = 0.9,
) -> Tuple[List[jax.Array], Optional[List[jax.Array]]]:
    """
    Update preconditioner Q using Q0.5EQ1.5 mode: dQ = Q^0.5 * E * Q^1.5.
    """
    _, exprGs, _ = exprs
    Pg_conj = Pg.conj()

    new_Qs = []
    new_Ls = [] if L is not None else None

    keys = jax.random.split(key, len(Q))

    for i, q in enumerate(Q):
        l = L[i] if L is not None else None
        term1 = jnp.einsum(exprGs[i], Pg, Pg_conj)

        if q.ndim < 2:  # Diagonal preconditioner
            term2 = total_numel / q.size
            ell = jnp.max(jnp.real(term1)) + term2

            if l is not None:
                new_l = jnp.maximum(beta_l * l + (1 - beta_l) * ell, ell)
                new_q = q * (1 - precond_lr / new_l * (term1 - term2))
            else:
                new_q = q * (1 - precond_lr / _add_tiny(ell) * (term1 - term2))
                new_l = None
        else:  # Matrix preconditioner
            key_norm, key_proc = jax.random.split(keys[i])

            term2 = total_numel / q.shape[0]
            ell = _norm_lower_bound_spd(term1, key_norm) + term2

            if l is not None:
                new_l = jnp.maximum(beta_l * l + (1 - beta_l) * ell, ell)
                new_q = q - precond_lr / new_l * (term1 @ q - term2 * q)
            else:
                new_q = q - precond_lr / _add_tiny(ell) * (term1 @ q - term2 * q)
                new_l = None

            # Apply Procrustes step to maintain SPD property
            new_q = _procrustes_step2(new_q, key_proc)

        new_Qs.append(new_q)
        if new_Ls is not None:
            new_Ls.append(new_l)

    return new_Qs, new_Ls


def _update_precond_quad(
    Q: List[jax.Array],
    L: Optional[List[jax.Array]],
    Pg: jax.Array,
    total_numel: int,
    exprs: Tuple[str, Tuple[str, ...], str],
    precond_lr: float,
    key: jax.Array,
    beta_l: float = 0.9,
) -> Tuple[List[jax.Array], Optional[List[jax.Array]]]:
    """
    Update preconditioner Q using QUAD mode with a quadratic form.

    This method ensures Q remains symmetric/Hermitian positive definite
    through the quadratic update structure.

    For diagonal Q: q_new = q * gain^2, where gain = 1 - (lr/2/L) * (term1 - term2)
    For matrix Q: Two-step update with Hermitian symmetrization (p + p^H) / 2
    """
    _, exprGs, _ = exprs
    Pg_conj = Pg.conj()

    new_Qs = []
    new_Ls = [] if L is not None else None

    keys = jax.random.split(key, len(Q))

    for i, q in enumerate(Q):
        l = L[i] if L is not None else None
        term1 = jnp.einsum(exprGs[i], Pg, Pg_conj)

        if q.ndim < 2:  # Diagonal or scalar Q
            term2 = total_numel / q.size  # times I
            ell = jnp.max(jnp.real(term1)) + term2

            if l is not None:
                new_l = jnp.maximum(beta_l * l + (1 - beta_l) * ell, ell)
                # Quadratic form: gain = 1 - lr/2/L * (term1 - term2), then q *= gain^2
                gain = 1 - precond_lr / 2 / new_l * (term1 - term2)
                new_q = q * (gain * gain)
            else:
                gain = 1 - precond_lr / 2 / _add_tiny(ell) * (term1 - term2)
                new_q = q * (gain * gain)
                new_l = None
        else:  # Matrix Q
            term2 = total_numel / q.shape[0]  # times I
            ell = _norm_lower_bound_spd(term1, keys[i]) + term2

            if l is not None:
                new_l = jnp.maximum(beta_l * l + (1 - beta_l) * ell, ell)
                half_lr = precond_lr / 2 / new_l
            else:
                half_lr = precond_lr / 2 / _add_tiny(ell)
                new_l = None

            # Two-step quadratic update:
            # p = q - (lr/2/L) * (term1 @ q - term2 * q)
            # p = p - (lr/2/L) * (p @ term1 - p * term2)
            # q_new = (p + p^H) / 2
            p = q - half_lr * (term1 @ q - term2 * q)
            p = p - half_lr * (p @ term1 - p * term2)
            new_q = (p + jnp.conj(p.T)) / 2  # Hermitian symmetrization

        new_Qs.append(new_q)
        if new_Ls is not None:
            new_Ls.append(new_l)

    return new_Qs, new_Ls


def _precond_grad(
    Q: List[jax.Array],
    G: jax.Array,
    exprs: Tuple[str, Tuple[str, ...], str],
) -> jax.Array:
    """Precondition gradient G with preconditioner Q: P @ G = Q^H @ Q @ G."""
    exprP = exprs[-1]
    return jnp.einsum(exprP, *[q.conj() for q in Q], *Q, G)


def _balance_Q(Q: List[jax.Array]) -> List[jax.Array]:
    """
    Balance the dynamic ranges of Q factors to avoid overflow/underflow.
    Scales factors so their max absolute values have equal geometric mean.
    """
    if len(Q) <= 1:
        return Q

    norms = jnp.array([jnp.max(jnp.abs(q)) for q in Q], dtype=jnp.float32)
    gmean = jnp.prod(norms) ** (1.0 / len(norms))
    to_mul = gmean / norms

    return [q * x.astype(q.dtype) for q, x in zip(Q, to_mul)]


def scale_by_kron(
    b1: float = 0.9,
    preconditioner_update_probability: Union[
        float, Callable[[int], float]
    ] = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    max_skew_triangular: float = 1.0,
    min_ndim_triangular: int = 2,
    memory_save_mode: Optional[str] = None,
    momentum_into_precond_update: bool = True,
    preconditioner_lr: float = 0.1,
    preconditioner_init_scale: float = 1.0,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_update_precision: Optional[str] = "tensorfloat32",
    precond_grads_precision: Optional[str] = None,
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    # New parameters for mode selection
    preconditioner_mode: Union[str, PreconditionerMode] = PreconditionerMode.Q0P5EQ1P5,
    beta_lipschitz: float = 0.9,
    track_lipschitz: bool = True,
    damping: float = 1e-9,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformationExtraArgs:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        b1: float, momentum parameter.
        preconditioner_update_probability: float or callable, probability of updating
            the preconditioner. Default anneals from 1.0 to 0.03 by 4000 steps.
        max_size_triangular: int, max size for dim's preconditioner to be triangular.
        max_skew_triangular: float, max skew for dim's preconditioner to be triangular.
            A dimension uses diagonal preconditioner if size**2 > max_skew * numel.
        min_ndim_triangular: int, minimum number of dimensions a layer needs to have
            triangular preconditioners.
        memory_save_mode: optional str, None, 'one_diag', or 'all_diag'.
        momentum_into_precond_update: bool, whether to send momentum into preconditioner
            update instead of raw gradients.
        preconditioner_lr: float, learning rate for preconditioner.
        preconditioner_init_scale: float, scale for preconditioner initialization.
        mu_dtype: optional dtype for momentum accumulator.
        precond_dtype: optional dtype for preconditioner.
        precond_update_precision: str, precision for matmul during preconditioner update.
        precond_grads_precision: str, precision for matmul during preconditioning grads.
        scanned_layers: optional tree of bool indicating scanned layers.
        lax_map_scanned_layers: bool, whether to use lax.map instead of vmap.
        lax_map_batch_size: int, batch size for lax.map.
        preconditioner_mode: str or PreconditionerMode, geometry for Q updates.
            'EQ': dQ = E * Q (triangular, original PSGD)
            'Q0.5EQ1.5': dQ = Q^0.5 * E * Q^1.5 (recommended, uses Procrustes)
        beta_lipschitz: float, EMA factor for Lipschitz constant estimation.
        track_lipschitz: bool, whether to track L per Q factor (PyTorch-style).
        damping: float, damping for numerical stability.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    mu_dtype = canonicalize_dtype(mu_dtype)
    precond_dtype = canonicalize_dtype(precond_dtype)

    # Normalize mode
    if isinstance(preconditioner_mode, str):
        mode_map = {
            "EQ": PreconditionerMode.EQ,
            "Q0.5EQ1.5": PreconditionerMode.Q0P5EQ1P5,
            "Q0p5EQ1p5": PreconditionerMode.Q0P5EQ1P5,
            "QUAD": PreconditionerMode.QUAD,
        }
        preconditioner_mode = mode_map.get(
            preconditioner_mode, PreconditionerMode.Q0P5EQ1P5
        )

    def map_fn(do_map, fn, *args):
        """Maybe map a fn along first axis."""
        if do_map:
            if lax_map_scanned_layers:
                return jax.lax.map(
                    lambda xs: fn(*xs),
                    xs=args,
                    batch_size=lax_map_batch_size if lax_map_batch_size > 1 else None,
                )
            else:
                return vmap(fn)(*args)
        else:
            return fn(*args)

    def init_fn(params):
        scanned_layers_ = scanned_layers
        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, params)

        # Momentum
        mu = None
        if b1 > 0:
            mu = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=mu_dtype), params)

        # Preconditioners - keep as flat list initially
        Qs = [
            _init_Q_exprs(
                t[0] if s else t,
                preconditioner_init_scale,
                max_size_triangular,
                max_skew_triangular,
                min_ndim_triangular,
                memory_save_mode,
                precond_dtype,
            )[0]
            for t, s in zip(jax.tree.leaves(params), jax.tree.leaves(scanned_layers_))
        ]

        # Broadcast for scanned layers
        Qs = [
            (
                jax.tree.map(
                    lambda d: jnp.repeat(jnp.expand_dims(d, 0), t.shape[0], axis=0), q
                )
                if s
                else q
            )
            for q, t, s in zip(
                Qs, jax.tree.leaves(params), jax.tree.leaves(scanned_layers_)
            )
        ]

        # Lipschitz constants - initialize BEFORE unflattening Qs
        # At this point, Qs is a flat list of lists: [[q1_f1, q1_f2], [q2_f1], ...]
        Ls = None
        if track_lipschitz:
            Ls = [
                [jnp.zeros([], dtype=jnp.float32) for _ in q]  # q is a list here
                for q in Qs
            ]
            # Broadcast for scanned layers
            Ls = [
                (
                    [jnp.repeat(jnp.expand_dims(l, 0), t.shape[0], axis=0) for l in ls]
                    if s
                    else ls
                )
                for ls, t, s in zip(
                    Ls, jax.tree.leaves(params), jax.tree.leaves(scanned_layers_)
                )
            ]
            Ls = jax.tree.structure(params).unflatten(Ls)

        # NOW unflatten Qs to match params structure
        Qs = jax.tree.structure(params).unflatten(Qs)

        # Log sizes
        Qs_n_elements = sum([q.size for q in jax.tree.leaves(Qs)])
        Qs_size_MB = sum(
            [q.size * q.dtype.itemsize / (2**20) for q in jax.tree.leaves(Qs)]
        )
        if jax.process_index() == 0:
            print(
                f"PSGD Preconditioners ({preconditioner_mode.value} mode): "
                f"{Qs_n_elements} elements, {Qs_size_MB:.2f} MB"
            )
        if mu is not None:
            mu_n_elements = sum([p.size for p in jax.tree.leaves(mu)])
            mu_size_MB = sum(
                [p.size * p.dtype.itemsize / (2**20) for p in jax.tree.leaves(mu)]
            )
            if jax.process_index() == 0:
                print(
                    f"PSGD Momentum size: {mu_n_elements} elements, {mu_size_MB:.2f} MB"
                )

        return KronState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            Qs_preconditioners=Qs,
            Ls_lipschitz=Ls,
            update_counter=jnp.zeros([], jnp.int32),
            key=key,
        )

    def update_fn(updates: base.Updates, state: KronState, params: base.Params = None):
        del params
        count_inc = safe_int32_increment(state.count)
        key, key_next = jax.random.split(state.key)

        scanned_layers_ = scanned_layers
        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, updates)

        update_prob_in = preconditioner_update_probability
        if isinstance(preconditioner_update_probability, Callable):
            update_prob_in = preconditioner_update_probability(count_inc)

        # Momentum
        mu = None
        momentum_updates = updates
        if state.mu is not None:
            mu = otu.tree_update_moment(updates, state.mu, b1, 1)
            momentum_updates = otu.tree_bias_correction(mu, b1, count_inc)

        # Flatten pytrees
        updates, grads_structure = jax.tree.flatten(updates)
        momentum_updates = grads_structure.flatten_up_to(momentum_updates)
        Qs = grads_structure.flatten_up_to(state.Qs_preconditioners)
        scanned_layers_ = grads_structure.flatten_up_to(scanned_layers_)

        # Handle Ls consistently - keep as None if not tracking, else flatten
        Ls = None
        if track_lipschitz and state.Ls_lipschitz is not None:
            Ls = grads_structure.flatten_up_to(state.Ls_lipschitz)

        # Get einsum expressions
        expressions = [
            _init_Q_exprs(
                t[0] if s else t,
                preconditioner_init_scale,
                max_size_triangular,
                max_skew_triangular,
                min_ndim_triangular,
                memory_save_mode,
                precond_dtype,
                existing_Q=jax.tree.map(lambda d: d[0], Q) if s else Q,
            )
            for t, s, Q in zip(updates, scanned_layers_, Qs)
        ]

        def update_preconditioner(key, Qs, Ls):
            with jax.default_matmul_precision(precond_update_precision):
                precond_updates_in = (
                    momentum_updates if momentum_into_precond_update else updates
                )

                # Balance preconditioners about every 100 updates
                key, key_balance = jax.random.split(key)

                def balance_Qs(Qs: List[List[jax.Array]]):
                    return [
                        map_fn(s, _balance_Q, Q) if len(Q) > 1 else Q
                        for Q, s in zip(Qs, scanned_layers_)
                    ]

                do_balances = jax.random.uniform(key_balance) < 0.01
                Qs = jax.lax.cond(do_balances, balance_Qs, lambda qs: qs, Qs)

                # Create random vectors for damping
                key, key_noise = jax.random.split(key)
                Vs_keys = jax.random.split(key_noise, len(precond_updates_in))
                Vs = [
                    jax.random.normal(k, shape=g.shape, dtype=g.dtype)
                    for k, g in zip(Vs_keys, precond_updates_in)
                ]

                # Apply damping: G + (damping + eps * |G|) * V
                eps = jnp.finfo(jnp.float32).eps
                precond_updates_in = jax.tree.map(
                    lambda g, v: g + (damping + eps * jnp.abs(g)) * v,
                    precond_updates_in,
                    Vs,
                )

                key, key_updates = jax.random.split(key)
                layer_keys = jax.random.split(key_updates, len(Qs))

                if preconditioner_mode == PreconditionerMode.EQ:
                    # EQ mode: compute conjB and use triangular update
                    conjBs = [
                        map_fn(s, _conjB, Q, g, v)
                        for s, Q, g, v in zip(
                            scanned_layers_, Qs, precond_updates_in, Vs
                        )
                    ]

                    if track_lipschitz:

                        def _update_eq_with_L(Q, L, g, conjb, exprs, layer_key):
                            return _update_precond_eq(
                                Q,
                                L,
                                g,
                                conjb,
                                exprs,
                                preconditioner_lr,
                                layer_key,
                                beta_lipschitz,
                            )

                        results = [
                            map_fn(
                                s,
                                partial(_update_eq_with_L, exprs=exprs, layer_key=lk),
                                Q,
                                L,
                                g,
                                conjb,
                            )
                            for s, exprs, Q, L, g, conjb, lk in zip(
                                scanned_layers_,
                                expressions,
                                Qs,
                                Ls,
                                precond_updates_in,
                                conjBs,
                                layer_keys,
                            )
                        ]
                        new_Qs = [r[0] for r in results]
                        new_Ls = [r[1] for r in results]
                    else:

                        def _update_eq_no_L(Q, g, conjb, exprs, layer_key):
                            return _update_precond_eq(
                                Q,
                                None,
                                g,
                                conjb,
                                exprs,
                                preconditioner_lr,
                                layer_key,
                                beta_lipschitz,
                            )

                        results = [
                            map_fn(
                                s,
                                partial(_update_eq_no_L, exprs=exprs, layer_key=lk),
                                Q,
                                g,
                                conjb,
                            )
                            for s, exprs, Q, g, conjb, lk in zip(
                                scanned_layers_,
                                expressions,
                                Qs,
                                precond_updates_in,
                                conjBs,
                                layer_keys,
                            )
                        ]
                        new_Qs = [r[0] for r in results]
                        new_Ls = None

                elif preconditioner_mode in (
                    PreconditionerMode.Q0P5EQ1P5,
                    PreconditionerMode.QUAD,
                ):
                    # Both modes use Pg = P @ G first
                    Pgs = [
                        map_fn(s, partial(_precond_grad, exprs=exprs), Q, g)
                        for s, exprs, Q, g in zip(
                            scanned_layers_, expressions, Qs, precond_updates_in
                        )
                    ]

                    # Select the appropriate update function
                    if preconditioner_mode == PreconditionerMode.QUAD:
                        _update_fn = _update_precond_quad
                    else:
                        _update_fn = _update_precond_q0p5eq1p5

                    if track_lipschitz:

                        def _update_with_L(
                            Q, L, Pg, total_numel, exprs, layer_key, _fn=_update_fn
                        ):
                            return _fn(
                                Q,
                                L,
                                Pg,
                                total_numel,
                                exprs,
                                preconditioner_lr,
                                layer_key,
                                beta_lipschitz,
                            )

                        results = [
                            map_fn(
                                s,
                                partial(
                                    _update_with_L,
                                    total_numel=g.size,
                                    exprs=exprs,
                                    layer_key=lk,
                                ),
                                Q,
                                L,
                                Pg,
                            )
                            for s, exprs, Q, L, g, Pg, lk in zip(
                                scanned_layers_,
                                expressions,
                                Qs,
                                Ls,
                                precond_updates_in,
                                Pgs,
                                layer_keys,
                            )
                        ]

                        new_Qs = [r[0] for r in results]
                        new_Ls = [r[1] for r in results]
                    else:

                        def _update_no_L(
                            Q, Pg, total_numel, exprs, layer_key, _fn=_update_fn
                        ):
                            return _fn(
                                Q,
                                None,
                                Pg,
                                total_numel,
                                exprs,
                                preconditioner_lr,
                                layer_key,
                                beta_lipschitz,
                            )

                        results = [
                            map_fn(
                                s,
                                partial(
                                    _update_no_L,
                                    total_numel=g.size,
                                    exprs=exprs,
                                    layer_key=lk,
                                ),
                                Q,
                                Pg,
                            )
                            for s, exprs, Q, g, Pg, lk in zip(
                                scanned_layers_,
                                expressions,
                                Qs,
                                precond_updates_in,
                                Pgs,
                                layer_keys,
                            )
                        ]
                        new_Qs = [r[0] for r in results]
                        new_Ls = None

                new_Qs = otu.tree_cast(new_Qs, precond_dtype)
                return new_Qs, new_Ls

        # Update preconditioner deterministically
        update_counter_inc = safe_int32_increment(state.update_counter)
        do_update = update_counter_inc >= 1 / update_prob_in
        update_counter_inc = jnp.where(do_update, 0, update_counter_inc)

        key, subkey = jax.random.split(key)

        # Both branches must return same structure: (Qs, Ls) where Ls is None or list
        Qs, Ls = jax.lax.cond(
            do_update,
            update_preconditioner,
            lambda _key, qs, ls: (qs, ls),  # Identity - preserve structure
            subkey,
            Qs,
            Ls,
        )

        # Precondition gradients
        with jax.default_matmul_precision(precond_grads_precision):
            precond_gs = [
                map_fn(s, partial(_precond_grad, exprs=exprs), Q, g)
                for s, exprs, Q, g in zip(
                    scanned_layers_, expressions, Qs, momentum_updates
                )
            ]

        # Clip preconditioned gradients (RMS should be ~1.0, cap at 1.1)
        def _clip_fn(u):
            clip_denom = jnp.maximum(1.0, jnp.sqrt(jnp.mean(numerics.abs_sq(u))) / 1.1)
            return u / clip_denom

        precond_gs = jax.tree.map(_clip_fn, precond_gs)

        # Unflatten pytrees
        updates = grads_structure.unflatten(precond_gs)
        Qs = grads_structure.unflatten(Qs)
        Ls = grads_structure.unflatten(Ls) if Ls is not None else None

        # Dtypes and new state
        mu = otu.tree_cast(mu, mu_dtype)
        Qs = otu.tree_cast(Qs, precond_dtype)

        new_state = KronState(
            count=count_inc,
            mu=mu,
            Qs_preconditioners=Qs,
            Ls_lipschitz=Ls,
            update_counter=update_counter_inc,
            key=key_next,
        )

        return updates, new_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def kron(
    learning_rate: Union[float, Callable[[int], float]] = 0.001,
    b1: float = 0.9,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    preconditioner_update_probability: Union[
        float, Callable[[int], float]
    ] = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    max_skew_triangular: float = 1.0,
    min_ndim_triangular: int = 2,
    memory_save_mode: Optional[str] = None,
    momentum_into_precond_update: bool = True,
    preconditioner_lr: float = 0.1,
    preconditioner_init_scale: float = 1.0,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_update_precision: Optional[str] = "tensorfloat32",
    precond_grads_precision: Optional[str] = None,
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    preconditioner_mode: Union[str, PreconditionerMode] = PreconditionerMode.Q0P5EQ1P5,
    beta_lipschitz: float = 0.9,
    track_lipschitz: bool = True,
    damping: float = 1e-9,
    key: jax.Array = jax.random.PRNGKey(42),
) -> base.GradientTransformationExtraArgs:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate.
        b1: float, momentum parameter.
        weight_decay: float, weight decay.
        weight_decay_mask: optional pytree of bool for selective weight decay.
        preconditioner_update_probability: float or callable, probability of updating
            the preconditioner.
        max_size_triangular: int, max size for triangular preconditioner.
        max_skew_triangular: float, max skew for dim's preconditioner to be triangular.
            A dimension uses diagonal preconditioner if size**2 > max_skew * numel.
        min_ndim_triangular: int, minimum dimensions for triangular preconditioners.
        memory_save_mode: optional str, None, 'one_diag', or 'all_diag'.
        momentum_into_precond_update: bool, use momentum for preconditioner updates.
        preconditioner_lr: float, learning rate for preconditioner.
        preconditioner_init_scale: float, initialization scale for preconditioner.
        mu_dtype: optional dtype for momentum.
        precond_dtype: optional dtype for preconditioner.
        precond_update_precision: str, matmul precision for preconditioner updates.
        precond_grads_precision: str, matmul precision for gradient preconditioning.
        scanned_layers: optional tree of bool for scanned layers.
        lax_map_scanned_layers: bool, use lax.map instead of vmap.
        lax_map_batch_size: int, batch size for lax.map.
        preconditioner_mode: str or PreconditionerMode, 'EQ' or 'Q0.5EQ1.5'.
        beta_lipschitz: float, EMA factor for Lipschitz constant.
        track_lipschitz: bool, whether to track Lipschitz constants.
        damping: float, damping for numerical stability.
        key: jax.Array, PRNGKey

    Returns:
        optax.GradientTransformationExtraArgs
    """
    optimizer = [
        scale_by_kron(
            b1=b1,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            max_skew_triangular=max_skew_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            momentum_into_precond_update=momentum_into_precond_update,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            precond_update_precision=precond_update_precision,
            precond_grads_precision=precond_grads_precision,
            scanned_layers=scanned_layers,
            lax_map_scanned_layers=lax_map_scanned_layers,
            lax_map_batch_size=lax_map_batch_size,
            preconditioner_mode=preconditioner_mode,
            beta_lipschitz=beta_lipschitz,
            track_lipschitz=track_lipschitz,
            damping=damping,
            key=key,
        )
    ]
    if weight_decay > 0.0:
        optimizer.append(transform.add_decayed_weights(weight_decay, weight_decay_mask))
    optimizer.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*optimizer)
