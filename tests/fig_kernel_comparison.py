"""
Figures: Prior vs Posterior for each kernel on the Forrester 1D function.

Generates one figure per kernel (RBF, Matern 5/2, DotProduct, WhiteKernel),
each with two panels:
  Left:  samples from the GP prior
  Right: posterior conditioned on the same 5 Forrester observations

All figures share the same observed points and grid for direct comparison.
EPS-compatible (no transparency).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.linalg import cholesky, cho_solve

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

SAMPLE_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
BAND_PRIOR = "#d9d9d9"
BAND_POST = "#c6dbef"
N_SAMPLES = 5
JITTER = 1e-8

# ---------------------------------------------------------------------------
# Forrester function
# ---------------------------------------------------------------------------
def forrester(x):
    return ((6 * x - 2) ** 2) * np.sin(12 * x - 4)

# ---------------------------------------------------------------------------
# Kernel implementations
# ---------------------------------------------------------------------------
def kernel_rbf(X1, X2, length_scale=0.2, sigma_f=1.0):
    sq = np.sum(X1**2, 1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2**2, 1)
    return sigma_f**2 * np.exp(-0.5 * sq / length_scale**2)


def kernel_matern52(X1, X2, length_scale=0.2, sigma_f=1.0):
    diff = X1 - X2.T
    r = np.abs(diff) / length_scale
    sqrt5r = np.sqrt(5) * r
    return sigma_f**2 * (1 + sqrt5r + 5 * r**2 / 3) * np.exp(-sqrt5r)


def kernel_dotproduct(X1, X2, sigma_0=1.0):
    return sigma_0**2 + X1 @ X2.T


def kernel_white(X1, X2, sigma_n=1.0):
    """White noise kernel: sigma_n^2 * I  (only on matching points)."""
    if X1.shape == X2.shape and np.allclose(X1, X2):
        return sigma_n**2 * np.eye(X1.shape[0])
    # Cross-covariance is zero everywhere
    return np.zeros((X1.shape[0], X2.shape[0]))


# ---------------------------------------------------------------------------
# GP computations
# ---------------------------------------------------------------------------
def gp_prior_samples(K_grid, n_grid, rng):
    K = K_grid + JITTER * np.eye(n_grid)
    L = cholesky(K, lower=True)
    return L @ rng.standard_normal((n_grid, N_SAMPLES))


def gp_posterior(K_func, X_obs, y_obs, X_grid):
    n_obs = X_obs.shape[0]
    n_grid = X_grid.shape[0]

    K_XX = K_func(X_obs, X_obs) + JITTER * np.eye(n_obs)
    K_Xs = K_func(X_obs, X_grid)
    K_ss = K_func(X_grid, X_grid)

    L = cholesky(K_XX, lower=True)
    alpha = cho_solve((L, True), y_obs)
    mu = K_Xs.T @ alpha

    v = cho_solve((L, True), K_Xs)
    cov = K_ss - K_Xs.T @ v + JITTER * np.eye(n_grid)

    sigma = np.sqrt(np.maximum(np.diag(cov), 0.0))
    L_post = cholesky(cov, lower=True)

    return mu, sigma, cov, L_post


# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------
np.random.seed(42)
rng = np.random.default_rng(42)

n_grid = 300
X_grid = np.linspace(0, 1, n_grid).reshape(-1, 1)
y_true = forrester(X_grid.ravel())

# 5 observations spread across the domain
X_obs = np.array([[0.05], [0.25], [0.5], [0.75], [0.95]])
y_obs = forrester(X_obs.ravel())

# ---------------------------------------------------------------------------
# Kernel configs
# ---------------------------------------------------------------------------
kernels = [
    {
        "name": "RBF (Squared Exponential)",
        "fname": "rbf",
        "func": lambda X1, X2: kernel_rbf(X1, X2, length_scale=0.2, sigma_f=4.0),
        "params": "$\\ell=0.2,\\; \\sigma_f=4.0$",
    },
    {
        "name": "Mat\u00e9rn 5/2",
        "fname": "matern52",
        "func": lambda X1, X2: kernel_matern52(X1, X2, length_scale=0.2, sigma_f=4.0),
        "params": "$\\ell=0.2,\\; \\sigma_f=4.0,\\; \\nu=5/2$",
    },
    {
        "name": "Lineal (DotProduct)",
        "fname": "dotproduct",
        "func": lambda X1, X2: kernel_dotproduct(X1, X2, sigma_0=2.0),
        "params": "$\\sigma_0=2.0$",
    },
    {
        "name": "White Noise",
        "fname": "white",
        "func": lambda X1, X2: kernel_white(X1, X2, sigma_n=4.0),
        "params": "$\\sigma_n=4.0$",
    },
]

# ---------------------------------------------------------------------------
# Generate figures
# ---------------------------------------------------------------------------
for kern in kernels:
    K_func = kern["func"]
    K_grid = K_func(X_grid, X_grid)

    # --- Prior samples ---
    prior_samples = gp_prior_samples(K_grid, n_grid, rng)
    sigma_prior = np.sqrt(np.maximum(np.diag(K_grid), 0.0))

    # --- Posterior ---
    mu_post, sigma_post, _, L_post = gp_posterior(K_func, X_obs, y_obs, X_grid)
    post_samples = mu_post[:, None] + L_post @ rng.standard_normal((n_grid, N_SAMPLES))

    # --- Figure ---
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(12, 4.5),
        gridspec_kw={"wspace": 0.15},
    )

    # Left: prior
    ax_l.fill_between(
        X_grid.ravel(), -1.96 * sigma_prior, 1.96 * sigma_prior,
        color=BAND_PRIOR, alpha=1.0, label="IC 95%",
    )
    for i in range(N_SAMPLES):
        ax_l.plot(X_grid, prior_samples[:, i], color=SAMPLE_COLORS[i], lw=1.0)
    ax_l.axhline(0, color="0.5", lw=0.8, ls="--")
    ax_l.set_xlabel("$x$")
    ax_l.set_ylabel("$f(x)$")
    ax_l.set_title("(a) Muestras del prior", pad=8)
    ax_l.set_xlim(0, 1)
    ax_l.legend(loc="upper left", framealpha=1.0)

    # Right: posterior
    ax_r.fill_between(
        X_grid.ravel(),
        mu_post - 1.96 * sigma_post,
        mu_post + 1.96 * sigma_post,
        color=BAND_POST, alpha=1.0, label="IC 95%",
    )
    ax_r.plot(X_grid, y_true, color="0.6", lw=1.2, ls=":", label="Forrester real")
    for i in range(N_SAMPLES):
        ax_r.plot(X_grid, post_samples[:, i], color=SAMPLE_COLORS[i], lw=0.9)
    ax_r.plot(X_grid, mu_post, color="k", lw=2.0, label="Media posterior")
    ax_r.scatter(
        X_obs, y_obs, s=70, c="red", zorder=5,
        edgecolors="k", linewidths=0.8, label="Observaciones",
    )
    ax_r.set_xlabel("$x$")
    ax_r.set_title("(b) Posterior (Forrester 1D, 5 obs.)", pad=8)
    ax_r.set_xlim(0, 1)
    ax_r.legend(loc="lower left", framealpha=1.0, fontsize=8)

    # Suptitle with spacing
    fig.suptitle(
        f"Kernel {kern['name']}  ({kern['params']})",
        fontsize=14, fontweight="bold", y=1.0,
    )
    fig.subplots_adjust(top=0.85, wspace=0.15)

    # Independent y-limits per panel
    y_abs_prior = np.abs(prior_samples).max() * 1.15
    ax_l.set_ylim(-y_abs_prior, y_abs_prior)

    y_abs_post = max(
        np.abs(post_samples).max(),
        np.abs(y_true).max(),
        np.abs(y_obs).max(),
    ) * 1.15
    ax_r.set_ylim(-y_abs_post, y_abs_post)
    ax_r.set_ylabel("$f(x)$")

    # Save
    base = f"tests/fig_kernel_{kern['fname']}"
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.eps", format="eps", bbox_inches="tight")
    print(f"Saved: {base}.png / .pdf / .eps")
    plt.close(fig)

print("\nDone. All kernel figures generated.")
