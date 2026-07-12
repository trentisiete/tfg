"""
Figure: Expected Improvement step-by-step on Forrester 1D.

Two rows, each with two panels:
  Row 1: GP posterior (3 initial points) | EI landscape + next point
  Row 2: GP posterior (4 points)         | EI landscape + next point

Shows how EI balances exploration and exploitation to find the minimum.
EPS-compatible (no transparency).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.linalg import cholesky, cho_solve
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

BAND_COLOR = "#c6dbef"
JITTER = 1e-8

# ---------------------------------------------------------------------------
# Forrester
# ---------------------------------------------------------------------------
def forrester(x):
    return ((6 * x - 2) ** 2) * np.sin(12 * x - 4)

# ---------------------------------------------------------------------------
# RBF kernel
# ---------------------------------------------------------------------------
def kernel_rbf(X1, X2, length_scale=0.15, sigma_f=5.0):
    sq = np.sum(X1**2, 1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2**2, 1)
    return sigma_f**2 * np.exp(-0.5 * sq / length_scale**2)

# ---------------------------------------------------------------------------
# GP posterior
# ---------------------------------------------------------------------------
def gp_posterior(X_obs, y_obs, X_grid, noise=1e-6):
    n = X_obs.shape[0]
    K_XX = kernel_rbf(X_obs, X_obs) + noise * np.eye(n)
    K_Xs = kernel_rbf(X_obs, X_grid)
    K_ss = kernel_rbf(X_grid, X_grid)

    L = cholesky(K_XX, lower=True)
    alpha = cho_solve((L, True), y_obs)
    mu = (K_Xs.T @ alpha).ravel()

    v = cho_solve((L, True), K_Xs)
    cov = K_ss - K_Xs.T @ v
    sigma = np.sqrt(np.maximum(np.diag(cov), 0.0))

    return mu, sigma

# ---------------------------------------------------------------------------
# Expected Improvement (minimization)
# ---------------------------------------------------------------------------
def expected_improvement(mu, sigma, y_best, xi=0.01):
    improvement = y_best - mu - xi
    z = improvement / np.clip(sigma, 1e-12, None)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
n_grid = 500
X_grid = np.linspace(0, 1, n_grid).reshape(-1, 1)
y_true = forrester(X_grid.ravel())

# 3 initial observations (cover the domain but miss the optimum region ~0.76)
X_init = np.array([[0.0], [0.4], [0.9]])
y_init = forrester(X_init.ravel())

# ---------------------------------------------------------------------------
# Step 1: predict with 3 points, compute EI, find next
# ---------------------------------------------------------------------------
mu1, sigma1 = gp_posterior(X_init, y_init, X_grid)
y_best1 = np.min(y_init)
ei1 = expected_improvement(mu1, sigma1, y_best1)
x_next1_idx = np.argmax(ei1)
x_next1 = X_grid[x_next1_idx, 0]
y_next1 = forrester(x_next1)

# ---------------------------------------------------------------------------
# Step 2: add the new point, repeat
# ---------------------------------------------------------------------------
X_step2 = np.vstack([X_init, [[x_next1]]])
y_step2 = np.append(y_init, y_next1)

mu2, sigma2 = gp_posterior(X_step2, y_step2, X_grid)
y_best2 = np.min(y_step2)
ei2 = expected_improvement(mu2, sigma2, y_best2)
x_next2_idx = np.argmax(ei2)
x_next2 = X_grid[x_next2_idx, 0]

# ---------------------------------------------------------------------------
# Figure: 2x2
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9), gridspec_kw={"hspace": 0.35, "wspace": 0.25})

def plot_gp_panel(ax, X_obs, y_obs, mu, sigma, step_label, x_next, y_next_true,
                  show_legend=True):
    """Left panel: GP posterior + true function."""
    x = X_grid.ravel()

    ax.fill_between(x, mu - 1.96 * sigma, mu + 1.96 * sigma,
                     color=BAND_COLOR, alpha=1.0, label="IC 95%")
    ax.plot(x, y_true, color="0.55", lw=1.2, ls=":", label="Forrester real")
    ax.plot(x, mu, color="k", lw=2.0, label="Media GP")
    ax.scatter(X_obs.ravel(), y_obs, s=80, c="red", zorder=5,
               edgecolors="k", linewidths=0.8, label="Observaciones")

    ax.axvline(x_next, color="#2166ac", lw=1.5, ls="--",
               label=f"$x_{{next}}={x_next:.3f}$")
    ax.scatter([x_next], [y_next_true], s=100, c="#2166ac", marker="*",
               zorder=6, edgecolors="k", linewidths=0.6)

    y_best = np.min(y_obs)
    ax.axhline(y_best, color="#b2182b", lw=1.0, ls="-.",
               label=f"$y_{{best}}={y_best:.2f}$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title(step_label, pad=10)
    ax.set_xlim(0, 1)
    if show_legend:
        ax.legend(loc="upper left", framealpha=1.0, fontsize=8)


def plot_ei_panel(ax, ei, x_next, step_label, show_legend=True):
    """Right panel: EI landscape."""
    x = X_grid.ravel()

    ax.fill_between(x, 0, ei, color="#fee0d2", alpha=1.0)
    ax.plot(x, ei, color="#de2d26", lw=1.8, label="$EI(x)$")

    ei_max = ei[np.argmax(ei)]
    ax.scatter([x_next], [ei_max], s=120, c="#de2d26", marker="v",
               zorder=5, edgecolors="k", linewidths=0.8,
               label=f"$\\arg\\max\\; EI = {x_next:.3f}$")
    ax.axvline(x_next, color="#2166ac", lw=1.5, ls="--")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$EI(x)$")
    ax.set_title(step_label, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    if show_legend:
        ax.legend(loc="upper right", framealpha=1.0, fontsize=9)


# Row 1: 3 initial points
plot_gp_panel(axes[0, 0], X_init, y_init, mu1, sigma1,
              "(a) GP posterior  —  3 observaciones iniciales",
              x_next1, y_next1)
plot_ei_panel(axes[0, 1], ei1, x_next1,
              "(b) Expected Improvement  —  Paso 1")

# Row 2: 4 points (after adding x_next1) — no legend (same as row 1)
plot_gp_panel(axes[1, 0], X_step2, y_step2, mu2, sigma2,
              f"(c) GP posterior  —  4 observaciones (+$x_{{new}}={x_next1:.3f}$)",
              x_next2, forrester(x_next2), show_legend=False)
plot_ei_panel(axes[1, 1], ei2, x_next2,
              "(d) Expected Improvement  —  Paso 2", show_legend=False)

fig.suptitle(
    "Proceso de Infill con Expected Improvement (Forrester 1D, kernel RBF)",
    fontsize=15, fontweight="bold",
)
fig.subplots_adjust(top=0.92)

# Save
base = "tests/fig_ei_demo"
fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{base}.pdf", bbox_inches="tight")
fig.savefig(f"{base}.eps", format="eps", bbox_inches="tight")
print(f"Saved: {base}.png / .pdf / .eps")
plt.close(fig)
