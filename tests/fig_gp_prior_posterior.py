"""
Figure: GP Prior vs Posterior (RBF kernel).

Left panel:  samples drawn from the prior  f ~ GP(0, k)
Right panel: samples drawn from the posterior after observing 2 points,
             plus the predictive mean and 95% confidence band.

Style inspired by Rasmussen & Williams (2006), Chapter 2.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.linalg import cholesky, cho_solve

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

# ---------------------------------------------------------------------------
# RBF kernel
# ---------------------------------------------------------------------------
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    sqdist = np.sum(X1**2, axis=1, keepdims=True) \
           - 2 * X1 @ X2.T \
           + np.sum(X2**2, axis=1)
    return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
np.random.seed(42)

n_grid = 200
n_samples = 5
length_scale = 1.0
sigma_f = 1.0
noise = 1e-8  # jitter for numerical stability

X_grid = np.linspace(-5, 5, n_grid).reshape(-1, 1)

# Two observed points on a simple curve
X_obs = np.array([[-2.0], [2.0]])
y_obs = np.sin(X_obs).ravel()  # simple curved function

# ---------------------------------------------------------------------------
# Prior samples
# ---------------------------------------------------------------------------
K_prior = rbf_kernel(X_grid, X_grid, length_scale, sigma_f)
K_prior += noise * np.eye(n_grid)
L_prior = cholesky(K_prior, lower=True)

prior_samples = L_prior @ np.random.randn(n_grid, n_samples)

# ---------------------------------------------------------------------------
# Posterior
# ---------------------------------------------------------------------------
K_XX = rbf_kernel(X_obs, X_obs, length_scale, sigma_f) + noise * np.eye(len(X_obs))
K_Xs = rbf_kernel(X_obs, X_grid, length_scale, sigma_f)
K_ss = rbf_kernel(X_grid, X_grid, length_scale, sigma_f)

L_XX = cholesky(K_XX, lower=True)
alpha = cho_solve((L_XX, True), y_obs)
mu_post = K_Xs.T @ alpha

v = cho_solve((L_XX, True), K_Xs)
cov_post = K_ss - K_Xs.T @ v
cov_post += noise * np.eye(n_grid)

sigma_post = np.sqrt(np.diag(cov_post))
L_post = cholesky(cov_post, lower=True)

posterior_samples = mu_post[:, None] + L_post @ np.random.randn(n_grid, n_samples)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]

fig, (ax_prior, ax_post) = plt.subplots(1, 2, figsize=(12, 4.5),
                                         sharey=True, gridspec_kw={"wspace": 0.08})

# --- Left: prior ---
for i in range(n_samples):
    ax_prior.plot(X_grid, prior_samples[:, i], color=COLORS[i],
                  alpha=1.0, linewidth=1.0)

# Prior confidence band (from diagonal of K_prior)
sigma_prior = sigma_f * np.ones(n_grid)
ax_prior.fill_between(X_grid.ravel(),
                       -1.96 * sigma_prior, 1.96 * sigma_prior,
                       color="#d9d9d9", alpha=1.0, label="IC 95%")
ax_prior.axhline(0, color="0.5", linewidth=0.8, linestyle="--")

ax_prior.set_xlabel("$x$")
ax_prior.set_ylabel("$f(x)$")
ax_prior.set_title("(a) Muestras del prior")
ax_prior.set_xlim(-5, 5)
ax_prior.set_ylim(-3.5, 3.5)
ax_prior.legend(loc="upper left", framealpha=1.0)

# --- Right: posterior ---
ax_post.fill_between(X_grid.ravel(),
                      mu_post - 1.96 * sigma_post,
                      mu_post + 1.96 * sigma_post,
                      color="#c6dbef", alpha=1.0, label="IC 95%")

for i in range(n_samples):
    ax_post.plot(X_grid, posterior_samples[:, i], color=COLORS[i],
                 alpha=1.0, linewidth=0.9)

ax_post.plot(X_grid, mu_post, color="k", linewidth=2.0, label="Media posterior")
ax_post.scatter(X_obs, y_obs, s=70, c="red", zorder=5,
                edgecolors="k", linewidths=0.8, label="Observaciones")

ax_post.set_xlabel("$x$")
ax_post.set_title("(b) Posterior tras 2 observaciones")
ax_post.set_xlim(-5, 5)
ax_post.legend(loc="upper left", framealpha=1.0)

fig.suptitle("Proceso Gaussiano con kernel RBF ($\\ell=1.0$, $\\sigma_f=1.0$)",
             fontsize=14, fontweight="bold")

fig.subplots_adjust(top=0.88, wspace=0.08)
plt.savefig("tests/fig_gp_prior_posterior.png", dpi=300, bbox_inches="tight")
plt.savefig("tests/fig_gp_prior_posterior.pdf", bbox_inches="tight")
plt.savefig("tests/fig_gp_prior_posterior.eps", format="eps", bbox_inches="tight")
print("Saved: tests/fig_gp_prior_posterior.png / .pdf / .eps")
plt.show()
