# Exported from problem1.ipynb (run from the code/ folder). Edit the notebook, not this file.


# %% [markdown]
# # Problem 1(c): MLP for the conditional mean and standard deviation
#
# Network $g:\mathbb{R}^7\to\mathbb{R}^2$ with the link from 1(a), $f_\mu(x)=g_1(x)$ and
# $f_\sigma(x)=\exp(g_2(x))$, trained with the Gaussian negative log-likelihood from 1(b).
# Compared against two baselines on a held-out test set.

# %%
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 0
DATA = "../data/taxi_trips.parquet"   # path relative to this notebook (code/)
TARGET = "trip_duration"              # minutes
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)

# %% [markdown]
# ## Data

# %%
df = pd.read_parquet(DATA)
print(df.shape)
df.head()

# %%
df[TARGET].describe()

# %% [markdown]
# Random 70 / 15 / 15 split into training, validation (for early stopping) and test sets.

# %%
X = df.drop(columns=[TARGET]).to_numpy(np.float32)
y = df[TARGET].to_numpy(np.float32)

idx = np.random.default_rng(SEED).permutation(len(df))
n_train, n_val = int(0.70 * len(df)), int(0.15 * len(df))
tr, va, te = np.split(idx, [n_train, n_train + n_val])
X_tr, y_tr = X[tr], y[tr]
X_va, y_va = X[va], y[va]
X_te, y_te = X[te], y[te]
print(f"train {len(y_tr)}, validation {len(y_va)}, test {len(y_te)}")

# %% [markdown]
# ## Test metrics (in minutes)
#
# - **NLL**: average Gaussian negative log-likelihood per trip (constant included); scores both $f_\mu$ and $f_\sigma$.
# - **RMSE**: accuracy of the mean $f_\mu$ alone.
# - **95% coverage**: fraction of test trips inside $f_\mu \pm 1.96 f_\sigma$ (should be close to 0.95).

# %%
def gaussian_nll(y, mu, sigma):
    return float(np.mean(np.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2) + HALF_LOG_2PI)

def rmse(y, mu):
    return float(np.sqrt(np.mean((y - mu) ** 2)))

def coverage95(y, mu, sigma):
    return float(np.mean(np.abs(y - mu) <= 1.96 * sigma))

def evaluate(mu, sigma):
    return {"NLL": gaussian_nll(y_te, mu, sigma),
            "RMSE": rmse(y_te, mu),
            "95% coverage": coverage95(y_te, mu, sigma)}

results = {}

# %% [markdown]
# ## Baselines
#
# 1. **Constant**: ignore $x$; predict the training mean and standard deviation for every trip.
# 2. **Linear mean, constant std**: least-squares linear regression for $f_\mu$, and the training residual
#    standard deviation as a constant $f_\sigma$.

# %%
mu_const = np.full_like(y_te, y_tr.mean())
sd_const = np.full_like(y_te, y_tr.std())
results["Constant mean and std"] = evaluate(mu_const, sd_const)
results["Constant mean and std"]

# %%
A_tr = np.c_[X_tr, np.ones(len(X_tr))]
coef, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
sd_lin = float(np.std(y_tr - A_tr @ coef))      # residual std on the training set
mu_lin = np.c_[X_te, np.ones(len(X_te))] @ coef
results["Linear mean, constant std"] = evaluate(mu_lin, np.full_like(y_te, sd_lin))
results["Linear mean, constant std"]

# %% [markdown]
# ## MLP
#
# Three hidden layers of width 128 with ReLU, and 2 outputs passed through the 1(a) link.
# $g_2$ is clamped to $[-10, 10]$ for numerical safety.

# %%
class MeanStdMLP(nn.Module):
    def __init__(self, d_in, d_hidden=128, n_hidden=3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(n_hidden):
            layers += [nn.Linear(d, d_hidden), nn.ReLU()]
            d = d_hidden
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        mu = out[:, 0]                          # f_mu = g_1 (identity link)
        log_sigma = out[:, 1].clamp(-10, 10)    # f_sigma = exp(g_2)
        return mu, log_sigma


def nll_loss(y, mu, log_sigma):
    # Loss from 1(b): mean of log f_sigma + (y - f_mu)^2 / (2 f_sigma^2), constant dropped
    return (log_sigma + 0.5 * (y - mu) ** 2 * torch.exp(-2 * log_sigma)).mean()

# %% [markdown]
# Training: Adam (learning rate $10^{-3}$), batch size 1024, up to 50 epochs, early stopping when the
# validation loss has not improved for 5 epochs (the best epoch is kept). Targets are standardized with the
# training mean and standard deviation $(\bar y, s)$ for training and converted back afterwards:
# $\mu = \bar y + s\,\tilde\mu$, $\sigma = s\,\tilde\sigma$.

# %%
torch.manual_seed(SEED)
LR, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 1024, 50, 5

y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
Xt, yt = torch.from_numpy(X_tr), torch.from_numpy((y_tr - y_mean) / y_std)
Xv, yv = torch.from_numpy(X_va), torch.from_numpy((y_va - y_mean) / y_std)

model = MeanStdMLP(X_tr.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=LR)
history, best_val, best_epoch, best_state, bad = [], math.inf, 0, None, 0
t0 = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Xt))
    total = 0.0
    for i in range(0, len(Xt), BATCH):
        b = perm[i:i + BATCH]
        loss = nll_loss(yt[b], *model(Xt[b]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(b)

    model.eval()
    with torch.no_grad():
        val_loss = nll_loss(yv, *model(Xv)).item()
    history.append({"epoch": epoch, "train loss": total / len(Xt), "val loss": val_loss})
    print(f"epoch {epoch:2d}   train {total / len(Xt):.4f}   val {val_loss:.4f}")

    if val_loss < best_val - 1e-4:
        best_val, best_epoch, bad = val_loss, epoch, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"early stopping: no improvement for {PATIENCE} epochs")
            break

model.load_state_dict(best_state)
model.eval()
print(f"trained in {time.time() - t0:.0f}s; keeping epoch {best_epoch} (val loss {best_val:.4f})")

# %% [markdown]
# ## Test-set results

# %%
with torch.no_grad():
    mu_s, log_sigma_s = model(torch.from_numpy(X_te))
mu_mlp = y_mean + y_std * mu_s.numpy()       # back to minutes
sd_mlp = y_std * np.exp(log_sigma_s.numpy())

results["MLP (f_mu, f_sigma)"] = evaluate(mu_mlp, sd_mlp)
pd.DataFrame(results).T.round(3)

# %% [markdown]
# Spread of the MLP's predicted standard deviation on the test set (minutes):

# %%
pd.Series(sd_mlp, name="predicted sigma").describe(percentiles=[0.05, 0.5, 0.95]).round(2)

# %% [markdown]
# # Problem 1(e): evaluating $P(Y < 45 \mid X = x)$ on the test set
#
# Predicted probability from 1(d): $p_i = \Phi\big((45 - f_\mu(x_i)) / f_\sigma(x_i)\big)$.
# True label: $z_i = 1$ if the trip actually took less than 45 minutes.
#
# - **Brier score** $= \frac1n\sum_i (p_i - z_i)^2$ (lower is better)
# - **Log-loss** $= -\frac1n\sum_i [z_i \log p_i + (1 - z_i)\log(1 - p_i)]$ (lower is better; $p_i$ clipped to $[10^{-15}, 1 - 10^{-15}]$)
# - **Accuracy**: predict "under 45" when $p_i \ge 0.5$, compare with $z_i$
#
# Baselines: a constant probability equal to the training fraction of trips under 45 minutes, and the two 1(c)
# baselines passed through the same formula.

# %%
from scipy.stats import norm

THRESH = 45.0
z_te = (y_te < THRESH).astype(np.float64)       # true label: trip under 45 minutes
base_rate = float((y_tr < THRESH).mean())       # training fraction under 45 minutes
print(f"fraction of trips under 45 min: train {base_rate:.4f}, test {z_te.mean():.4f}")
print(f"test trips taking 45+ min: {int((1 - z_te).sum())} of {len(z_te)}")

def prob_metrics(p, z=z_te, eps=1e-15):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return {"Brier": float(np.mean((p - z) ** 2)),
            "Log-loss": float(-np.mean(z * np.log(p) + (1 - z) * np.log(1 - p))),
            "Accuracy": float(np.mean((p >= 0.5) == (z == 1)))}

probs = {
    "Constant (training base rate)": np.full(len(z_te), base_rate),
    "Constant Gaussian (1(c) baseline 1)": norm.cdf(THRESH, loc=mu_const, scale=sd_const),
    "Linear Gaussian (1(c) baseline 2)": norm.cdf(THRESH, loc=mu_lin, scale=sd_lin),
    "MLP from 1(c)": norm.cdf(THRESH, loc=mu_mlp, scale=sd_mlp),
}
pd.DataFrame({k: prob_metrics(v) for k, v in probs.items()}).T.round(4)

# %% [markdown]
# Average predicted probability of "under 45 minutes", split by what actually happened:

# %%
long = z_te == 0
pd.DataFrame({k: {"trips under 45": float(v[~long].mean()), "trips 45+": float(v[long].mean())}
              for k, v in probs.items()}).T.round(3)
