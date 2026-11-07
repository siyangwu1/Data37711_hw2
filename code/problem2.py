# Exported from problem2.ipynb (run from the code/ folder). Edit the notebook, not this file.


# %% [markdown]
# # Problem 2: encoding categorical variables (retail data)
#
# Features `StockCode` and `CustomerID` are categorical; the target `Quantity` is a positive, heavy-tailed integer.

# %%
import numpy as np
import pandas as pd

SEED = 0
DATA = "../data/retail.csv"   # path relative to this notebook (code/)
FEATURES = ["StockCode", "CustomerID"]
TARGET = "Quantity"

df = pd.read_csv(DATA, dtype={"StockCode": str, "CustomerID": str})
df[FEATURES] = df[FEATURES].astype("category")
print(df.shape)
df.head()

# %% [markdown]
# # Problem 2(a): cardinality and data split

# %%
card = df[FEATURES].nunique()
card

# %% [markdown]
# How many rows each category has (rare categories are hard to learn from):

# %%
pd.DataFrame({f: df[f].value_counts().describe(percentiles=[0.1, 0.5, 0.9]).round(1)
              for f in FEATURES})

# %% [markdown]
# **Is one-hot encoding feasible?** It would create one column per category. A dense float64 matrix of that
# size is far too large to fit in memory, so it would need a sparse matrix, and most columns would still rest on
# only a handful of rows.

# %%
n_cols = int(card.sum())
dense_gb = len(df) * n_cols * 8 / 1e9
print(f"one-hot columns: {n_cols:,}")
print(f"dense float64 one-hot matrix: {len(df):,} x {n_cols:,} = {dense_gb:,.1f} GB")

# %% [markdown]
# Random 70 / 15 / 15 split into training, validation (for early stopping) and test sets.

# %%
idx = np.random.default_rng(SEED).permutation(len(df))
n_train, n_val = int(0.70 * len(df)), int(0.15 * len(df))
tr, va, te = np.split(idx, [n_train, n_train + n_val])
train, val, test = df.iloc[tr], df.iloc[va], df.iloc[te]
print(f"train {len(train):,}, validation {len(val):,}, test {len(test):,}")

# %% [markdown]
# Share of test rows whose category never appears in the training set (encoders must handle these):

# %%
pd.Series({f: float((~test[f].isin(set(train[f]))).mean()) for f in FEATURES},
          name="unseen in train").round(4)

# %% [markdown]
# # Problem 2(b): target encoding
#
# **Target transformation / loss.** `Quantity` is heavy-tailed, so squared error on the raw scale would be
# dominated by a few enormous orders. All models in Problem 2 are trained with squared error on
# $\log(1 + \text{Quantity})$, which models relative (multiplicative) errors.
#
# **Test metrics** (used for every model in Problem 2):
# - **RMSE (log scale)**: RMSE of the prediction of $\log(1 + \text{Quantity})$; matches the training loss.
# - **MAE (units)**: mean absolute error after converting back with $\exp(\cdot) - 1$; readable in items.

# %%
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

y_tr = np.log1p(train[TARGET].to_numpy(np.float64))
y_va = np.log1p(val[TARGET].to_numpy(np.float64))
y_te = np.log1p(test[TARGET].to_numpy(np.float64))
q_te = test[TARGET].to_numpy(np.float64)

def evaluate(pred_log):
    return {"RMSE (log scale)": float(np.sqrt(np.mean((pred_log - y_te) ** 2))),
            "MAE (units)": float(np.mean(np.abs(np.expm1(pred_log) - q_te)))}

results = {}
print(train[TARGET].describe(percentiles=[0.5, 0.99]).round(1))

# %% [markdown]
# Baseline: predict the training mean of $\log(1 + \text{Quantity})$ for every row.

# %%
results["Baseline: constant"] = evaluate(np.full(len(y_te), y_tr.mean()))
results["Baseline: constant"]

# %% [markdown]
# **Encoding.** scikit-learn's `TargetEncoder` (equivalent to `category_encoders.TargetEncoder`, which is not
# installed here), fit on the training set only, encoding the mean of $\log(1 + \text{Quantity})$ per category.
# It shrinks rare categories toward the overall mean (`smooth="auto"`), uses 5-fold cross-fitting when encoding
# the training rows themselves (so a row's own target does not leak into its feature), and maps categories
# unseen in training to the overall mean.

# %%
te_enc = TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=SEED)
Z_tr = te_enc.fit_transform(train[FEATURES], y_tr)   # cross-fitted on the training set
Z_va = te_enc.transform(val[FEATURES])
Z_te = te_enc.transform(test[FEATURES])
pd.DataFrame(Z_tr[:5], columns=[f + " (encoded)" for f in FEATURES]).round(3)

# %% [markdown]
# **Model.** A histogram gradient-boosted tree regressor on the two encoded features (squared error on the log
# scale). Number of boosting rounds chosen by early stopping on our validation set.

# %%
best = None
for n_iter in [50, 100, 200, 400, 800]:
    gbm = HistGradientBoostingRegressor(max_iter=n_iter, learning_rate=0.1, max_leaf_nodes=31,
                                        early_stopping=False, random_state=SEED).fit(Z_tr, y_tr)
    val_rmse = float(np.sqrt(np.mean((gbm.predict(Z_va) - y_va) ** 2)))
    print(f"max_iter {n_iter:4d}: validation RMSE (log) {val_rmse:.4f}")
    if best is None or val_rmse < best[0]:
        best = (val_rmse, n_iter, gbm)
print(f"selected max_iter = {best[1]}")
gbm_te = best[2]

results["2(b) Target encoding + GBM"] = evaluate(gbm_te.predict(Z_te))
pd.DataFrame(results).T.round(4)

# %% [markdown]
# # Problem 2(c): neural network with an embedding layer per feature
#
# **Step 1: integer indices.** Each category seen at least twice in the training set gets an index $1, 2, \dots$;
# index 0 means "unknown". Categories seen only once in training are also mapped to 0, so the unknown embedding
# is actually trained; validation/test categories not seen in training then fall back to it.

# %%
import math
import time

import torch
import torch.nn as nn

def build_vocab(col):
    counts = train[col].value_counts()
    keep = counts[counts >= 2].index
    return {cat: i + 1 for i, cat in enumerate(keep)}      # 0 = unknown

vocab = {f: build_vocab(f) for f in FEATURES}
n_cats = {f: len(vocab[f]) for f in FEATURES}
print("categories with their own embedding:", n_cats)

def to_index(frame, f):
    return torch.tensor(frame[f].astype(object).map(vocab[f]).fillna(0).astype(np.int64).to_numpy())

idx = {name: [to_index(frame, f) for f in FEATURES]
       for name, frame in [("train", train), ("val", val), ("test", test)]}
yt = {name: torch.tensor(arr, dtype=torch.float32)
      for name, arr in [("train", y_tr), ("val", y_va), ("test", y_te)]}

# %% [markdown]
# **Steps 2-4: model.** One embedding table per feature (rows = categories + 1 unknown row, columns = embedding
# dimension), randomly initialized. The two looked-up vectors are concatenated and passed through an MLP
# (64 and 32 ReLU units) that predicts $\log(1 + \text{Quantity})$.

# %%
class EmbeddingNet(nn.Module):
    def __init__(self, n_stock, n_cust, dim):
        super().__init__()
        self.stock_emb = nn.Embedding(n_stock + 1, dim)    # one row per StockCode (+ unknown)
        self.cust_emb = nn.Embedding(n_cust + 1, dim)      # one row per CustomerID (+ unknown)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 1))

    def features(self, stock_idx, cust_idx):
        return torch.cat([self.stock_emb(stock_idx), self.cust_emb(cust_idx)], dim=1)

    def forward(self, stock_idx, cust_idx):
        return self.mlp(self.features(stock_idx, cust_idx)).squeeze(1)

# %% [markdown]
# **Step 5: training.** Squared error on $\log(1 + \text{Quantity})$ (as in 2(b)); AdamW (learning rate $10^{-3}$,
# weight decay $10^{-4}$), batch size 1024, up to 30 epochs, early stopping on validation RMSE (patience 3, best
# epoch kept). Gradients flow into the MLP and into the embedding rows looked up in each batch.

# %%
def rmse_on(model, split, extra=None):
    model.eval()
    with torch.no_grad():
        args = idx[split] if extra is None else idx[split] + [extra[split]]
        return float(torch.sqrt(torch.mean((model(*args) - yt[split]) ** 2)))

def train_net(model, extra=None, lr=1e-3, wd=1e-4, batch=1024, max_epochs=30, patience=3, verbose=True):
    # extra: optional dict split -> float tensor of extra inputs (used in 2(d))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(yt["train"])
    best, best_state, best_epoch, bad = math.inf, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            args = [t[b] for t in idx["train"]] + ([] if extra is None else [extra["train"][b]])
            loss = torch.mean((model(*args) - yt["train"][b]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = rmse_on(model, "val", extra)
        if verbose:
            print(f"  epoch {epoch:2d}  train RMSE {rmse_on(model, 'train', extra):.4f}  val RMSE {val:.4f}")
        if val < best - 1e-4:
            best, best_epoch, bad = val, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, best_epoch

# %% [markdown]
# **Choosing the embedding dimension** on the validation set:

# %%
sweep = {}
t0 = time.time()
for dim in [4, 8, 16, 32]:
    torch.manual_seed(SEED)
    print(f"embedding dimension {dim}")
    model, val_rmse, ep = train_net(EmbeddingNet(n_cats["StockCode"], n_cats["CustomerID"], dim))
    sweep[dim] = {"val RMSE (log)": val_rmse, "best epoch": ep, "model": model}
print(f"sweep took {time.time() - t0:.0f}s")
pd.DataFrame({d: {k: v for k, v in s.items() if k != "model"} for d, s in sweep.items()}).T

# %%
best_dim = min(sweep, key=lambda d: sweep[d]["val RMSE (log)"])
emb_net = sweep[best_dim]["model"]
emb_net.eval()
print(f"selected embedding dimension: {best_dim}")

with torch.no_grad():
    pred_emb = emb_net(*idx["test"]).numpy().astype(np.float64)
results["2(c) Embedding NN"] = evaluate(pred_emb)
pd.DataFrame(results).T.round(4)

# %% [markdown]
# # Problem 2(d): target encoding + embeddings
#
# Same network as 2(c) (embedding dimension 32, the value selected there), but the MLP also receives the two
# target-encoded features from 2(b). The training rows use the cross-fitted encodings (`Z_tr`), so a row's own
# target never leaks into its input. The two encoded columns are standardized with the training mean and
# standard deviation. Training is identical to 2(c).

# %%
mu_z, sd_z = Z_tr.mean(axis=0), Z_tr.std(axis=0)
te_feat = {name: torch.tensor((Z - mu_z) / sd_z, dtype=torch.float32)
           for name, Z in [("train", Z_tr), ("val", Z_va), ("test", Z_te)]}

class CombinedNet(nn.Module):
    def __init__(self, n_stock, n_cust, dim, n_extra=2):
        super().__init__()
        self.stock_emb = nn.Embedding(n_stock + 1, dim)
        self.cust_emb = nn.Embedding(n_cust + 1, dim)
        self.mlp = nn.Sequential(nn.Linear(2 * dim + n_extra, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 1))

    def forward(self, stock_idx, cust_idx, te):
        v = torch.cat([self.stock_emb(stock_idx), self.cust_emb(cust_idx), te], dim=1)
        return self.mlp(v).squeeze(1)

torch.manual_seed(SEED)
comb_net, comb_val, comb_epoch = train_net(
    CombinedNet(n_cats["StockCode"], n_cats["CustomerID"], best_dim), extra=te_feat)
comb_net.eval()
print(f"best epoch {comb_epoch}, validation RMSE (log) {comb_val:.4f}")

# %%
with torch.no_grad():
    pred_comb = comb_net(*idx["test"], te_feat["test"]).numpy().astype(np.float64)
results["2(d) Target encoding + embedding NN"] = evaluate(pred_comb)
pd.DataFrame(results).T.round(4)

# %% [markdown]
# Where does each model do better? Test RMSE (log) split by how often the row's stock code appears in training:

# %%
stock_count = test["StockCode"].astype(object).map(train["StockCode"].value_counts()).fillna(0).to_numpy()
bins = pd.cut(stock_count, [-1, 10, 100, np.inf], labels=["<= 10 rows", "11-100 rows", "> 100 rows"])
preds = {"2(b) target encoding": gbm_te.predict(Z_te), "2(c) embeddings": pred_emb,
         "2(d) combined": pred_comb}
by_bin = pd.DataFrame({name: pd.Series((p - y_te) ** 2).groupby(bins, observed=True).mean() ** 0.5
                       for name, p in preds.items()})
by_bin["test rows"] = pd.Series(bins).value_counts()
by_bin.round(4)
