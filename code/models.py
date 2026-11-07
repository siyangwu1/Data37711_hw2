"""
Skeleton code for creating the Random Kitchen Sinks (RKS) and 1-hidden-layer MLP (MLP) models.

Note you are *welcome* to completely ignore this file and use your own code, but it's here if you want to use it as a starting point.
"""
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn

"""
For Random Kitchen Sinks (RKS)
"""
class RandomKitchenSinks:
    """
    Random Kitchen Sinks featurization with ridge regression.
    """
    def __init__(self, D: int = 1000, sigma: float = 1.0, ridge_lambda: float = 1e-3, seed: int = 0):
        """
        Potential hyper-parameters you can consider (feel free to add more):
            D: total number of random features (= 2h in the assignment text; if you use both sin and cos features, D should be an even number)
            sigma: scale parameter for the RBF kernel, used for initializing the weight parameters
            ridge_lambda: ridge regularization strength
            seed: random seed for drawing the random projections
        """
        if D % 2 != 0:
            raise ValueError("D must be even (h sin features + h cos features)")
        self.D = D
        self.h = D // 2
        self.sigma = sigma
        self.ridge_lambda = ridge_lambda
        self.seed = seed
        self.W = None      # (d, h) random directions, w_j ~ N(0, I / sigma^2); drawn in fit
        self.b = None      # (h,) random phases, b_j ~ U[0, 2 pi]
        self.coef_ = None  # (D + 1,) ridge coefficients; the last one is the intercept

    def _features(self, X: np.ndarray) -> np.ndarray:
        """
        Convert the input data X into the random kitchen sinks features. You will likely need to use the parameters initialized in the __init__ method.
        The resulting features should be of shape (n, D or D[+1]), where the plus one is for the intercept term.
        """
        proj = X @ self.W + self.b                                    # (n, h): w_j^T x + b_j
        return np.hstack([np.sin(proj), np.cos(proj), np.ones((len(X), 1))])   # (n, D + 1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the ridge regression model to the (1) features computed from the input data X and (2) target values y.
        """
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0.0, 1.0 / self.sigma, size=(X.shape[1], self.h))
        self.b = rng.uniform(0.0, 2 * np.pi, size=self.h)
        phi = self._features(X)  # shape (n, D or D[+1])
        # Ridge solution beta = (Phi^T Phi + lambda I')^{-1} Phi^T y, where I' does not penalize the intercept
        penalty = self.ridge_lambda * np.eye(phi.shape[1])
        penalty[-1, -1] = 0.0
        self.coef_ = np.linalg.solve(phi.T @ phi + penalty, phi.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the input data X.
        """
        phi = self._features(X)  # shape (n, D or D[+1])
        return phi @ self.coef_  # shape (n,)

def sweep_rks_model(sweep_D: List[int], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple:
    """
    An example sweeping function that sweeps over the number of random features D, it then returns the best model in terms of validation MSE.
    Args:
        sweep_D: list of numbers of random features to sweep over
        X_train: training input data
        y_train: training target values
        X_val: validation input data
        y_val: validation target values
    Returns:
        a tuple (best_model, best_val_mse, best_D)
    """
    best_model = None
    best_val_mse = float('inf')
    best_D = None
    for D in sweep_D:
        model = RandomKitchenSinks(D=D).fit(X_train, y_train)
        val_mse = float(np.mean((model.predict(X_val) - y_val) ** 2))
        if val_mse < best_val_mse:
            best_model = model
            best_val_mse = val_mse
            best_D = D

    return best_model, best_val_mse, best_D

"""
For 1-hidden-layer MLP (MLP)
"""
class MLP(nn.Module):
    """1-hidden-layer MLP for regression."""
    def __init__(self, d_in: int, d_hidden: int = 128, d_out: int = 1, lr: float = 1e-3, batch_size: int = 128, activation: str = "relu", device: str = "cpu",
                 epochs: int = 1000, seed: int = 0):
        """
        Potential hyper-parameters you can consider (feel free to add more):
            d_hidden: hidden dimension
            activation: activation function
            lr: learning rate
            batch_size: batch size
            epochs: number of passes over the training data
            seed: random seed (weight initialization and mini-batch order)
        """
        super().__init__()
        # DO NOT forget this -- GPU will make NN training much faster if you have access to one
        self.device = device
        self.lr, self.batch_size, self.epochs, self.seed = lr, batch_size, epochs, seed
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            activations[activation](),
            nn.Linear(d_hidden, d_out),
        ).to(device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # turn the numpy data into torch tensors and move to the device
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().view(-1, 1).to(self.device)

        self.net.train()  # make sure to set the model to training mode
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(self.seed)
        n = len(X)
        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, self.batch_size):
                b = perm[i:i + self.batch_size]
                loss = loss_fn(self.net(X[b]), y[b])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.net.eval()
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            return self.forward(X).detach().cpu().numpy()

def mlp_sweep(sweep_values: List[float], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple:
    """
    See the example `sweep_rks_model` above for inspiration. Good luck!

    Sweeps the hidden dimension d_hidden (other hyper-parameters at their defaults).
    Returns (best_model, best_val_mse, best_value, records), where records is a list of
    dicts with the training and validation MSE for every value.
    """
    best_model, best_val_mse, best_value, records = None, float("inf"), None, []
    for value in sweep_values:
        model = MLP(d_in=X_train.shape[1], d_hidden=int(value)).fit(X_train, y_train)
        train_mse = float(np.mean((model.predict(X_train).ravel() - y_train) ** 2))
        val_mse = float(np.mean((model.predict(X_val).ravel() - y_val) ** 2))
        records.append({"d_hidden": int(value), "train MSE": train_mse, "val MSE": val_mse})
        if val_mse < best_val_mse:
            best_model, best_val_mse, best_value = model, val_mse, value
    return best_model, best_val_mse, best_value, records
