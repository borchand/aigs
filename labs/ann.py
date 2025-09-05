# %% Imports
from sklearn.datasets import fetch_openml
import numpy as np

# %%
mnist = fetch_openml("mnist_784", version=1)
X = np.array(mnist.data.astype("float32"))  # shape (70000, 784)
y = np.array(np.eye(10)[mnist.target.astype("int64")])  # shape (70000,)


def relu(x):
    return np.maximum(0, x)


def optim(W_1, W_2, loss):
    raise NotImplementedError


def main(cfg):
    # init params
    W_1 = np.random.rand(X.shape[1], cfg.hidden)
    W_2 = np.random.rand(cfg.hidden, 10)

    # model output
    for epoch in range(cfg.epochs):
        y_hat = relu(X @ W_1) @ W_2
        loss = loss_fn(y, y_hat)
        # W_1, W_2 = optim(W_1, W_2, loss)
        print(loss)


def loss_fn(y, y_hat):
    return np.abs(y - y_hat).mean()
