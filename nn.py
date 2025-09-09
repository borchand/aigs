from array import array
from sklearn.datasets import fetch_openml
import pickle
import jax.numpy as jnp
import numpy as np
from jax import grad

hidden = 30
epochs = 1000
lr = 0.001

def init(X):
    return {
        "W_1": jnp.array(np.random.randn(X.shape[1], hidden)) * 0.01,
        "W_2": jnp.array(np.random.randn(hidden, 10)) * 0.01
    }

def relu(x):
    return jnp.maximum(0, x)


def apply_fn(params, X):
    return relu(relu(X @ params["W_1"]) @ params["W_2"])

def loss_fn(params, X, y):
    y_hat = apply_fn(params, X)
    return ((y - y_hat) ** 2).mean() # MSE


def optim(W_1, W_2, loss):
    W_1 -= 0.00001 * loss
    W_2 -= 0.00001 * loss

    return W_1, W_2

def acc(y, y_hat):
    return (jnp.argmax(y, axis=1) == jnp.argmax(y_hat, axis=1)).mean()

# check if pickle file exsits
try:
    with open('mnist.pkl', 'rb') as f:
        X, y = pickle.load(f)
    print("Loaded MNIST from pickle file.")

except FileNotFoundError:
    print("Downloading MNIST from openml...")
    mnist = fetch_openml('mnist_784', version=1)

    X = jnp.array(mnist.data.astype('float32'))
    y = jnp.array(jnp.eye(10)[mnist.target.astype('int64')])

    # store X and y with prickle so we don't need to download again
    with open('mnist.pkl', 'wb') as f:
        pickle.dump((X, y), f)

# plot some of the data
# import matplotlib.pyplot as plt
# print(y[0])
# plt.imshow(X[0].reshape(28, 28), cmap='gray')
# plt.show()

print(X.shape, y.shape)

params = init(X)

grad_fn = grad(loss_fn)

for epoch in range(epochs):
    grads = grad_fn(params, X ,y)
    params = { "W_1": params["W_1"] - lr * grads["W_1"], "W_2": params["W_2"] - lr * grads["W_2"] }
    loss = loss_fn(params, X, y)
    a = acc(y, apply_fn(params, X))
    print(f"Epoch {epoch}: loss = {loss} acc = {a}")
