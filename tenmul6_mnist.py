import numpy as np
np.random.seed(42)
import jax.numpy as jnp

from tenmul6 import NeuroCodingTensorNetwork

def generate_stupid_large_tensor_ring_adj_matrix(order, rank):
    adjm = np.diag(np.full((order-1,), rank), 1)
    np.fill_diagonal(adjm, rank)
    adjm[0, -1] = rank
    
    return adjm

def reform_dataset():
    dataset = np.load('mnist.npz')
    x_train = dataset['x_train'].reshape(60000, -1).astype(float) / 510.0 * np.pi
    x_train = np.stack([np.cos(x_train), np.sin(x_train)], axis=-1)
    y_train = dataset['y_train']
    x_test = dataset['x_test'].reshape(10000, -1).astype(float) / 510.0 * np.pi
    y_test = dataset['y_test']
    
    return

reform_dataset()
exit()
adjm = generate_stupid_large_tensor_ring_adj_matrix(28*28,2)
TN = NeuroCodingTensorNetwork(adjm, activation=jnp.tanh)



