import numpy as np
np.random.seed(42)
import jax.numpy as jnp

from tenmul6 import NeuroCodingTensorNetwork

## unit test of random dataset
dataset_x = np.random.randn(500, 4, 2)
dataset_y = np.random.randn(500)


## initialize TN with a given adjcency matrix 
adj_m = np.array(
    [
        [2, 3, 5, 0],
        [0, 2, 7, 8],
        [0, 0, 2, 9],
        [0, 0, 0, 2],
    ])

TN = NeuroCodingTensorNetwork(adj_m, activation=jnp.tanh)

## test all class method by fitting dataset
print('len(TN.giff_cores()[0])', len(TN.giff_cores()[0]))
print('TN.giff_cores(3)', TN.giff_cores(3)[0].shape, TN.giff_cores(3)[1].shape)
print('TN.target_retraction', TN.target_retraction(dataset_x, return_retr=True).shape)

## train the TN with batch size 64
for i in range(0, 500, 64):
    loss, grad = TN.target_retraction_grads(dataset_x[i:i+64], dataset_y[i:i+64], verbose=True)
    print(i, loss)
    break

for epoch in range(10):
    for i in range(0, 500, 64):
        loss = TN.iteration(1e-7, dataset_x[i:i+64], dataset_y[i:i+64], verbose=True)
        print(i, loss)
