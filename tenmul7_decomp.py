import numpy as np
import numpy as np
np.random.seed(420)
import jax.numpy as jnp
import functools, itertools

from tenmul7 import NeuroTN

def generate_TT_adj_matrix(order, rank, dim_mode):
    adjm = np.diag(np.full((order-1,),rank), 1)
    adjm = adjm = adjm.transpose()
    np.fill_diagonal(adjm, dim_mode)

    return adjm

def generate_TR_adj_matrix(order, rank, dim_mode):
    adjm = np.diag(np.full((order-1,), rank), 1) if np.isscalar(rank) else np.diag(rank[:-1], 1)
    adjm[0, order-1] = rank if np.isscalar(rank) else rank[-1]
    adjm = adjm + adjm.transpose()
    np.fill_diagonal(adjm, dim_mode)

    return adjm

def index_to_onehot(indices, num_class):
    idx = np.asarray(indices) if isinstance(indices, list) else indices
    
    if idx.ndim != 2:
        raise ValueError("indices must be a 2D list or array")
    
    N, M = idx.shape

    one_hot_encoded = np.zeros((N, M, num_class), dtype=float)

    one_hot_encoded[np.arange(N)[:, None], np.arange(M), idx] = 1

    return one_hot_encoded

# Parameters
order_tensor = 5
rank_tensor = 2
dim_tensor = 4
percentage_of_obsveration = 0.7

# Use NeuroTN to generate tensor
adjm = generate_TR_adj_matrix(order_tensor,rank_tensor,dim_tensor)


# Data generation 
output_dim = [0] * (order_tensor-1)+[1]
init_TN = functools.partial(np.random.normal, loc=0.0, scale=1)
DATA =  NeuroTN(adjm, output_dim, activation=lambda x:x, initializer = init_TN, core_mode=2)

idx_data = [list(combo) for combo in itertools.product(range(dim_tensor), repeat=order_tensor)]
idx_onehot = index_to_onehot(idx_data, num_class=dim_tensor)
values = DATA.network_contraction(idx_onehot, return_contraction=True)

permuted_idx = np.random.permutation(idx_onehot.shape[0])
length_training = int(len(permuted_idx)*percentage_of_obsveration)

data_training = idx_onehot[permuted_idx[:length_training]]
values_training = values[permuted_idx[:length_training]]
data_test = idx_onehot[permuted_idx[length_training:]]
values_test = values[permuted_idx[length_training:]]

print('data_test.shape', data_test.shape)
print('values_test.shape', values_test.shape)

# # Tensor netowrk decomposition 
# adjm_decomp = generate_TR_adj_matrix(order_tensor,rank_tensor,dim_tensor)
adjm_decomp = adjm
output_decomp = [0] * (order_tensor-1)+[1]

print('========================')

TN = NeuroTN(adjm_decomp, output_decomp, activation=lambda x:x, initializer = init_TN, core_mode=2)



size_data = idx_onehot.shape[0]
batch_size = size_data
learning_rate = 1e-3

for epoch in range(100000):
    loss_training = TN.iteration(learning_rate, data_training, values_training, verbose=True)
    if epoch % 100 == 0:
        predicted = TN.network_contraction(data_test, return_contraction=True)
        loss_test = np.mean(np.sum(np.square(predicted - values_test).reshape(predicted.shape[0],-1), axis=-1))
        print('Epoch: ', epoch, 'Training Loss: ', loss_training, '; Testing Loss: ', loss_test)

