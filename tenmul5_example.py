import numpy as np
np.random.seed(42)

from tenmul5 import TensorNetwork

## unit test of random dataset
dataset_x_batch_first = np.random.randn(500, 2, 3, 5, 7)
dataset_x_batch_last = dataset_x_batch_first.transpose(1, 2, 3, 4, 0)
dataset_y = np.random.randn(500)


## initialize TN with a given adjcency matrix 
adj_m = np.array(
    [
        [2, 1, 4, 0],
        [0, 3, 6, 8],
        [0, 0, 5, 9],
        [0, 0, 0, 7],
    ])

TN = TensorNetwork(adj_m)

## test all class method by fitting dataset
print('len(TN.giff_cores())', len(TN.giff_cores()))
print('TN.giff_cores(3)', TN.giff_cores(3)[0,0])
print('TN.retraction()', TN.retraction())

## batch first, train the TN with batch size 64
print('TN.target_retraction', TN.target_retraction(dataset_x_batch_first, return_retr=True))
for i in range(0, 500, 64):
    loss, grad = TN.target_retraction_grads(dataset_x_batch_first[i:i+64], dataset_y[i:i+64], batch_first=True, verbose=True)
    print(i, loss)

for epoch in range(10):
    for i in range(0, 500, 64):
        loss = TN.iteration(1e-10, dataset_x_batch_first[i:i+64], dataset_y[i:i+64], batch_first=True, verbose=True)
        print(i, loss)
    
print('TN.giff_cores(3)', TN.giff_cores(3)[0,0])
print('TN.target_retraction', TN.target_retraction(dataset_x_batch_first, return_retr=True))

## batch last, train the TN with batch size 64
print('TN.target_retraction', TN.target_retraction(dataset_x_batch_last, batch_first=False, return_retr=True))
for i in range(0, 500, 64):
    loss, grad = TN.target_retraction_grads(dataset_x_batch_last[:,:,:,:,i:i+64], dataset_y[i:i+64], batch_first=False, verbose=True)
    print(i, loss)

for i in range(0, 500, 64):
    loss = TN.iteration(1e-9, dataset_x_batch_last[:,:,:,:,i:i+64], dataset_y[i:i+64], batch_first=False, verbose=True)
    print(i, loss)

print('TN.target_retraction', TN.target_retraction(dataset_x_batch_last, batch_first=False, return_retr=True))

print('TN.giff_cores(3)', TN.giff_cores(3)[0,0])