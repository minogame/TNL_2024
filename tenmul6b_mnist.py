import numpy as np
np.random.seed(420)
import jax.numpy as jnp
import cv2
import more_itertools, functools

from tenmul6b import NeuroCodingTensorNetwork

def generate_stupid_large_tensor_ring_adj_matrix(order, rank):
    adjm = np.diag(np.full((order-1,), rank), 1)
    np.fill_diagonal(adjm, 2)
    adjm[0, -1] = rank
    
    return adjm

def reform_dataset(center_crop=24, resize=None):
    unpad = (28 - center_crop) // 2
    dataset = np.load('mnist.npz')
    x_train = dataset['x_train'][:, unpad:28-unpad, unpad:28-unpad]
    if resize:
        x_train_r = []
        for x in np.split(x_train, 500, axis=0):
            x_train_r.append(cv2.resize(x.transpose(1, 2, 0), dsize=(resize, resize)).transpose(2, 0, 1))
        x_train = np.stack(x_train_r, axis=0).reshape(-1, resize, resize)
    x_train = x_train.reshape(60000, -1).astype(float) / 510.0 * np.pi
    x_train = np.stack([np.cos(x_train), np.sin(x_train)], axis=-1)

    y_train = dataset['y_train']
    y_train = np.eye(10)[y_train]
    x_test = dataset['x_test'][:, unpad:28-unpad, unpad:28-unpad]
    if resize:
        x_test_r = []
        for x in np.split(x_test, 500, axis=0):
            x_test_r.append(cv2.resize(x.transpose(1, 2, 0), dsize=(resize, resize)).transpose(2, 0, 1))
        x_test = np.stack(x_test_r, axis=0).reshape(-1, resize, resize)
    
    x_test = x_test.reshape(10000, -1).astype(float) / 510.0 * np.pi
    x_test = np.stack([np.cos(x_test), np.sin(x_test)], axis=-1)
    y_test = dataset['y_test']
    y_test = np.eye(10)[y_test]
    
    return x_train, x_test, y_train, y_test

resize = 8
x_train, x_test, y_train, y_test = reform_dataset(center_crop=24, resize=resize)

adjm = generate_stupid_large_tensor_ring_adj_matrix(resize**2,4)
adop = np.zeros((resize**2), dtype=int)
adop[-1] = 10
TN = NeuroCodingTensorNetwork(adjm, adop, activation=jnp.tanh, initializer = functools.partial(np.random.normal, loc=0.0, scale=np.sqrt(1.0/5)))

batch_size = 50
learning_rate = 3e-1

def train_one_epoch(e):
    np.random.default_rng().permuted(x_train, axis=0)
    for idx, (x, y) in enumerate(zip(more_itertools.batched(x_train, batch_size), more_itertools.batched(y_train, batch_size))):
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        if e > 2:
            lr = learning_rate / 10
        elif e > 5:
            lr = learning_rate / 100
        else:
            lr = learning_rate
        loss = TN.iteration(lr, x, y, optimize='random-greedy-128', verbose=True)
        if idx % 50 == 0:
            print(e, idx, loss)
    return

def evaluate():
    correct = 0
    for x, y in zip(more_itertools.batched(x_test, batch_size), more_itertools.batched(y_test, batch_size)):
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        logits = TN.target_retraction(x, return_retr=True)
        correct += np.sum(np.argmax(logits, -1) == np.argmax(y, -1))
    print(correct)
    return

# print('TN.giff_cores(35)', TN.giff_cores(35)[0], TN.giff_cores(35)[1])
for e in range(100):
    train_one_epoch(e+1)
    evaluate()