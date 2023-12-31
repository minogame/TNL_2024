import jax
import opt_einsum
import numpy as np
import jax.numpy as jnp


class TNHelper:

    ## The helper class for TensorNetwork,
    ## contains utilizations.

    @staticmethod
    def is_triu_matrix(mat):
        return np.allclose(mat, np.triu(mat))

    @staticmethod
    def to_full(mat):
        if TNHelper.is_triu_matrix(mat):
            dim = mat.shape[0]
            mat_return = np.copy(mat)
            mat_return[np.tril_indices(dim, -1)] = mat_return.transpose()[np.tril_indices(dim, -1)]
            return mat_return
        else:
            return mat
    
    @staticmethod
    def to_triu(mat):
        if TNHelper.is_triu_matrix(mat):
            return mat
        else:
            return np.triu(mat)

    @staticmethod
    def adjm_to_expr(adjm):
        adjm = TNHelper.to_triu(adjm)
        adjm_str = adjm.astype(str)
        adjm_diag = np.copy(np.diag(adjm_str))
        np.fill_diagonal(adjm_str, '0')
        
        symbol_id = 0
        for i in np.ndindex(adjm_diag.shape):
            if adjm_diag[i] == '0':
                continue
            else:
                adjm_diag[i] = opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

        for i, j in np.ndindex(adjm_str.shape):
            if adjm_str[i,j] == '0':
                continue
            else:
                adjm_str[i,j] = opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

        adjm_str = TNHelper.to_full(adjm_str)
        np.fill_diagonal(adjm_str, adjm_diag)

        einsum_str = []
        for a in adjm_str:
            einsum_str.append(''.join([ x for x in a if x.__ne__('0')]))
        adjm_diag = ''.join([ x for x in adjm_diag if x.__ne__('0')])
        einsum_str = f'{",".join(einsum_str)}->{adjm_diag}'

        return einsum_str

    @staticmethod
    def expand_adj_matrix(adj_matrix, batch_size, batch_first=True):

        adj_matrix = TNHelper.to_triu(adj_matrix)
        if batch_first:
            new_adj_matrix = np.c_[np.zeros((adj_matrix.shape[0]+1, 1)), np.r_[np.diag(adj_matrix)[np.newaxis, :], adj_matrix]]
            np.fill_diagonal(new_adj_matrix, 0)
            new_adj_matrix[0, 0] = batch_size
        else:
            new_adj_matrix = np.r_[np.c_[adj_matrix, np.diag(adj_matrix)], np.zeros((1, adj_matrix.shape[0]+1))]
            np.fill_diagonal(new_adj_matrix, 0)
            new_adj_matrix[-1, -1] = batch_size

        return TNHelper.to_full(new_adj_matrix)

class TensorNetwork:

    def init_cores(self, adj_matrix, initializer) -> None:
        adj_matrix = TNHelper.to_full(adj_matrix)
        self.adj_matrix = adj_matrix
        core_shapes = [ s[s!=0] for s in np.vsplit(adj_matrix, self.dim) ]
        self.cores = [initializer(*s) for s in core_shapes]

    def __init__(self, adj_matrix, initializer=None, trainable_list=None):
        self.shape = adj_matrix.shape
        assert self.shape[0] == self.shape[1], 'adj_matrix must be a square matrix.'
        self.dim = self.shape[0]

        if trainable_list is None:
            self.trainable_list = [True] * self.dim

        if initializer is None:
            initializer = np.random.rand

        self.init_cores(adj_matrix, initializer)

        ## for the retraction recording
        self.einsum_expr = None
        self.einsum_target_expr = None
        self.target_shape = None

    def giff_cores(self, idx=None):
        if idx is None or idx >= self.dim:
            return self.cores
        else:
            return self.cores[idx]
        
    def retraction(self, optimize='dp'):
        if not self.einsum_expr:
            einsum_str = TNHelper.adjm_to_expr(self.adj_matrix)
            adjm = TNHelper.to_full(self.adj_matrix)
            shapes = [ s[s!=0] for s in np.vsplit(adjm, self.dim) ]

            # 'dp' 'auto-hq'
            self.einsum_expr = opt_einsum.contract_expression(einsum_str, *shapes, optimize=optimize)

        jit_foo = jax.jit(self.einsum_expr)
        retract_TN = jit_foo(*self.cores)
        return retract_TN

    def fit_target(self, target, batch_first=True, optimize='dp'):

        if not (self.einsum_target_expr and target.shape == self.target_shape):
            ## initialization or target_shape (batch size) changed
            self.target_shape = target.shape
            self.einsum_target_expr = opt_einsum.contract_expression(self.einsum_str, *shapes, optimize=optimize)


            adjm_diag = np.diag(self.adj_matrix)
            adjm_diag_nonzero = adjm_diag[adjm_diag!=0]
            if batch_first:
                if adjm_diag_nonzero


        jit_foo = jax.jit(self.einsum_target_expr)
        retract_TN = jit_foo(*self.cores)

        else:
        
        return retract_TN

        if not np.allclose(adjm_diag_nonzero, target):
            raise ValueError

        pass

    def giff_grads(self, ):
        def expr_mae(x):
            x = expr(*x)
            return jnp.sum(x)
        
        jit_dfoo = jax.jit(jax.grad(expr_mae))
        z = jit_dfoo(cores)

        print(len(z))
        print(z[0].shape)


    def gradient_descent(self, learning_rate):
        self.a
        pass

    def opt_opeartions(self, opt, loss):
        return opt.minimize(loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES))
