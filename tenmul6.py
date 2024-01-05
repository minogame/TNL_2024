import jax
import opt_einsum
import numpy as np
import jax.numpy as jnp


class TNHelper:

    ## The helper class for TensorNetwork,
    ## contains utilizations.

    @staticmethod
    def is_triu_matrix(mat):
        try:
            return np.allclose(mat, np.triu(mat))
        except:
            dim = mat.shape[0]
            return np.all(mat[np.tril_indices(dim, -1)] == '0')
                

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
    def adjm_to_expr(adjm, only_cores=True, batch_first=True):
        adjm = TNHelper.to_triu(adjm)
        adjm_str = adjm.astype(int).astype(str)
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
        
        if only_cores:
            for a in adjm_str:
                einsum_str.append(''.join([ x for x in a if x.__ne__('0')]))
        else:
            if batch_first:
                src = [(j, 0) for j in range(1, adjm_str.shape[0])]
                dst = [(j, j) for j in range(1, adjm_str.shape[0])]
            else:
                src = [(j, -1) for j in range(0, adjm_str.shape[0]-1)]
                dst = [(j, j) for j in range(0, adjm_str.shape[0]-1)]
            
            adjm_str[[(s[0]) for s in src+dst], [(s[1]) for s in src+dst]] = adjm_str[[(s[0]) for s in dst+src], [(s[1]) for s in dst+src]]
            
            for a in adjm_str:
                einsum_str.append(''.join([ x for x in a if x.__ne__('0')]))
            
        adjm_diag = ''.join([ x for x in adjm_diag if x.__ne__('0')])
        einsum_str = f'{",".join(einsum_str)}->{adjm_diag}'

        return einsum_str

    @staticmethod
    def expand_adj_matrix(adj_matrix, target, batch_first=True):
        
        batch_size = target.shape[0] if batch_first else target.shape[-1]
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
    
    @staticmethod
    def expr_shape_feeder(adj_matrix, only_cores=True, batch_first=True):
        if only_cores:
            return [ s[s!=0] for s in np.vsplit(adj_matrix, adj_matrix.shape[0]) ]
        else:
            adjm = np.copy(adj_matrix)
            if batch_first:
                src = [(j, 0) for j in range(1, adj_matrix.shape[0])]
                dst = [(j, j) for j in range(1, adj_matrix.shape[0])]
            else:
                src = [(j, -1) for j in range(0, adj_matrix.shape[0]-1)]
                dst = [(j, j) for j in range(0, adj_matrix.shape[0]-1)]
            
            adjm[[(s[0]) for s in src+dst], [(s[1]) for s in src+dst]] = adjm[[(s[0]) for s in dst+src], [(s[1]) for s in dst+src]]
            return [ s[s!=0] for s in np.vsplit(adjm, adjm.shape[0]) ]
            
class NeuroCodingTensorNetwork:
    
    ## In NeuroCodingTensorNetwork, we do the following steps to initlize it.
    ## 1. We need a original shape adj_matrix, with its diagonal elements equal feature dimension.
    ## 2. Initilize weights and bias based on this shape adj_matrix.
    ## 3. Create cores with jax operators.
    ## 4. Create target shape adj_matrix, with its diagonal elements equal batch size.
    ## 5. Create einsum expression and other things.

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
        self.trainable_list = [int(t) for t in self.trainable_list]

        if initializer is None:
            initializer = np.random.rand

        self.init_cores(adj_matrix, initializer)

        ## for the retraction recording
        self.einsum_expr = None
        self.einsum_target_expr = None
        self.target_shape = None
        self.jit_retraction = None
        self.jit_target_retraction = None
        self.jit_target_retraction_gradient = None
        self.jit_target_retraction_value_gradient = None

    def giff_cores(self, idx=None):
        if idx is None or idx >= self.dim:
            return self.cores
        else:
            return self.cores[idx]
        
    def retraction(self, optimize='dp'):
        if not self.einsum_expr:
            einsum_str = TNHelper.adjm_to_expr(self.adj_matrix)
            adjm = TNHelper.to_full(self.adj_matrix)
            shapes = TNHelper.expr_shape_feeder(adjm)

            # 'dp' 'auto-hq'
            self.einsum_expr = opt_einsum.contract_expression(einsum_str, *shapes, optimize=optimize)
            self.jit_retraction = jax.jit(self.einsum_expr)

        retract_TN = self.jit_retraction(*self.cores)
        return retract_TN

    def target_retraction(self, target, batch_first=True, optimize='dp', return_retr=True):

        ## initialization or target_shape (batch size) changed
        if not (self.einsum_target_expr and target.shape == self.target_shape):

            ## check if the adj_matrix's output fit the target (without batchsize.)
            adjm_diag = np.diag(self.adj_matrix)
            adjm_diag_nonzero = adjm_diag[adjm_diag!=0]
            if batch_first:
                if not adjm_diag_nonzero.tolist() == list(target.shape[1:]):
                    raise ValueError('Target shape not equal to diag of adjm.')
            else:
                if not adjm_diag_nonzero.tolist() == list(target.shape[:-1]):
                    raise ValueError('Target shape not equal to diag of adjm.')

            self.target_shape = target.shape
            expanded_adj_matrix = TNHelper.expand_adj_matrix(self.adj_matrix, target, batch_first)
            einsum_str = TNHelper.adjm_to_expr(expanded_adj_matrix, False, batch_first)
            adjm = TNHelper.to_full(expanded_adj_matrix)
            shapes = TNHelper.expr_shape_feeder(adjm, False, batch_first)
            self.einsum_target_expr = opt_einsum.contract_expression(einsum_str, *shapes, optimize=optimize)
            self.jit_target_retraction = jax.jit(self.einsum_target_expr)

            if not return_retr:
                return True

        if return_retr:
            if batch_first:
                retract_target_TN = self.jit_target_retraction(target, *self.cores)
            else:
                retract_target_TN = self.jit_target_retraction(*self.cores, target)

            return retract_target_TN
        else:
            return False

    def target_retraction_grads(self, target, label, batch_first=True, optimize='dp', verbose=False):
        ## call target_retraction for initialization
        ## True if initialization is conducted, so we need to initilize jax.grad here
        if self.target_retraction(target, batch_first, optimize, False):

            ## change the loss function below
            def expr_with_mse_loss(cores, target, label):
                if batch_first:
                    retract_target_TN = self.einsum_target_expr(target, *cores)
                else:
                    retract_target_TN = self.einsum_target_expr(*cores, target)
                    
                mse_loss = jnp.mean(jnp.square(retract_target_TN - label), axis=0)

                return mse_loss
        
            self.jit_target_retraction_gradient = jax.jit(jax.grad(expr_with_mse_loss))
            self.jit_target_retraction_value_gradient = jax.jit(jax.value_and_grad(expr_with_mse_loss))
            

        ## This calculate the gradient
        if verbose:
            loss, grad_cores = self.jit_target_retraction_value_gradient(self.cores, target, label)
        else:
            grad_cores = self.jit_target_retraction_gradient(self.cores, target, label)
            loss = None

        return loss, grad_cores

    def gradient_descent(self, learning_rate, grad_cores):

        ## change this function for different update rules
        for idx, (t, g) in enumerate(zip(self.trainable_list, grad_cores)):
            if t:
                self.cores[idx] -= learning_rate * g
        
        return

    def iteration(self, learning_rate, target, label, batch_first=True, optimize='dp', verbose=False):
        loss, grad_cores = self.target_retraction_grads(target, label, batch_first, optimize, verbose)
        self.gradient_descent(learning_rate, grad_cores)

        if verbose:
            return loss
        else:
            return None
        

if __name__ == '__main__':
    adjm = np.array([
        [500,   2,   2,   2,   2,],
        [  2,   0,   3,   4,   0,],
        [  2,   3,   0,   3,   4,],
        [  2,   4,   3,   0,   3,],
        [  2,   0,   4,   3,   0,]])
    
    x = TNHelper.adjm_to_expr(adjm)
    print(x)