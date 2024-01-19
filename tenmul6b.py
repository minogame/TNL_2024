import jax
import opt_einsum
import numpy as np
import jax.numpy as jnp

    
class NCTNHelper:

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
        if NCTNHelper.is_triu_matrix(mat):
            dim = mat.shape[0]
            mat_return = np.copy(mat)
            mat_return[np.tril_indices(dim, -1)] = mat_return.transpose()[np.tril_indices(dim, -1)]
            return mat_return
        else:
            return mat
    
    @staticmethod
    def to_triu(mat):
        if NCTNHelper.is_triu_matrix(mat):
            return mat
        else:
            return np.triu(mat)

    @staticmethod
    def adjm_to_expr(adjm, additional_output):
        adjm = NCTNHelper.to_triu(adjm)
        adjm_str = adjm.astype(int).astype(str)
        np.fill_diagonal(adjm_str, '0')

        symbol_id = 1
        for i, j in np.ndindex(adjm_str.shape):
            if adjm_str[i,j] == '0':
                continue
            else:
                adjm_str[i,j] = opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

        adjm_str = NCTNHelper.to_full(adjm_str)

        ## in the the NCTN.span_cores, the last dimension is always "batch"
        ## so we append batch to all the shapes, and the retraction result is also batch
        einsum_str = []
        for a in adjm_str:
            einsum_str.append(''.join([ x for x in a if x.__ne__('0')]) + opt_einsum.get_symbol(0))

        adjm_diag = opt_einsum.get_symbol(0)
        einsum_str_z = []
        for e, k in zip(einsum_str, additional_output):
            if k:
                einsum_str_z.append(opt_einsum.get_symbol(symbol_id)+e)
                adjm_diag += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1
            else:
                einsum_str_z.append(e)
        einsum_str = f'{",".join(einsum_str_z)}->{adjm_diag}'
    
        return einsum_str

    @staticmethod
    def span_cores(weight, bias, target, activation=jnp.tanh):
        ## target is in shape (batch, feature, dimension)
        target_splits = jnp.split(target, indices_or_sections=target.shape[1], axis=1)
        return [ activation(jnp.einsum('...d,b...d->...b', w, t.squeeze()) + jnp.expand_dims(b, -1)) for w, b, t in zip(weight, bias, target_splits)]
    
    @staticmethod
    def expr_shape_feeder(adj_m, additional_output, target):
        adjm = NCTNHelper.to_full(adj_m)
        np.fill_diagonal(adjm, 0)

        shape = [ s[s!=0] for s in np.vsplit(adjm, adjm.shape[0]) ]
        shape = [np.append(s, target.shape[0]) for s in shape]
        shape_z = []
        for s, k in zip(shape, additional_output):
            if k:
                shape_z.append(np.concatenate([np.array([k]), s]))
            else:
                shape_z.append(s)
        
        return shape_z
            
class NeuroCodingTensorNetwork:
    
    ## Tenmul6b now has a free leg in a (randomly selected) core
    
    ## In NeuroCodingTensorNetwork, we do the following steps to initlize it.
    ## 1. We need a original shape adj_matrix, with its diagonal elements equal feature dimension.
    ## 2. Initilize weights and bias based on this shape adj_matrix.
    ## 3. Create cores with jax operators.
    ## 4. Create target shape adj_matrix, with its diagonal elements equal batch size.
    ## 5. Create einsum expression and other things.

    def init_cores_weights(self, adj_matrix, additional_output, initializer) -> None:
        adj_matrix = NCTNHelper.to_full(adj_matrix)
        self.original_adj_matrix = adj_matrix
        self.additional_output = additional_output
        adjm = np.copy(adj_matrix)
        adjm_diag = np.copy(np.diag(adjm))
        np.fill_diagonal(adjm, 0)
        W_shapes, B_shapes = [], []
        for s, k in zip(np.vsplit(np.c_[adjm, adjm_diag], adjm.shape[0]), additional_output):
            if k:
                W_shapes.append(np.concatenate([np.array([k]), s[s!=0]]))
            else:
                W_shapes.append(s[s!=0])
            
        for s, k in zip(np.vsplit(adjm, adjm.shape[0]), additional_output):
            if k:
                B_shapes.append(np.concatenate([np.array([k]), s[s!=0]]))
            else:
                B_shapes.append(s[s!=0])

        self.W = [initializer(*s) for s in W_shapes]
        self.B = [initializer(*s) for s in B_shapes]

    def __init__(self, adj_matrix, additional_output=None, initializer=None, trainable_list=None, activation=jnp.tanh):
        self.shape = adj_matrix.shape
        assert self.shape[0] == self.shape[1], 'adj_matrix must be a square matrix.'
        self.dim = self.shape[0]
        self.activation = activation

        if trainable_list is None:
            self.trainable_list = [True] * self.dim
        self.trainable_list = [int(t) for t in self.trainable_list]

        if initializer is None:
            initializer = np.random.rand

        self.init_cores_weights(adj_matrix, additional_output, initializer)

        ## for the retraction recording
        ## the retraction einsum_str is always the same one desipte input batch size
        self.einsum_str = NCTNHelper.adjm_to_expr(self.original_adj_matrix, self.additional_output)
        
        self.einsum_target_expr = None
        self.target_shape = None
        self.jit_target_retraction = None
        self.jit_target_retraction_gradient = None
        self.jit_target_retraction_value_gradient = None

    def giff_cores(self, idx=None):
        if idx is None or idx >= self.dim:
            return self.W, self.B
        else:
            return self.W[idx], self.B[idx]

    def target_retraction(self, target, optimize='dp', return_retr=True):
        
        ## initialization or target_shape (batch size) changed
        if not (self.einsum_target_expr and target.shape == self.target_shape):
            self.target_shape = target.shape
            
            shapes = NCTNHelper.expr_shape_feeder(self.original_adj_matrix, self.additional_output, target)
            self.einsum_target_expr = opt_einsum.contract_expression(self.einsum_str, *shapes, optimize=optimize)
            self.jit_target_retraction = jax.jit(self.einsum_target_expr)

            if not return_retr:
                return True

        cores = NCTNHelper.span_cores(self.W, self.B, target, self.activation)
        if return_retr:
            retract_target_TN = self.jit_target_retraction(*cores)
            return retract_target_TN
        else:
            return False

    def target_retraction_grads(self, target, label, optimize='dp', verbose=False):
        ## call target_retraction for initialization
        ## True if initialization is conducted, so we need to initilize jax.grad here
        if self.target_retraction(target, optimize, False):

            ## change the loss function below
            def expr_with_mse_loss(W, B, target, label):
                cores = NCTNHelper.span_cores(W, B, target, self.activation)
                retract_target_TN = self.einsum_target_expr(*cores)
                mse_loss = jnp.sum(jnp.square(retract_target_TN - label), axis=0)
                mse_loss = jnp.mean(mse_loss)

                return mse_loss

            self.jit_target_retraction_gradient = jax.jit(jax.grad(expr_with_mse_loss, argnums=[0, 1]))
            self.jit_target_retraction_value_gradient = jax.jit(jax.value_and_grad(expr_with_mse_loss, argnums=[0, 1]))

        ## This calculate the gradient
        if verbose:
            loss, grad_cores = self.jit_target_retraction_value_gradient(self.W, self.B, target, label)
        else:
            grad_cores = self.jit_target_retraction_gradient(self.W, self.B, target, label)
            loss = None

        return loss, grad_cores

    def gradient_descent(self, learning_rate, grad_cores):
        ## change this function for different update rules
        grad_W, grad_B = grad_cores
        for idx, (t, gW, gB) in enumerate(zip(self.trainable_list, grad_W, grad_B)):
            if t:
                self.W[idx] -= learning_rate * gW
                self.B[idx] -= learning_rate * gB
        
        return

    def iteration(self, learning_rate, target, label, optimize='dp', verbose=False):
        loss, grad_cores = self.target_retraction_grads(target, label, optimize, verbose)
        self.gradient_descent(learning_rate, grad_cores)

        if verbose:
            return loss
        else:
            return None
