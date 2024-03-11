import jax
import opt_einsum
import numpy as np
import jax.numpy as jnp
import functools

class NeuroTNHelper:
    ## The helper class for TensorNetwork,
    ## contains utilizations.

    @staticmethod
    def is_triu_matrix(mat):
        """
        Check if the matrix is upper triangular.
        
        Parameters:
        - mat (np.ndarray): The matrix to check.
        
        Returns:
        - bool: True if `mat` is upper triangular, False otherwise.
        """
        try:
            return np.allclose(mat, np.triu(mat))
        except:
            dim = mat.shape[0]
            return np.all(mat[np.tril_indices(dim, -1)] == '0')
        
    @staticmethod
    def to_full(mat):
        """
        Convert an upper triangular matrix to its full representation.
        
        Parameters:
        - mat (np.ndarray): The matrix to convert.
        
        Returns:
        - np.ndarray: The full matrix.
        """
        if NeuroTNHelper.is_triu_matrix(mat):
            dim = mat.shape[0]
            mat_return = np.copy(mat)
            mat_return[np.tril_indices(dim, -1)] = mat_return.transpose()[np.tril_indices(dim, -1)]
            return mat_return
        else:
            return mat
        
    @staticmethod
    def to_triu(mat):
        """
        Extract the upper triangular part of a matrix.
        
        Parameters:
        - mat (np.ndarray): The matrix to process.
        
        Returns:
        - np.ndarray: The upper triangular part of `mat`.
        """
        if NeuroTNHelper.is_triu_matrix(mat):
            return mat
        else:
            return np.triu(mat)
        
    @staticmethod
    def adjm_to_expr(adjm, additional_output, external_list):
        """
        Converts an adjacency matrix to an Einstein summation expression using symbols.

        Parameters:
        - adjm (np.ndarray): The adjacency matrix to convert.
        - additional_output (list): A list indicating which rows need an additional output symbol.
        - external_list (list): A list indicating external indices to be appended to the einsum string.

        Returns:
        - str: The Einstein summation expression.

        Note:
        On the right hand side of the expr, the last index always coresponds to "batch",
              i.e., in the form of "additional output" + "batch"
        """
        if not isinstance(adjm, np.ndarray):
            raise ValueError("Input adjm must be a NumPy array.")
        
        # Convert the adjacency matrix to upper triangular 
        adjm = NeuroTNHelper.to_triu(adjm)
        adjm_str = adjm.astype(int).astype(str)
        np.fill_diagonal(adjm_str, '0') # Ensure diagonal elements are '0'

        # Assign symbols to non-zero elements
        symbol_id = 1
        for i, j in np.ndindex(adjm_str.shape):
            if adjm_str[i,j] != '0':
                adjm_str[i, j] = opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

        # Convert back to full matrix to account for both upper and lower indices in einsum
        adjm_str = NeuroTNHelper.to_full(adjm_str)

        # Construct einsum strings based on external_list
        einsum_str = []
        for a, is_external in zip(adjm_str, external_list):
            current_str = ''.join([x for x in a if x != '0'])
            if is_external:
                current_str += opt_einsum.get_symbol(0)
            einsum_str.append(current_str)
            
        # Append additional output symbols if required
        str_right_hand_side = opt_einsum.get_symbol(0)  # Initialize with a common symbol corresponding to batch
        einsum_str_with_output = []
        for e, needs_additional in zip(einsum_str, additional_output):
            if needs_additional:
                output_symbol = opt_einsum.get_symbol(symbol_id)
                einsum_str_with_output.append(output_symbol + e)
                str_right_hand_side += output_symbol
                symbol_id += 1
            else:
                einsum_str_with_output.append(e)

        # Final Einstein summation expression
        einsum_str = f'{",".join(einsum_str_with_output)}->{str_right_hand_side}'
        print('einsum_str:', einsum_str)

        return einsum_str
    
    @staticmethod
    def core_formulation(weight, bias, target, external_list=None, mode=3, activation=jnp.tanh):
        """
        Generate tensor cores with multiple forms.

        Parameters:
        - weight (JAX Array): Tensor representing weights.
        - bias (JAX Array): Tensor representing biases.
        - target (Array): Input data.
        - external_list (list, optional): Specifies which cores are external. Defaults to all externals if None.
        - mode (int): Operation mode. Supports 1 (MLP layer), 2 (linear layer), or 3 (ResMLP).
        - activation (function): Activation function to apply for mode 1,3. Defaults to jnp.tanh.

        Returns:
        - list: Tensor cores.
        """
        target_splits = jnp.split(target, indices_or_sections=target.shape[1], axis=1)
        external_list = [1] * weight.shape[0] if external_list is None else external_list
        cores = []

        if mode == 1:
            counter_cores = 0
            for idx, is_external in enumerate(external_list):
                if is_external: # external cores
                    cores.append(activation(jnp.einsum('...d,b...d->...b', weight[idx], target_splits[counter_cores].squeeze()) \
                                            + jnp.expand_dims(bias[idx], -1)))
                    counter_cores += 1
                else: # internal cores
                    cores.append(weight[idx])
        elif mode == 2:
            counter_cores = 0
            for idx, is_external in enumerate(external_list): # external cores
                if is_external:
                    cores.append(jnp.einsum('...d,b...d->...b', weight[idx], target_splits[counter_cores].squeeze()))
                    counter_cores += 1
                else: # internal cores
                    cores.append(weight[idx])
        elif mode == 3:
            counter_cores = 0
            for idx, is_external in enumerate(external_list):
                if is_external: # external cores
                    if weight[idx].ndim <3:
                        cores.append(activation(jnp.einsum('...d,b...d->...b', weight[idx], target_splits[counter_cores].squeeze()) \
                                            + jnp.expand_dims(bias[idx], -1)))
                    else:
                        w_short_path_const = np.zeros(weight[idx].shape[:-1])
                        np.fill_diagonal(w_short_path_const, 1)
                        w_short_path_const = np.reshape(w_short_path_const,list(w_short_path_const.shape)+[1])
                        cores.append(activation(jnp.einsum('...d,b...d->...b', weight[idx], target_splits[counter_cores].squeeze()) \
                                            + jnp.expand_dims(bias[idx], -1)) + w_short_path_const)
                    counter_cores += 1
                else: # internal cores
                    if weight[idx].ndim < 2:
                        cores.append(weight[idx])
                    else:
                        w_short_path_const = np.zeros(weight[idx].shape)
                        np.fill_diagonal(w_short_path_const, 1)
                        cores.append(weight[idx] + w_short_path_const)
        else:
            raise ValueError("Unsupported mode specified.")
        
        return cores
        
    @staticmethod
    def expr_shape_feeder(adj_m, additional_output, target):
        """
        Determines the shapes of tensor cores based on an adjacency matrix.

        Parameters:
        - adj_m (np.ndarray): The adjacency matrix to process.
        - additional_output (list or np.ndarray): Indicates additional dimensions to be added based on the operation.
        - target (np.ndarray): data, providing one of the dimensions for the output shapes.

        Returns:
        - list of np.ndarray: A list of shapes for each core
        """
        adjm_full = NeuroTNHelper.to_full(adj_m).copy()
        external_list = np.diagonal(adjm_full).copy() # the external_list is obtain from the diagonal of adjm.
        np.fill_diagonal(adjm_full, 0)

        # shape extraction without batch_size and output_size
        shape = [ s[s!=0] for s in np.vsplit(adjm_full, adjm_full.shape[0]) ]

        # add batch size
        shapes_ext_adjusted = []
        for s, is_external in zip(shape, external_list):
            if is_external:
                shapes_ext_adjusted.append(np.append(s, target.shape[0]))
            else:
                shapes_ext_adjusted.append(s)

        # add output size
        shapes_final = []
        for s, dim_output in zip(shapes_ext_adjusted, additional_output):
            if dim_output:
                shapes_final.append(np.concatenate([np.array([dim_output]), s]))
            else:
                shapes_final.append(s)
        
        # print('shapes_final:', shapes_final)
        return shapes_final
    
class NeuroTN:

    def init_cores_weights(self, adj_matrix, additional_output, initializer) -> None:
        """
        Initializes the weights and biases for generating cores based on an adjacency matrix and additional outputs.

        Parameters:
        - adj_matrix (np.ndarray): The adjacency matrix representing connections in the tensor network.
        - additional_output (list or np.ndarray): Indicates nodes with additional output requirements.
        - initializer (callable): Function to initialize weights and biases, must accept a `size` parameter.
        """
        adj_matrix = NeuroTNHelper.to_full(adj_matrix)
        self.original_adj_matrix = adj_matrix.copy()
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

        self.W = [initializer(size=s) for s in W_shapes]

        self.B = []
        if self.core_mode == 2: # linear layer without bias
            self.B = None
        else:
            for s, is_external in zip(B_shapes, adjm_diag):
                if is_external == 0: # internal core
                    self.B.append(None)
                else:
                    self.B.append(initializer(size=s))

    def __init__(self, adj_matrix, additional_output=None, initializer=None, trainable_list=None, core_mode=1, activation=jnp.tanh):
        self.shape = adj_matrix.shape
        if self.shape[0] != self.shape[1]:
            raise ValueError('adj_matrix must be a square matrix.')
        self.dim = self.shape[0]
        self.core_mode = core_mode
        self.activation = activation
        self.external_list = np.diag(adj_matrix).copy()

        # Handle default for trainable_list
        if trainable_list is None:
            self.trainable_list = [True] * self.dim
        self.trainable_list = [int(t) for t in self.trainable_list]

        # Safe handling for initializer with a default function
        if initializer is None:
            initializer = functools.partial(np.random.normal, loc=0.0, scale=0.1)

        # Initial core weights setup
        self.init_cores_weights(adj_matrix, additional_output, initializer)

        # Setup for tensor network contraction
        self.einsum_str = NeuroTNHelper.adjm_to_expr(self.original_adj_matrix, self.additional_output, self.external_list)    
        
        # Initialize placeholders for JIT-compiled functions and target shapes
        self.einsum_target_expr = None
        self.target_shape = None
        self.jit_target_contraction = None
        self.jit_target_contraction_gradient = None
        self.jit_target_contraction_value_gradient = None

    def retrieve_cores(self, idx=None):
        """
        Retrieves the weights and biases for the entire network or a specific core, based on the index.

        Parameters:
        - idx (int, optional): The index of the core weights and biases to retrieve. If None or out of bounds,
                               returns weights and biases for all cores.

        Returns:
        - tuple: A tuple containing weights and biases. If an index is specified and valid, returns the weights
                 and biases for that specific core. Otherwise, returns weights and biases for all cores.
        """
        if idx is None or idx >= self.dim:
            return self.W, self.B
        else:
            return self.W[idx], self.B[idx]
        
    def network_contraction(self, target, optimize='dp', return_contraction=True):
        """
        Performs the network contraction. Optionally compiles the contraction
        expression for improved performance using JAX JIT compilation.

        Parameters:
        - target (np.ndarray): Data for the contraction.
        - optimize (str): The optimization strategy for the tensor contraction (defaults to 'dp').
        - return_contraction (bool): If True, returns the contraction result; otherwise, returns a boolean
          indicating if the contraction setup was updated.

        Returns:
        - The result of the tensor network contraction if `return_contraction` is True, otherwise returns a
          boolean indicating whether the contraction setup was updated.
        """
        
        # Update the batch size and corresponding contract_expression if the batch size is changed.
        if not (self.einsum_target_expr and target.shape == self.target_shape):
            self.target_shape = target.shape
            
            shapes = NeuroTNHelper.expr_shape_feeder(self.original_adj_matrix, self.additional_output, target)
            # print('adjm_orignal', self.original_adj_matrix, 'bingo', target.shape, 'shapes:', shapes)
            self.einsum_target_expr = opt_einsum.contract_expression(self.einsum_str, *shapes, optimize=optimize)
            self.jit_target_contraction = jax.jit(self.einsum_target_expr)

            if not return_contraction:
                return True
            
        
        # Prepare cores for contraction
        cores = NeuroTNHelper.core_formulation(self.W, self.B, target, self.external_list, self.core_mode, self.activation)
        
        # Perform the contraction if requested
        
        if return_contraction:
            contracted_target_TN = self.jit_target_contraction(*cores)
            return contracted_target_TN
        else:
            return False
        
        
    def network_contraction_grads(self, target, label, optimize='dp', verbose=False):
        """
        Computes the gradients of the loss function with respect to the weights and biases of the network.

        Parameters:
        - target (np.ndarray): The input tensor for the network.
        - label (np.ndarray): The target labels for the loss calculation.
        - optimize (str, optional): Optimization strategy for tensor contraction. Defaults to 'dp'.
        - verbose (bool, optional): If True, also returns the loss value along with the gradients.

        Returns:
        - tuple: The loss (if verbose is True) and the gradients of the weights and biases.
        """
        if self.network_contraction(target, optimize, False):
            ## change the loss function below
            def expr_with_mse_loss(W, B, target, label):
                label_size = label.shape[0]
                cores = NeuroTNHelper.core_formulation(W, B, target, self.external_list, self.core_mode, self.activation)
                contracted_target_TN = self.einsum_target_expr(*cores)
                mse_loss = jnp.sum(jnp.square(contracted_target_TN - label).reshape(label_size, -1), axis=-1)
                mse_loss = jnp.mean(mse_loss)

                return mse_loss
            
            def expr_with_softmax_loss(W, B, target, label):
                label_size = label.shape[0]
                label = label.reshape(label_size, -1)
                cores = NeuroTNHelper.core_formulation(W, B, target, self.external_list, self.core_mode, self.activation)
                contracted_target_TN = self.einsum_target_expr(*cores)
                logits = jax.nn.softmax(contracted_target_TN).reshape(label_size, -1)
                log_softmax_ce_loss = -jnp.mean(jnp.sum(jnp.log(logits) * label, axis=-1))
                
                return log_softmax_ce_loss
            
            def expr_with_softmax_loss_withWD(W, B, target, label):
                label_size = label.shape[0]
                label = label.reshape(label_size, -1)
                cores = NeuroTNHelper.core_formulation(W, B, target, self.external_list, self.core_mode, self.activation)
                contracted_target_TN = self.einsum_target_expr(*cores)
                logits = jax.nn.softmax(contracted_target_TN).reshape(label_size, -1)
                log_softmax_ce_loss = -jnp.mean(jnp.sum(jnp.log(logits) * label, axis=-1))
                
                WD_loss = 0.0
                for w in W:
                    WD_loss += jnp.sum(jnp.square(w)) * 1e-6
                for b in B:
                    WD_loss += jnp.sum(jnp.square(b)) * 1e-6
                
                return log_softmax_ce_loss + WD_loss
            
            # the_loss = expr_with_softmax_loss
            the_loss = expr_with_mse_loss

            self.jit_target_contraction_gradient = jax.jit(jax.grad(the_loss, argnums=[0, 1]))
            self.jit_target_contraction_value_gradient = jax.jit(jax.value_and_grad(the_loss, argnums=[0, 1]))

        ## This calculate the gradient
        if verbose:
            loss, grad_cores = self.jit_target_contraction_value_gradient(self.W, self.B, target, label)
        else:
            grad_cores = self.jit_target_contraction_gradient(self.W, self.B, target, label)
            loss = None

        return loss, grad_cores
    
        # if self.network_contraction(target, optimize, False):
        #     # Choose the loss function based on your requirements
        #     the_loss = self.mse_loss  # or self.softmax_loss or self.softmax_loss_with_weight_decay

        #     self.jit_target_contraction_gradient = jax.jit(jax.grad(the_loss, argnums=[0, 1]))
        #     self.jit_target_contraction_value_gradient = jax.jit(jax.value_and_grad(the_loss, argnums=[0, 1]))

        # # Compute the gradients (and loss if verbose)
        # if verbose:
        #     loss, grad_cores = self.jit_target_contraction_value_gradient(self.W, self.B, target, label)
        # else:
        #     grad_cores = self.jit_target_contraction_gradient(self.W, self.B, target, label)
        #     loss = None

        # return loss, grad_cores
    
    def gradient_descent(self, learning_rate, grad_cores):
        """
        Updates the network's weights and biases using the gradient descent optimization method.

        Parameters:
        - learning_rate (float): The learning rate for the gradient descent update.
        - grad_cores (tuple): A tuple containing gradients of weights and biases (grad_W, grad_B).

        The method updates weights and biases based on their gradients and the learning rate, considering
        whether each component is marked as trainable.
        """

        grad_W, grad_B = grad_cores

        if self.core_mode == 2:
            for idx, (t, gW) in enumerate(zip(self.trainable_list, grad_W)):
                if t:
                    self.W[idx] -= learning_rate * gW
        else:
            for idx, (t, gW, gB) in enumerate(zip(self.trainable_list, grad_W, grad_B)):
                if t:
                    self.W[idx] -= learning_rate * gW
                    self.B[idx] -= learning_rate * gB
        
        return
    
    def iteration(self, learning_rate, target, label, optimize='dp', verbose=False):
        """
        Performs a single training iteration, including gradient computation and parameter update.

        Parameters:
        - learning_rate (float): The learning rate for the gradient descent update.
        - target (np.ndarray): The input tensor for the network.
        - label (np.ndarray): The target labels used for loss calculation.
        - optimize (str, optional): Optimization strategy for tensor contraction. Defaults to 'dp'.
        - verbose (bool, optional): If True, returns the computed loss for this iteration.

        Returns:
        - The loss for this iteration if verbose is True; otherwise, returns None.
        """
        loss, grad_cores = self.network_contraction_grads(target, label, optimize, verbose)
        self.gradient_descent(learning_rate, grad_cores)

        if verbose:
            return loss
        else:
            return None
    
    def ntk(self, target, opt_path='db'):
        dim_output = jnp.prod(self.additional_output[self.additional_output != 0])

        def network_model(W, B, target):
                cores = NeuroTNHelper.core_formulation(W, B, target, self.external_list, self.core_mode, self.activation)
                return self.einsum_target_expr(*cores)
        
        self.network_contraction(target, optimize=opt_path, return_contraction=False)
        
        jac_W, jac_B = jax.jit(jax.jacfwd(network_model, argnums=[0, 1]))(self.W, self.B, target)
        
        ntk_tensor = jnp.zeros((target.shape[0], target.shape[0]))
        
        @jax.jit
        def process_tensor(tensor):
            tensor_plain = tensor.reshape(tensor.shape[0], dim_output, -1).swapaxes(0, 1)
            return jnp.matmul(tensor_plain, tensor_plain.swapaxes(1, 2))
                
        for w in jac_W:
            ntk_tensor += process_tensor(w)
        
        if self.core_mode != 2:  # Mode 2 means no bias
            for b in jac_B:
                ntk_tensor += process_tensor(b)

        return ntk_tensor