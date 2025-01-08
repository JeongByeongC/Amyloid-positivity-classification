from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.utils import gcn_filter
from scipy import sparse as sp
from scipy import linalg
import numpy as np

def degree_matrix(A):
    degrees = np.array(A.sum(1)).flatten()
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D

def laplacian(A):
    return degree_matrix(A) - A


def modal_dot(a, b, transpose_a=False, transpose_b=False):
    """
    Computes the matrix multiplication of a and b, handling the data modes
    automatically.
    This is a wrapper to standard matmul operations, for a and b with rank 2
    or 3, that:
    - Supports automatic broadcasting of the "batch" dimension if the two inputs
    have different ranks.
    - Supports any combination of dense and sparse inputs.
    This op is useful for multiplying matrices that represent batches of graphs
    in the different modes, for which the adjacency matrices may or may not be
    sparse and have different ranks from the node attributes.
    Additionally, it can also support the case where we have many adjacency
    matrices and only one graph signal (which is uncommon, but may still happen).
    If you know a-priori the type and shape of the inputs, it may be faster to
    use the built-in functions of TensorFlow directly instead.
    Examples:
        - `a` rank 2, `b` rank 2 -> `a @ b`
        - `a` rank 3, `b` rank 3 -> `[a[i] @ b[i] for i in range(len(a))]`
        - `a` rank 2, `b` rank 3 -> `[a @ b[i] for i in range(len(b))]`
        - `a` rank 3, `b` rank 2 -> `[a[i] @ b for i in range(len(a))]`
    :param a: Tensor or SparseTensor with rank 2 or 3;
    :param b: Tensor or SparseTensor with rank 2 or 3;
    :param transpose_a: transpose the innermost 2 dimensions of `a`;
    :param transpose_b: transpose the innermost 2 dimensions of `b`;
    :return: Tensor or SparseTensor with rank = max(rank(a), rank(b)).
    """
    a_ndim = K.ndim(a)
    b_ndim = K.ndim(b)
    assert a_ndim in (2, 3), "Expected a of rank 2 or 3, got {}".format(a_ndim)
    assert b_ndim in (2, 3), "Expected b of rank 2 or 3, got {}".format(b_ndim)

    if transpose_a:
        perm = None if a_ndim == 2 else (0, 2, 1)
        a = ops.transpose(a, perm)
    if transpose_b:
        perm = None if b_ndim == 2 else (0, 2, 1)
        b = ops.transpose(b, perm)

    if a_ndim == b_ndim:
        # ...ij,...jk->...ik
        # print('1')
        return ops.dot(a, b)
    elif a_ndim == 2:
        # ij,bjk->bik
        # print('2')
        return ops.mixed_mode_dot(a, b)
    else:  # a_ndim == 3
        # bij,jk->bik
        # print('3')
        if not K.is_sparse(a) and not K.is_sparse(b):
            # print('4')
            # Immediately fallback to standard dense matmul, no need to reshape
            return tf.matmul(a, b)
    

        # If either input is sparse, we use dot(a, b)
        # This implementation is faster than using rank 3 sparse matmul with tfsp
        a_shape = tf.shape(a)
        b_shape = tf.shape(b)
        a_flat = ops.reshape(a, (-1, a_shape[2]))
        output = ops.dot(a_flat, b)
        return ops.reshape(output, (-1, a_shape[1], b_shape[1]))

class GraphConvolution(Conv):
    r"""
    A graph convolutional layer (GCN) from the paper
    > [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)<br>
    > Thomas N. Kipf and Max Welling
    **Mode**: single, disjoint, mixed, batch.
    This layer computes:
    $$
        \X' = \hat \D^{-1/2} \hat \A \hat \D^{-1/2} \X \W + \b
    $$
    where \( \hat \A = \A + \I \) is the adjacency matrix with added self-loops
    and \(\hat\D\) is its degree matrix.
    **Input**
    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Modified Laplacian of shape `([batch], n_nodes, n_nodes)`; can be computed with
    `spektral.utils.convolution.gcn_filter`.
    **Output**
    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.
    **Arguments**
    - `channels`: number of output channels;
    - `activation`: activation function;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(
        self,
        channels,
        # batch_norm=True,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):

        super().__init__(
            activation=activation,
            # use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        # self.use_batch_norm = batch_norm
        self.use_bias = use_bias
        # self.activation = activation
        # self.kernel_initializer = kernel_initializer
        # self.bias_initializer = bias_initializer
        # self.kernel_regularizer = kernel_regularizer
        # self.bias_regularizer = bias_regularizer
        # self.activity_regularizer = activity_regularizer
        # self.kernel_constraint = kernel_constraint
        # self.bias_constraint = bias_constraint

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        # if self.use_batch_norm:
            # self.batch_norm = BatchNormalization()
        self.kernel = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        x, a = inputs

        output = K.dot(x, self.kernel)
        output = modal_dot(a, output)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        # if self.use_batch_norm:
            # output = self.batch_norm(output)
        output = self.activation(output)

        return output

    @property
    def config(self):
        return {"channels": self.channels}

    @staticmethod
    def preprocess(a, norm=True):
        if norm:
            return gcn_filter(a)
        else:
            return laplacian(a)