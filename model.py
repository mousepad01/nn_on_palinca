"""Misc model-related classes used globally in this project"""

import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *

class Res1D(Layer):
    """implementation of a residual layer
        takes as params only:
        * kernel_size
        * strides
        """

    def __init__(self, filters, kernel_size=3, strides=1):
        super(Res1D, self).__init__()

        self.conv0 = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', strides=strides)
        self.n0 = BatchNormalization()
        self.a0 = Activation('relu')

        self.conv1 = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')
        self.n1 = BatchNormalization()

        self.conv_input = Conv1D(filters=filters, kernel_size=1, strides=strides)

        self.a1 = Activation('relu')

    def call(self, input_tensor):
        
        _tmp = self.conv0(input_tensor)
        _tmp = self.n0(_tmp)
        _tmp = self.a0(_tmp)

        _tmp = self.conv1(_tmp)
        _tmp = self.n1(_tmp)

        _tmp2 = self.conv_input(input_tensor)
        _tmp += _tmp2

        output_tensor = self.a1(_tmp)
        return output_tensor

class Inception1D(Layer):
    """implementation of a custom inception layer
        does not take parameters, all of them are fixed\n
        (batch, size, filters in) -> (batch, size * 2, filters)
        """

    def __init__(self, filters):
        super(Inception1D, self).__init__()

        # branch 0

        self.conv_b0_0 = Conv1D(filters=filters, kernel_size=3, padding='same')
        self.n_b0_0 = BatchNormalization()
        self.a_b0_0 = Activation('relu')

        self.conv_b0_1 = Conv1D(filters=filters, kernel_size=3, padding='same')
        self.n_b0_1 = BatchNormalization()

        self.conv_proj_b0 = Conv1D(filters=filters, kernel_size=1)

        # branch 1

        self.conv_b1_0 = Conv1D(filters=filters, kernel_size=3, padding='same')
        self.conv_b1_01 = Conv1D(filters=filters, kernel_size=3, padding='same')
        self.n_b1_0 = BatchNormalization()
        self.a_b1_0 = Activation('relu')

        self.conv_b1_1 = Conv1D(filters=filters, kernel_size=3, padding='same')
        self.n_b1_1 = BatchNormalization()
        
        self.conv_proj_b1 = Conv1D(filters=filters, kernel_size=1)

        # final

        self.a_fin = Activation('relu')

    def call(self, input_tensor):
        
        # branch 0

        _tmp = self.conv_b0_0(input_tensor)
        _tmp = self.n_b0_0(_tmp)
        _tmp = self.a_b0_0(_tmp)

        _tmp = self.conv_b0_1(_tmp)
        output_branch0 = self.n_b0_1(_tmp)

        _tmp2 = self.conv_proj_b0(input_tensor)
        output_branch0 += _tmp2

        # branch 1
        
        _tmp = self.conv_b1_0(input_tensor)
        _tmp = self.conv_b1_01(_tmp)
        _tmp = self.n_b1_0(_tmp)
        _tmp = self.a_b1_0(_tmp)

        _tmp = self.conv_b1_1(_tmp)
        output_branch1 = self.n_b1_1(_tmp)

        _tmp2 = self.conv_proj_b1(input_tensor)
        output_branch1 += _tmp2

        # concatenation

        return tf.concat([output_branch0, output_branch1], axis=1)

# TODO test, eliminated one separate res connection
class Inception1D_(Layer):
    """implementation of a custom inception layer
        does not take parameters, all of them are fixed\n
        (batch, size, filters in) -> (batch, size * 2, filters)
        """

    def __init__(self, filters):
        super(Inception1D, self).__init__()

        # residual connection

        self.conv_id = Conv1D(filters=filters, kernel_size=1)

        # branch 0

        self.conv_b0_0 = Conv1D(filters=filters, kernel_size=3, padding='same')
        self.n_b0_0 = BatchNormalization()
        self.a_b0_0 = Activation('relu')

        self.conv_b0_1 = Conv1D(filters=filters, kernel_size=3, padding='same')
        self.n_b0_1 = BatchNormalization()

        # branch 1

        self.conv_b1_0 = Conv1D(filters=filters, kernel_size=3, padding='same')
        self.conv_b1_01 = Conv1D(filters=filters, kernel_size=3, padding='same')
        self.n_b1_0 = BatchNormalization()
        self.a_b1_0 = Activation('relu')

        self.conv_b1_1 = Conv1D(filters=filters, kernel_size=3, padding='same')
        self.n_b1_1 = BatchNormalization()

        # final

        self.a_fin = Activation('relu')

    def call(self, input_tensor):

        # residual connection

        _tmp_id = self.conv_id(input_tensor)
        
        # branch 0

        _tmp = self.conv_b0_0(input_tensor)
        _tmp = self.n_b0_0(_tmp)
        _tmp = self.a_b0_0(_tmp)

        _tmp = self.conv_b0_1(_tmp)
        output_branch0 = self.n_b0_1(_tmp)

        output_branch0 += _tmp_id

        # branch 1
        
        _tmp = self.conv_b1_0(input_tensor)
        _tmp = self.conv_b1_01(_tmp)
        _tmp = self.n_b1_0(_tmp)
        _tmp = self.a_b1_0(_tmp)

        _tmp = self.conv_b1_1(_tmp)
        output_branch1 = self.n_b1_1(_tmp)

        output_branch1 += _tmp_id

        # concatenation

        return tf.concat([output_branch0, output_branch1], axis=1)

class Foldl(Layer):
    """Foldl over slices of input tensor"""

    def __init__(self, timeslice_len, op):
        super(Foldl, self).__init__()

        self.tslice_len = timeslice_len
        self.op = op

    def call(self, input_tensor):

        def _loop(total_len, idx, acc, input_tensor, slen):
            
            acc = self.op(acc, input_tensor[:, idx * slen: (idx + 1) * slen])
            idx += 1

            return total_len, idx, acc, input_tensor, slen

        total_len = input_tensor.shape[1]
        idx = tf.constant(1)
        acc = input_tensor[:, :idx * self.tslice_len]
        slen = self.tslice_len

        len_ok = lambda total_len_, idx_, acc_, input_tensor_, slen_ : tf.less(idx_ * slen_, total_len_)
        
        _, _, acc, _, _ = tf.while_loop(len_ok, _loop, [total_len, idx, acc, input_tensor, slen])
        return acc

# https://arxiv.org/pdf/2004.11362.pdf
# https://keras.io/examples/vision/supervised-contrastive-learning/
class SupCon(Loss):
    
    def __init__(self, temperature = 1):
        super(SupCon, self).__init__()

        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight = None):

        # obtaining cosine distances, then scale them with temperature

        feature_vectors = tf.math.l2_normalize(feature_vectors, axis = 1)
        feature_matrix = tf.divide(tf.matmul(feature_vectors, tf.transpose(feature_vectors)), self.temperature)

        # softmax and then cross entropy on the previously calculated distances
        return tfa.losses.npairs_loss(tf.squeeze(labels, axis = 1), feature_matrix)
