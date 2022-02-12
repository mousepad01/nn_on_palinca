"""Misc model-related classes used globally in this project"""

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
        * strides"""

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
