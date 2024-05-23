import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import InputSpec
from tensorflow.keras import activations, initializers, regularizers, constraints

class DenseTied(Layer):
        def __init__(self, units,
                 tie_to=None, # input layer name for weight-tying
                 transpose=False, # transpose weights from tie_to layer
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 **kwargs):
            super(DenseTied, self).__init__(**kwargs)
            self.units = units
            self.tie_to = tie_to
            self.transpose = transpose
            self.activation = activations.get(activation)
            self.use_bias = use_bias
            self.kernel_initializer = initializers.get(kernel_initializer)
            self.bias_initializer = initializers.get(bias_initializer)
            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.activity_regularizer = regularizers.get(activity_regularizer)
            self.kernel_constraint = constraints.get(kernel_constraint)
            self.bias_constraint = constraints.get(bias_constraint)
            self.trainable = trainable
            self.input_spec = InputSpec(min_ndim=2)
            self.supports_masking = True
        
        def build(self, input_shape):
            assert len(input_shape) >= 2
            input_dim = input_shape[-1]
            if self.transpose:
                self.kernel = tf.transpose(self.tie_to.kernel)
            else:
                self.kernel = self.tie_to.kernel
            if self.use_bias:
                self.bias = self.add_weight(shape=(self.units,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint,
                                            trainable=True)
            else:
                self.bias = None

        def call(self, inputs):
            output = tf.tensordot(inputs, self.kernel, axes=1)
            if self.use_bias:
                output = tf.math.add(output, self.bias)
            if self.activation is not None:
                output = self.activation(output)
            return output
        
        def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) >= 2
            assert input_shape[-1]
            output_shape = list(input_shape)
            output_shape[-1] = self.units
            return tuple(output_shape)

        def get_config(self):
            config = {
                'units': self.units,
                'activation': activations.serialize(self.activation),
                'use_bias': self.use_bias,
                'kernel_initializer': initializers.serialize(self.kernel_initializer),
                'bias_initializer': initializers.serialize(self.bias_initializer),
                'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                'kernel_constraint': constraints.serialize(self.kernel_constraint),
                'bias_constraint': constraints.serialize(self.bias_constraint)
            }

            base_config = super(DenseTied, self).get_config()

            return dict(list(base_config.items()) + list(config.items()))





