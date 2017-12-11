import tensorflow as tf
from utils import upsample_block, residual_block

def build_generator(input_tensor):


    with tf.variable_scope("mnist_generator"):


        # Hidden layer:
        x = tf.layers.dense(input_tensor, 14*14)

        # Leaky relu:
        x = tf.maximum(0.01*x, x)

        # Hidden layer:
        x = tf.layers.dense(input_tensor, 14*14)

        # Leaky relu:
        x = tf.maximum(0.01*x, x)


        x = tf.layers.dense(x, 28*28)

        x = tf.nn.tanh(x)

        return x