import tensorflow as tf
from utils import downsample_block, residual_block

def build_discriminator(input_tensor, 
                       reuse=False):

    with tf.variable_scope("mnist_discriminator", reuse=reuse):

        # Do something very simple:

        x = tf.reshape(input_tensor, (-1, 28*28))

        # Hidden layer:
        x = tf.layers.dense(x, 256)

        x = tf.nn.relu(x)

        x = tf.layers.dense(x, 1)

        # Apply the activation:
        x = tf.nn.sigmoid(x)

    return x