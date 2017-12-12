import tensorflow as tf
from utils import upsample_block, residual_block

def build_generator_fc(input_tensor):


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
    
def build_generator(input_tensor,
                    n_initial_filters=64,
                    n_blocks = 2,
                    is_training=True,
                    reuse=False,
                   ):
    """
    Build a DC GAN generator with deep convolutional layers
    Input_tensor is assumed to be reshaped into a (BATCH, L, H, F) type format (rectangular)
    """
    
    
    
    with tf.variable_scope("generator"):
        # Map the input to a small but many-filtered set up:
        
        input_shape = input_tensor.get_shape()
        print input_shape
        # Want the output tensor to have a small number of spatial dimensions (7x7 for mnist)
        
        # output size will be (W-F+2P)/S + 1
        # input of 10, F=5, P = 0, S= 2 gives output size of (10 - 5)/2 + 1 = 3
        # To get a specific outputsize (here == 7) with P == 1, and S = 2, set F as
        # 7 = 1 + (10 - F)/1 -> 6 = 10 -F, or F = 4
        
        x = tf.layers.conv2d(input_tensor,
                             n_initial_filters,
                             kernel_size=[4, 4],
                             strides=[1, 1],
                             padding='valid',
                             activation=None,
                             use_bias=False,
                             kernel_initializer=None,  # automatically uses Xavier initializer
                             kernel_regularizer=None,
                             activity_regularizer=None,
                             trainable=is_training,
                             name="Conv2D",
                             reuse=None)
    
        print x.get_shape()
    
        # Apply residual mappings and upsamplings:
        for block in xrange(n_blocks):
            x = residual_block(x, is_training = is_training,
                               kernel=[3, 3], stride=[1, 1],
                               alpha=0.0,
                               name="res_block_{}".format(block),
                               reuse=reuse)
    
            x = upsample_block(x, is_training = is_training,
                               kernel=[3, 3],
                               stride=[1, 1],
                               name="res_block_upsample_{}".format(block))
    
    
        # Apply a 1x1 convolution to map to just one output filter:
        x = tf.layers.conv2d(x,
                             1,
                             kernel_size=[3, 3],
                             strides=[1, 1],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             kernel_initializer=None,  # automatically uses Xavier initializer
                             kernel_regularizer=None,
                             activity_regularizer=None,
                             trainable=is_training,
                             name="FinalConv2D1x1",
                             reuse=None)
    
        # For the final activation, apply tanh:
        return tf.nn.tanh(x)
    
    
    