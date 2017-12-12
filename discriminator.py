import tensorflow as tf
from utils import downsample_block, residual_block

def build_discriminator_fc(input_tensor, 
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

def build_discriminator(input_tensor,
                        n_initial_filters=12,
                        n_blocks = 2,
                        is_training=True,
                        alpha=0.2,
                        reuse=False):
    """
    Build a DC GAN discriminator with deep convolutional layers
    Input_tensor is assumed to be reshaped into a (BATCH, L, H, F) type format (rectangular)
    """
    
    
    
    with tf.variable_scope("discriminator", reuse=reuse):
        # Map the input to a small but many-filtered set up:
        
        x = tf.layers.conv2d(input_tensor,
                             n_initial_filters,
                             kernel_size=[3, 3],
                             strides=[1, 1],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             kernel_initializer=None,  # automatically uses Xavier initializer
                             kernel_regularizer=None,
                             activity_regularizer=None,
                             trainable=is_training,
                             name="Conv2D",
                             reuse=reuse)
    
    
        # Apply residual mappings and upsamplings:
        for block in xrange(n_blocks):
            x = residual_block(x, is_training = is_training,
                               kernel=[3, 3], stride=[1, 1],
                               alpha=alpha,
                               name="res_block_{}".format(block),
                               reuse=reuse)
    
            x = downsample_block(x, is_training = is_training,
                                 kernel=[3, 3],
                                 stride=[1, 1],
                                 alpha=alpha,
                                 name="res_block_upsample_{}".format(block),
                                 reuse=reuse)
    
    
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
                             reuse=reuse)
        
        #Apply global average pooling to the final layer, then sigmoid activation
        
        # For global average pooling, need to get the shape of the input:
        shape = (x.shape[1], x.shape[2])

        x = tf.nn.pool(x,
                       window_shape=shape,
                       pooling_type="AVG",
                       padding="VALID",
                       dilation_rate=None,
                       strides=None,
                       name="GlobalAveragePool",
                       data_format=None)
        x = tf.reshape(x, (-1, 1))
            
        
        # For the final activation, apply sigmoid:
        return tf.nn.sigmoid(x)
    



def build_discriminator_progressive(input_tensor,
                                    leaky_relu_param=0.0,
                    n_filters=64,
                    n_blocks = 2,
                    is_training=True,
                    reuse=True,):

    """
    This function will build a discriminator to decide if an image is real
    or fake.
    Starting at a low resolution of 4x4 output, and gradually increasing

    To make it easy to reuse weights, every level has a fixed number of
    filters

    """

    with tf.variable_scope("discriminator_progressive", reuse = tf.AUTO_REUSE):

        # Map the initial set of random numbers to a 4x4xn_initial_filters space:

        x = tf.layers.conv2d(input_tensor,
                             n_filters,
                             kernel_size=[3, 3],
                             strides=[1, 1],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             kernel_initializer=None,  # automatically uses Xavier initializer
                             kernel_regularizer=None,
                             activity_regularizer=None,
                             trainable=is_training,
                             name="Conv2D")
    
        current_size = int(x.get_shape()[1])

        while current_size > 4:
            current_size = int(x.get_shape()[1])

            next_size = int(0.5*current_size)
            subname = "{}to{}".format(current_size, next_size)
            
            x = residual_block(x,
                               is_training,
                               alpha=leaky_relu_param,
                               name="res_block_{}".format(subname))


            x = downsample_block(x,
                                 is_training,
                                 alpha=leaky_relu_param,
                                 name="downsample_block_{}".format(subname))


        # For global average pooling, need to get the shape of the input:
        shape = (x.shape[1], x.shape[2])

        x = tf.nn.pool(x,
                       window_shape=shape,
                       pooling_type="AVG",
                       padding="VALID",
                       dilation_rate=None,
                       strides=None,
                       name="GlobalAveragePool",
                       data_format=None)
        x = tf.reshape(x, (-1, 1))
            
        
        # For the final activation, apply sigmoid:
        return tf.nn.sigmoid(x)

    return x