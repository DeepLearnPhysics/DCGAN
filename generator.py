import tensorflow as tf
from utils import upsample_block, residual_block, leaky_relu

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
    
    
    
def build_generator_progressive(input_tensor,
                    alphas,
                    leaky_relu_param=0.0,
                    n_initial_filters=64,
                    n_blocks = 2,
                    is_training=True,
                    output_size = 8,
                    reuse=True,):

    """
    This function will build a generator (in a non resnet way)
    Such that it will progressively generate higher resolution images, 
    Starting at a low resolution of 4x4 output, and gradually increasing
    """

    with tf.variable_scope("generator_progressive", reuse = tf.AUTO_REUSE):

        # Map the initial set of random numbers to a 4x4xn_initial_filters space:

        x = tf.layers.conv2d(input_tensor,
                             n_initial_filters,
                             kernel_size=[7, 7],
                             strides=[1, 1],
                             padding='valid',
                             activation=None,
                             use_bias=False,
                             kernel_initializer=None,  # automatically uses Xavier initializer
                             kernel_regularizer=None,
                             activity_regularizer=None,
                             trainable=is_training,
                             name="Conv2D")
    
        #This should start x as 4x4xn_filters, assuming 10x10 input

        current_size = int(x.get_shape()[1])

        current_n_filters = n_initial_filters
        while current_size < output_size:

            next_size = 2*current_size
            subname = "{}to{}".format(current_size, next_size)

            next_n_filters = int(current_n_filters)
            if next_n_filters == 0:
                next_n_filters = 1
            # Need to upsample the data.  
            # First, apply a linear interpolation to just double the resolution
            # Also, halve the number of filters

            x = tf.image.resize_nearest_neighbor(x,(next_size, next_size))

            # On one path, this goes through a convolutional step:

            # Batch normalization is applied first:
            y = tf.layers.batch_normalization(x,
                                          axis=-1,
                                          momentum=0.99,
                                          epsilon=0.001,
                                          center=True,
                                          scale=True,
                                          beta_initializer=tf.zeros_initializer(),
                                          gamma_initializer=tf.ones_initializer(),
                                          moving_mean_initializer=tf.zeros_initializer(),
                                          moving_variance_initializer=tf.ones_initializer(),
                                          beta_regularizer=None,
                                          gamma_regularizer=None,
                                          training=is_training,
                                          trainable=is_training,
                                          name="BatchNorm_{}".format(subname),)
        
            # ReLU:
            y = leaky_relu(y, leaky_relu_param)


            y = tf.layers.conv2d(y,
                             next_n_filters,
                             kernel_size=[3, 3],
                             strides=[1, 1],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             kernel_initializer=None,  # automatically uses Xavier initializer
                             kernel_regularizer=None,
                             activity_regularizer=None,
                             trainable=is_training,
                             name="Conv2D_{}".format(subname))

            # Use the correct alpha to do it:
            alpha = tf.get_default_graph().get_tensor_by_name("alpha_{}:0".format(subname))

            x = (1 - alpha)*x + alpha*y

            # # Put the current stage through a residual block:
            # x = residual_block(x, is_training = is_training,
            #                    kernel=[3, 3], stride=[1, 1],
            #                    alpha=0.0,
            #                    name="res_block_{}".format(subname),
            #                    reuse=reuse)


            current_size *= 2

        #Before returning, map the current images to a single filter:

        # Batch normalization is applied first:
        x = tf.layers.batch_normalization(x,
                                      axis=-1,
                                      momentum=0.99,
                                      epsilon=0.001,
                                      center=True,
                                      scale=True,
                                      beta_initializer=tf.zeros_initializer(),
                                      gamma_initializer=tf.ones_initializer(),
                                      moving_mean_initializer=tf.zeros_initializer(),
                                      moving_variance_initializer=tf.ones_initializer(),
                                      beta_regularizer=None,
                                      gamma_regularizer=None,
                                      training=is_training,
                                      trainable=is_training,
                                      name="BatchNormFinal_{}".format(current_size),)
    
        # ReLU:
        x = leaky_relu(x, leaky_relu_param)


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
                         name="Conv2DFinal_{}".format(current_size))

        # Apply tanh as the final activation:
        x = tf.nn.tanh(x)


    return x