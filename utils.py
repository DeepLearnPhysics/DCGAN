import tensorflow as tf

def leaky_relu(input_tensor, alpha=0.01):
    """
    @brief      Leaky Relu allows a small gradient to flow when values are below zero
    
    @param      alpha         Set alpha = 0 to reproduce the normal relu operation
    """
    return tf.maximum(input_tensor, alpha*input_tensor)

def residual_block( input_tensor,
                   is_training,
                   kernel=[3, 3],
                   stride=[1, 1],
                   alpha=0.0,
                   name="",
                   reuse=False):
    """
    @brief      Create a residual block and apply it to the input tensor

    @param      input_tensor  The input tensor
    @param      kernel        Size of convolutional kernel to apply
    @param      n_filters     Number of output filters

    @return     { Tensor with the residual network applied }
    """

    # Residual block has the identity path summed with the output of
    # BN/Relu/Conv2d applied twice

    # Assuming channels last here:
    n_filters = input_tensor.shape[-1]

    with tf.variable_scope(name + "_0"):
        # Batch normalization is applied first:
        x = tf.layers.batch_normalization(input_tensor,
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
                                          name="BatchNorm",
                                          reuse=reuse)
        # ReLU:
        x = leaky_relu(x, alpha)

        # Conv2d:
        x = tf.layers.conv2d(x, n_filters,
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

    # Apply everything a second time:
    with tf.variable_scope(name + "_1"):

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
                                          name="BatchNorm",
                                          reuse=reuse)
        # ReLU:
        x = leaky_relu(x, alpha)

        # Conv2d:
        x = tf.layers.conv2d(x,
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
                             name="Conv2D",
                             reuse=reuse)

    # Sum the input and the output:
    with tf.variable_scope(name+"_add"):
      x = tf.add(x, input_tensor, name="Add")
    return x


def downsample_block(input_tensor,
                     is_training,
                     kernel=[3, 3],
                     stride=[1, 1],
                     alpha=0.01,
                     name="",
                     reuse = False):
    """
    @brief      Create a residual block and apply it to the input tensor

    @param      input_tensor  The input tensor
    @param      kernel        Size of convolutional kernel to apply
    @param      n_filters     Number of output filters

    @return     { Tensor with the residual network applied }
    """

    # Residual block has the identity path summed with the output of
    # BN/Relu/Conv2d applied twice

    # Assuming channels last here:
    n_filters = 2*input_tensor.get_shape().as_list()[-1]

    with tf.variable_scope(name + "_0"):
        # Batch normalization is applied first:
        x = tf.layers.batch_normalization(input_tensor,
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
                                          name="BatchNorm",
                                          reuse=reuse)
        # ReLU:
        x = leaky_relu(x, alpha)

        # Conv2d:
        x = tf.layers.conv2d(x, n_filters,
                             kernel_size=[3, 3],
                             strides=[2, 2],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             kernel_initializer=None,  # automatically uses Xavier initializer
                             kernel_regularizer=None,
                             activity_regularizer=None,
                             trainable=is_training,
                             name="Conv2D",
                             reuse=reuse)

    # Apply everything a second time:
    with tf.variable_scope(name + "_1"):

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
                                          name="BatchNorm",
                                          reuse=reuse)
        # ReLU:
        x = leaky_relu(x, alpha)

        # Conv2d:
        x = tf.layers.conv2d(x,
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
                             name="Conv2D",
                             reuse=reuse)

    # Map the input tensor to the output tensor with a 1x1 convolution
    with tf.variable_scope(name+"identity"):
        y = tf.layers.conv2d(input_tensor,
                             n_filters,
                             kernel_size=[1, 1],
                             strides=[2, 2],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             kernel_initializer=None,  # automatically uses Xavier initializer
                             kernel_regularizer=None,
                             activity_regularizer=None,
                             trainable=is_training,
                             name="Conv2D1x1",
                             reuse=reuse)

    # Sum the input and the output:
    with tf.variable_scope(name+"_add"):
        x = tf.add(x, y)
    return x



def upsample_block(input_tensor,
                   is_training,
                   kernel=[3, 3],
                   stride=[1, 1],
                   alpha = 0.0,
                   name=""):
    """
    @brief      Create a residual block and apply it to the input tensor

    @param      input_tensor  The input tensor
    @param      kernel        Size of convolutional kernel to apply
    @param      n_filters     Number of output filters

    @return     { Tensor with the residual network applied }
    """

    # Residual block has the identity path summed with the output of
    # BN/Relu/Conv2d applied twice

    # Assuming channels last here:
    n_filters = int(0.5*input_tensor.get_shape().as_list()[-1])
    if n_filters == 0:
      n_filters = 1

    with tf.variable_scope(name + "_0"):
        # Batch normalization is applied first:
        x = tf.layers.batch_normalization(input_tensor,
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
                                          name="BatchNorm",
                                          reuse=None)
        # ReLU:
        x = leaky_relu(x, alpha)

        # Conv2d:
        x = tf.layers.conv2d_transpose(x, n_filters,
                             kernel_size=[3, 3],
                             strides=[2, 2],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             kernel_initializer=None,  # automatically uses Xavier initializer
                             kernel_regularizer=None,
                             activity_regularizer=None,
                             trainable=is_training,
                             name="Conv2DTrans",
                             reuse=None)

    # Apply everything a second time:
    with tf.variable_scope(name + "_1"):

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
                                          name="BatchNorm",
                                          reuse=None)
        # ReLU:
        x = leaky_relu(x, alpha)

        # Conv2d:
        x = tf.layers.conv2d(x,
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
                             name="Conv2D",
                             reuse=None)

    # Map the input tensor to the output tensor with a 1x1 convolution
    with tf.variable_scope(name+"identity"):
        y = tf.layers.conv2d_transpose(input_tensor,
                             n_filters,
                             kernel_size=[1, 1],
                             strides=[2, 2],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             kernel_initializer=None,  # automatically uses Xavier initializer
                             kernel_regularizer=None,
                             activity_regularizer=None,
                             trainable=is_training,
                             name="Conv2DTrans1x1",
                             reuse=None)

    # Sum the input and the output:
    with tf.variable_scope(name+"_add"):
        x = tf.add(x, y)
    return x