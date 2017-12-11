import tensorflow as tf
from utils import downsample_block, residual_block

def build_classifier(input_tensor, 
                     n_output_classes=10,
                     is_training=True,
                     n_initial_filters=12,
                     initial_kernel=3,
                     initial_stride=1,
                     n_blocks=4,
                     downsample_interval=1):

    with tf.variable_scope("mnist_classifier"):

        # Initial convolutional layer:
        x = tf.layers.conv2d(input_tensor,
                             n_initial_filters,
                             kernel_size=(initial_kernel,
                                          initial_kernel),
                             strides=(initial_stride,
                                      initial_stride),
                             padding='same',
                             activation=None,
                             use_bias=False,
                             bias_initializer=tf.zeros_initializer(),
                             trainable=is_training,
                             name="InitialConv2D",
                             reuse=None)

        for i in xrange(n_blocks):

            if i != 0 and i % downsample_interval == 0:
                x = downsample_block(x, name="res_block_downsample_{}".format(i),
                                     is_training=is_training)
            else:
                x = residual_block(x, name="res_block_{}".format(i),
                                   is_training=is_training)

        # A final convolution to map the features onto the right space:
        with tf.variable_scope("final_pooling"):
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
            x = tf.nn.relu(x, name="final_pooling")

            x = tf.layers.conv2d(x,
                                 n_output_classes,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 padding='same',
                                 data_format='channels_last',
                                 dilation_rate=(1, 1),
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=None,  # automatically uses Xavier initializer
                                 bias_initializer=tf.zeros_initializer(),
                                 kernel_regularizer=None,
                                 bias_regularizer=None,
                                 activity_regularizer=None,
                                 trainable=is_training,
                                 name="Conv2DBottleNeck",
                                 # name="convolution_globalpool_bottleneck1x1",
                                 reuse=None)

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

            # Reshape to remove empty dimensions:
            x = tf.reshape(x, [tf.shape(x)[0], n_output_classes],
                           name="global_pooling_reshape")
            # Apply the activation:
            x = tf.nn.softmax(x, dim=-1)

        return x