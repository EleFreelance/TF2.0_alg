import tensorflow as tf


class ConvBlock(tf.keras.Model):
    """Convolutional Block consisting of (conv->bn->relu).
    Arguments:
      num_filters: number of filters passed to a convolutional layer.
      kernel_size: the size of convolution kernel
      weight_decay: weight decay
      dropout_rate: dropout rate.
    """

    def __init__(self, num_filters, kernel_size,
                 weight_decay=1e-4, dropout_rate=0.):
        super(ConvBlock, self).__init__()

        self.conv = tf.keras.layers.Conv2D(num_filters,
                                           kernel_size,
                                           padding="same",
                                           use_bias=False,
                                           kernel_initializer="he_normal",
                                           kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=True):
        output = self.conv(x)
        output = self.bn(x, training=training)
        output = tf.nn.relu(output)
        output = self.dropout(output, training=training)
        return output
