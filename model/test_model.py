import tensorflow as tf
from model.common import ConvBlock


class SimpleCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.block1 = ConvBlock(16, 3)
        self.block2 = ConvBlock(32, 3)
        self.block3 = ConvBlock(64, 3)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=True):
        output = self.block1(x, training=training)
        output = self.block2(output, training=training)
        output = self.block3(output, training=training)
        output = self.global_pool(output)
        logits = self.classifier(output)
        return logits
