import io
import numpy as np
import tensorflow as tf


def tf_gamma(img):
    return tf.clip_by_value(img**(1/2.2),0,1)

def tf_tone_mapping(img):
    return img

class Tensorboard():
    def __init__(self, logdir):
        self.writer = tf.summary.create_file_writer(logdir)

    def close(self):
        self.writer.close()

    def log_scalar(self, tag, value, global_step):
        with self.writer.as_default():
            tf.summary.scalar(name=tag,data=value,step=global_step)

    def log_histogram(self, tag, values, global_step, bins):
        pass

    def log_image(self, tag, img, global_step):
        ima_show=img[0:3]
        with self.writer.as_default():
            tf.summary.image(name=tag,data=ima_show,step=global_step,max_outputs=3)