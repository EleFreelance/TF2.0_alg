import os
import tensorflow as tf
import numpy as np
from utils.decorate import log_builder
from importlib import import_module
from functools import partial


def parse_raw(raw_image):
    return raw_image


def parse_func(example_proto,config,preprocess_fn):
    features = tf.parse_single_example(example_proto, features={
        'images_raw': tf.FixedLenFeature([], tf.string),
        'labels': tf.FixedLenFeature([], tf.int64),
        'pixels': tf.FixedLenFeature([], tf.int64)}
                                       )

    images_raw = features['images_raw']
    labels = features['labels']
    pixels = features['pixels']
    images = tf.decode_raw(images_raw, tf.uint8)
    images = tf.reshape(images, [28, 28, 1])

    # if preprocess_fn_list is not None:
    #     images = preprocess_fn_list(images)
    return parse_raw(images)
