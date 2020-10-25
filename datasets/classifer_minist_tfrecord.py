import os
import tensorflow as tf
import numpy as np
from datasets import tfrecord_data
from functools import partial


class classifer_minist_tfrecord(tfrecord_data.tfrecord_data):
    def __init__(self, config, filenames, preprocess_fn, is_training, map_func=None):
        super(classifer_minist_tfrecord, self).__init__(config, filenames, preprocess_fn, is_training,map_func)
        print("classifer_minist_tfrecord init")

    # def __call__(self, *args, **kwargs):
    #     super(classifer_minist_tfrecord,self).__call__(*args, **kwargs)
    #     print("classifer_minist_tfrecord call")
    #     pass

    def next(self):
        one_iter = self._dataset.make_one_shot_iterator()
        return one_iter.get_next()
