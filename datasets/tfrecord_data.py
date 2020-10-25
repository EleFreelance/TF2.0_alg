import os
import tensorflow as tf
import numpy as np
from utils.decorate import log_builder
from datasets import base_data
from functools import partial


class tfrecord_data(base_data.base_data):
    def __init__(self, config, file_names, preprocess_fn, is_training, map_func=None):
        super(tfrecord_data, self).__init__(is_training)
        print("tfrecord_data init")
        # 1、init config params
        self._config = config
        self._batch_size = config.batch_size
        if 'repeat_time' in config:
            self._repeat_time = config.repeat_time
        else:
            self._repeat_time = None
        if 'buffer_size' in config:
            self._buffer_size = config.buffer_size
        else:
            self._buffer_size = self._batch_size * 10

        # 2、init filenames and preprocess datasets
        self._file_names = file_names
        self._preprocess_fn = preprocess_fn

        self._dataset = tf.data.TFRecordDataset(self._file_names, num_parallel_reads=32)

        # 3、init map_func
        if map_func is not None:
            self._parse_func = partial(map_func, config=self._config, preprocess_fn=self._preprocess_fn)
        else:
            self._parse_func = None

    def __call__(self, *args, **kwargs):
        print("tfrecord_data call")
        # 1、将文件名读入fn_dataset
        self._dataset = tf.data.Dataset.list_files(self._file_names)

        # 2、并行化从fn_dataset中读取数据(也可使用prefetch)
        self._dataset = self._dataset.apply(
            tf.data.experimental.parallel_interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(tf.data.experimental.AUTOTUNE),
                                                     cycle_length=len(self._file_names)))

        # 3、基于dataset中的map函数并行化进行图像parse_raw
        if self._parse_func is not None:
            self._dataset = self._dataset.map(self._parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # record_dataset = record_dataset.map(self.tfrecord_parser)

        # 4、将数据集进行打乱和分批
        self._dataset = self._dataset.apply(tf.data.experimental.shuffle_and_repeat(self._batch_size))
        self._dataset = self._dataset.batch(batch_size=self._batch_size, drop_remainder=True)

        # 5、开始prefetch，分批数据在多GPU中并行
        self._dataset = self._dataset.prefetch()
        return self._dataset

