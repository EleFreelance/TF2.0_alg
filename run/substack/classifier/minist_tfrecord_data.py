import os
import tensorflow as tf
import numpy as np
from utils.decorate import log_builder
from importlib import import_module
from functools import partial
from base_class import tfrecord_data_base


class read_tfrecord_data(tfrecord_data_base.tfrecord_data_base):
    def __init__(self, config, tfrecord_fn_list, preprocess_fn_list, tfrecord_parser):
        super(read_tfrecord_data, self).__init__(config)
        self.tfrecord_fn_list = tfrecord_fn_list
        self.preprocess_fn_list = preprocess_fn_list
        self.tfrecord_parser = partial(tfrecord_parser)

    def __call__(self, *args, **kwargs):
        # 1、将文件名读入fn_dataset
        filename_dataset = tf.data.Dataset.list_files(self.tfrecord_fn_list)
        # 2、并行化从fn_dataset中读取数据(也可使用prefetch)
        #record_dataset = tf.data.TFRecordDataset(self.tfrecord_fn_list)
        record_dataset = filename_dataset.apply(
            tf.data.experimental.parallel_interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(self.batch_size),
                                                     cycle_length=4))
        # 3、基于dataset中的map函数并行化进行图像parse_raw
        record_dataset = record_dataset.map(self.tfrecord_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # record_dataset = record_dataset.map(self.tfrecord_parser)
        # 4、将数据集进行打乱和分批
        record_dataset = record_dataset.apply(tf.data.experimental.shuffle_and_repeat(self.buffer_size))
        record_dataset = record_dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        # 5、开始prefetch，分批数据在多GPU中并行
        record_dataset = record_dataset.prefetch(self.batch_size)
        # 6、生成数据batch迭代器
        one_iter = record_dataset.make_one_shot_iterator()
        return one_iter
