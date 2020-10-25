import os
import tensorflow as tf
from utils.decorate import log_builder
from importlib import import_module


# 使用修饰器对Build函数实现log修饰
@log_builder
def build(train_config):
    initial_learning_rate = 1e-4
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    return optimizer
