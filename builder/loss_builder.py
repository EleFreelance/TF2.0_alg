import os
import tensorflow as tf
from utils.decorate import log_builder
from importlib import import_module


# 使用修饰器对Build函数实现log修饰
@log_builder
def build(config):

    package = import_module('trainer.%s' % config.trainer_type)
    trainer = getattr(package, config.trainer_type)
    return trainer(config)
