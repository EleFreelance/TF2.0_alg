import os
from abc import ABCMeta
from abc import abstractmethod


class base_data(object):
    def __init__(self, is_training):
        __metaclass__=ABCMeta
        print("base_data init")
        self._is_training = is_training

    @abstractmethod
    def __call__(self, *args, **kwargs):
        print("base_data call")
        # callable的意义在哪里？ 封装整个对象初始化的过程，返回一个对象，外部调用对象，根据不同的输入来改变对象的内部属性或行为
        pass

