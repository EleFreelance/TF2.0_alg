from functools import wraps
import tensorflow as tf


def log_builder(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        return output

    return wrapper
