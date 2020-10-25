from tensorflow.keras.callbacks import Callback


class MyCallBack(Callback):
    """Keras (version=2.3.1) Callback 编写模板"""

    def __init__(self):
        super().__init__()

    def on_train_begin(self, logs=None):
        """在整个训练开始时会调用次函数

            Parameters:
            ----------
                logs: dict, 该参数在当前版本默认为 None, 主要是为未来的 keras 版本的新行为预留位置
        """
        input_data = self.model.inputs
        print('On train begin', logs)
        return

    def on_train_end(self, logs=None):
        """在整个训练结束时调用次函数

            Parameters:
            ----------
                logs: dict, 该参数在当前版本默认为 None, 主要是为未来的 keras 版本的新行为预留位置
        """
        print('On train end', logs)
        return

    def on_epoch_begin(self, epoch, logs=None):
        """在每个 epoch 开始的时候调用此函数

            Parameters:
            ----------
                epoch: int, 当前为第几个 epoch, 从 1 开始
                logs: dict, 为空
        """
        print('On epoch begin', epoch, logs)
        return

    def on_epoch_end(self, epoch, logs=None):
        """在每个 epoch 结束的时候调用此函数

            Parameters:
            ----------
                epoch: int, 当前为第几个 epoch, 从 1 开始
                logs: dict, 包含了当前 epoch 的一些信息，主要的 key 有:
                    - accuracy
                    - loss
                    - val-accuracy（仅在 fit 中开启 validation 时才有）
                    - val-loss（仅在 fit 中开启 validation 时才有）
        """
        print('On epoch end', epoch, logs)
        pass

    def on_batch_begin(self, batch, logs=None):
        """在每个 batch 开始的时候调用此函数

            Parameters:
            ----------
                batch: int, 当前为第几个 batch, 从 1 开始
                logs: dict, 包含了当前 batch 的一些信息，主要的 key 有:
                    - batch: 同参数 batch
                    - size: batch 的大小
        """
        print('On batch begin', batch, logs)
        return

    def on_batch_end(self, batch, logs=None):
        """在每个 batch 结束的时候调用此函数

            Parameters:
            ----------
                batch: int, 当前为第几个 batch, 从 1 开始
                logs: dict, 包含了当前 batch 的一些信息，主要的 key 有:
                    - batch: 同参数 batch
                    - size: batch 的大小
                    - loss
                    - accuracy（仅当启用了 acc 监视）
        """
        print('On batch end', batch, logs)
        return
