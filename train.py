import os
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.callbacks import Callback
import numpy as np
import yaml
import time
from builder import input_reader_builder, optimizer_builder
from model import test_model
import config_local


def main(config):
    # 1、init config
    config = config.base_parse()

    # 2、build input reader
    data_sets = input_reader_builder.build(config, is_training=config.is_training)
    steps_per_epoch = 300000 // config.batch_size
    # 3、build model
    num_classes = 4
    model = test_model.SimpleCNN(num_classes=num_classes)

    # 4、build optimizal and loss
    optimizer = optimizer_builder.build(config.train_config)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 5、callback for checkpoint and tensor board
    
    # 6、train loop (compile and fit)
    model.compile(optimizer, loss_object)
    model.fit(data_sets, epochs=10, steps_per_epoch=steps_per_epoch, callbacks=)

    # # define a train step
    # @tf.function
    # def train_step(inputs, targets):
    #     with tf.GradientTape() as tape:
    #         predictions = model(inputs, training=True)
    #         loss = loss_object(targets, predictions)
    #         loss += sum(model.losses)  # add other losses
    #     # compute gradients and update variables
    #     fp32_grads = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(fp32_grads, model.trainable_variables))
    #
    # # train loop
    # epochs = 10
    # for epoch in range(epochs):
    #     print('Start of epoch %d' % (epoch,))
    #     # Iterate over the batches of the dataset
    #     for step, (inputs, targets) in enumerate(data_sets):
    #         start_time = time.time()
    #         train_step(inputs, targets)
    #         checkpoint.step.assign_add(1)
    #         time_elapse = time.time() - start_time
    #         # log every 20 step
    #         if step % 20 == 0:
    #             manager.save()  # save checkpoint
    #             print('Epoch: {}, Step: {} ,Time loss:{}'.format(
    #                 epoch, step, time_elapse)
    #             )


if __name__ == "__main__":
    main(config_local)
