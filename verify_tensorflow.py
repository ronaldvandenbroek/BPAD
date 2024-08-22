# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
# print(tf.config.list_physical_devices('GPU'))