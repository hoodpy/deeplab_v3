import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def parser(record):
	features = tf.io.parse_single_example(record, features={
		"image": tf.io.FixedLenFeature([], tf.string),
		"label": tf.io.FixedLenFeature([], tf.string),
		"high": tf.io.FixedLenFeature([], tf.int64),
		"width": tf.io.FixedLenFeature([], tf.int64),
		"depth": tf.io.FixedLenFeature([], tf.int64)
		})
	high, width = tf.cast(features["high"], tf.int32), tf.cast(features["width"], tf.int32)
	depth = tf.cast(features["depth"], tf.int32)
	decode_image, decode_label = tf.decode_raw(features["image"], tf.uint8), tf.decode_raw(features["label"], tf.uint8)
	decode_image = tf.reshape(decode_image, [high, width, depth])
	decode_label = tf.reshape(decode_label, [high, width])
	return decode_image, decode_label

file_path = "D:/program/deeplab_v3/data/city.tfrecord"
dataset = tf.data.TFRecordDataset(file_path)
dataset = dataset.map(parser)
dataset = dataset.shuffle(20).repeat(5).batch(5)
iterator = dataset.make_initializable_iterator()
image_batch, label_batch = iterator.get_next()

sess = tf.compat.v1.Session()
sess.run(iterator.initializer)
result1, result2 = sess.run([image_batch, label_batch])
result1, result2 = result1[0], result2[0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6))
ax1.imshow(result1)
ax2.imshow(result2, cmap="gray")
plt.show()