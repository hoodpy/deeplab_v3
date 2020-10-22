import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from model import Timer, DeepLab_v3


class Trainer():
	def __init__(self):
		self._batch_size = 5
		self._shuffle_size = 15
		self._image_size = [513, 513]
		self._num_classes = 20
		self._learning_rate = 1e-4
		self._epochs = 50
		self._pixel_val = np.array([[123.68, 116.78, 103.94]])
		self._file_path = "D:/program/deeplab_v3/data/city.tfrecord"
		self._pre_trained = "D:/program/deeplab_v3/pretrained_model/resnet_v2_101.ckpt"
		self._save_path = "D:/program/deeplab_v3/model_city/"
		self._log_path = "D:/program/deeplab_v3/log_city/"
		self.network = DeepLab_v3(num_classes=self._num_classes)
		self.timer = Timer()

	def parser(self, record):
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
		decode_label = tf.reshape(decode_label, [high, width, 1])
		return decode_image, decode_label

	def preprocess(self, image, label):
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		image_shape = tf.shape(image)
		scale = tf.random.uniform([1], minval=0.5, maxval=2.0)
		new_high = tf.cast(tf.math.multiply(tf.cast(image_shape[0], tf.float32), scale), tf.int32)
		new_width = tf.cast(tf.math.multiply(tf.cast(image_shape[1], tf.float32), scale), tf.int32)
		new_shape = tf.squeeze(tf.stack([new_high, new_width]), axis=1)
		image = tf.image.resize_images(image, new_shape, method=0)
		label = tf.image.resize_images(label, new_shape, method=1)
		label = tf.cast(tf.cast(label, tf.int32) - 255, tf.float32)
		image_with_label = tf.concat([image, label], axis=-1)
		image_with_label = tf.image.pad_to_bounding_box(image_with_label, 0, 0, tf.math.maximum(new_shape[0], self._image_size[0]),
			tf.math.maximum(new_shape[1], self._image_size[1]))
		image_with_label = tf.image.random_crop(image_with_label, [self._image_size[0], self._image_size[1], 4])
		image, label = image_with_label[:, :, :image_shape[2]], image_with_label[:, :, image_shape[2]]
		image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
		image = tf.cast(tf.cast(image, tf.int32), tf.float32) - tf.convert_to_tensor(self._pixel_val, dtype=tf.float32)
		label = tf.cast(label + 255., tf.int32)
		image.set_shape([self._image_size[0], self._image_size[1], 3])
		label.set_shape([self._image_size[0], self._image_size[1]])
		return image, label

	def get_dataset(self):
		dataset = tf.data.TFRecordDataset(self._file_path)
		dataset = dataset.map(self.parser)
		dataset = dataset.map(lambda image, label: self.preprocess(image, label))
		dataset = dataset.shuffle(self._shuffle_size).repeat(self._epochs).batch(self._batch_size)
		self.iterator = dataset.make_initializable_iterator()
		image_batch, label_batch = self.iterator.get_next()
		return image_batch, label_batch

	def get_load_variables(self):
		added, exclusion = "resnet_v2_101", "resnet_v2_101/logits"
		variables_to_restore = []
		for var in slim.get_model_variables():
			if (added in var.op.name) and (exclusion not in var.op.name):
				variables_to_restore.append(var)
		return variables_to_restore

	def train(self):
		config = tf.compat.v1.ConfigProto()
		config.allow_soft_placement = True
		config.gpu_options.allow_growth = True
		with tf.compat.v1.Session(config=config) as sess:
			global_step = tf.Variable(0, trainable=False)
			learning_rate = tf.Variable(self._learning_rate, trainable=False)
			tf.compat.v1.summary.scalar("learning_rate", learning_rate)

			image_batch, label_batch = self.get_dataset()
			image_batch = tf.reshape(image_batch, [self._batch_size, self._image_size[0], self._image_size[1], 3])
			label_batch = tf.reshape(label_batch, [self._batch_size, self._image_size[0], self._image_size[1]])

			self.network.build_network(image_batch, is_training=True)
			cross_entropy = self.network.add_loss(label_batch)
			tf.compat.v1.summary.scalar("cross_entropy", cross_entropy)

			update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

			self.saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=5)
			load_fn = slim.assign_from_checkpoint_fn(self._pre_trained, self.get_load_variables(), ignore_missing_vars=True)
			merged = tf.compat.v1.summary.merge_all()
			summary_writer = tf.compat.v1.summary.FileWriter(self._log_path, sess.graph)

			sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
			load_fn(sess)
			print("Load network: " + self._pre_trained)
			sess.run(self.iterator.initializer)
			sess.run([tf.compat.v1.assign(learning_rate, self._learning_rate), tf.compat.v1.assign(global_step, 0)])

			while True:
				try:
					self.timer.tic()
					_cross_entropy, steps, summary = self.network.train_step(sess, train_op, global_step, merged)
					summary_writer.add_summary(summary, steps)
					self.timer.toc()
					if (steps + 1) % 24880:
						sess.run(tf.compat.v1.assign(learning_rate, self._learning_rate * 0.1))
					if (steps + 1) % 622 == 0:
						print(">>>Steps: %d\n>>>Cross_entropy: %.6f\n>>>Average_time: %.6fs\n" % (steps + 1, _cross_entropy, 
							self.timer.average_time))
					if (steps + 1) % 6220 == 0:
						self.snap_shot(sess, steps + 1)
				except tf.errors.OutOfRangeError:
					break

	def snap_shot(self, sess, iter):
		network = self.network
		file_name = self._save_path + "model" + str(iter) + ".ckpt"
		self.saver.save(sess, file_name)
		print("Wrote snapshot to: " + file_name + "\n")


if __name__ == "__main__":
	trainer = Trainer()
	trainer.train()