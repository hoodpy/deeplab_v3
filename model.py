import tensorflow as tf
import time
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers


class Timer():
	def __init__(self):
		self.total_time = 0
		self.calls = 0
		self.start_time = 0
		self.diff = 0
		self.average_time = 0

	def tic(self):
		self.start_time = time.time()

	def toc(self, average=True):
		self.diff = time.time() - self.start_time
		self.total_time += self.diff
		self.calls += 1
		self.average_time = self.total_time / self.calls
		if average:
			return self.average_time
		return self.diff


class DeepLab_v3():
	def __init__(self, num_classes):
		self._num_classes = num_classes

	def build_network(self, images, is_training):
		with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=0.9997)):
			input_size = tf.shape(images)[1:3]
			logits, end_points = resnet_v2.resnet_v2_101(images, num_classes=None, is_training=is_training, global_pool=False, 
				output_stride=16)
			net = end_points["resnet_v2_101/block4"]

			with tf.compat.v1.variable_scope("aspp"):
				with arg_scope([layers.batch_norm], is_training=is_training):
					the_shape = tf.shape(net)[1:3]
					conv_1x1 = layers_lib.conv2d(net, 256, [1, 1], stride=1, scope="conv_1x1")
					conv_3x3_1 = layers_lib.conv2d(net, 256, [3, 3], stride=1, rate=6, scope="conv_3x3_1")
					conv_3x3_2 = layers_lib.conv2d(net, 256, [3, 3], stride=1, rate=12, scope="conv_3x3_2")
					conv_3x3_3 = layers_lib.conv2d(net, 256, [3, 3], stride=1, rate=18, scope="conv_3x3_3")

					with tf.compat.v1.variable_scope("image_level_features"):
						image_level_features = tf.reduce_mean(net, [1, 2], name="global_average_pooling", keep_dims=True)
						image_level_features = layers_lib.conv2d(image_level_features, 256, [1, 1], stride=1, scope="conv_1x1")
						image_level_features = tf.compat.v1.image.resize_bilinear(image_level_features, the_shape, name="upsample")
					net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=-1, name="concat")
					encoder_output = layers_lib.conv2d(net, 256, [1, 1], stride=1, scope="conv_1x1_concat")

			with tf.compat.v1.variable_scope("decoder"):
				with arg_scope([layers.batch_norm], is_training=is_training):
					with tf.compat.v1.variable_scope("low_level_features"):
						low_level_features = end_points["resnet_v2_101/block1/unit_3/bottleneck_v2/conv1"]
						low_level_features = layers_lib.conv2d(low_level_features, 48, [1, 1], stride=1, scope="conv_1x1")
						low_level_features_size = tf.shape(low_level_features)[1:3]

					with tf.compat.v1.variable_scope("upsampling_logits"):
						net = tf.compat.v1.image.resize_bilinear(encoder_output, low_level_features_size, name="upsample_1")
						net = tf.concat([net, low_level_features], axis=-1, name="concat")
						net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope="conv_3x3_1")
						net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope="conv_3x3_2")
						net = layers_lib.conv2d(net, self._num_classes, [1, 1], stride=1, activation_fn=None, normalizer_fn=None, 
							scope="conv_1x1")
						logits = tf.compat.v1.image.resize_bilinear(net, input_size, name="upsample_2")
						logits_softmax = tf.nn.softmax(logits, axis=-1, name="logits_softmax")
						logits_argmax = tf.argmax(logits, dimension=-1, name="logits_argmax")

		self._logits = logits
		self._logits_softmax = logits_softmax
		self._logits_argmax = logits_argmax

	def add_loss(self, annotations):
		labels, annotations = tf.reshape(self._logits, [-1, self._num_classes]), tf.reshape(annotations, [-1])
		effective_indics = tf.squeeze(tf.compat.v2.where(tf.math.less_equal(annotations, self._num_classes-1)), 1)
		prediction, ground_truth = tf.gather(labels, effective_indics), tf.gather(annotations, effective_indics)
		cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ground_truth, logits=prediction), 
			name="cross_entropy")
		self._cross_entropy = cross_entropy
		return cross_entropy

	def train_step(self, sess, train_op, global_step, merged):
		_, _cross_entropy, steps, summary = sess.run([train_op, self._cross_entropy, global_step, merged])
		return _cross_entropy, steps, summary
		