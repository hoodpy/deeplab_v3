import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from model import DeepLab_v3


def vis_detection(image, label):
	color_list = [np.array([[128,64,128]]), np.array([[244,35,232]]), np.array([[70,70,70]]), np.array([[102,102,156]]), 
	np.array([[190,153,153]]), np.array([[153,153,153]]), np.array([[250,170,30]]), np.array([[220,220,0]]), np.array([[107,142,35]]), 
	np.array([[152,251,152]]), np.array([[70,130,180]]), np.array([[220,20,60]]), np.array([[255,0,0]]), np.array([[0,0,142]]), 
	np.array([[0,0,70]]), np.array([[0,60,100]]), np.array([[0,80,100]]), np.array([[0,0,230]]), np.array([[119,11,32]])]
	for i in range(19):
		h_list, w_list = np.where(label==i)
		image[h_list, w_list, :] = 0.5 * image[h_list, w_list, :].astype(np.int32).astype(np.float32) + \
		0.5 * color_list[i].astype(np.float32)
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))
	ax.imshow(image.astype(np.uint8))


files_path = "E:/BaiduNetdiskDownload/cityscapes/leftImg8bit/test/bielefeld"
image_shape, pixels_mean = [int(1024 * 513 / 2048), 513], np.array([[123.68, 116.78, 103.94]])
network = DeepLab_v3(num_classes=20)
image_input = tf.placeholder(tf.uint8, [1024, 2048, 3])
image_prepare = tf.image.convert_image_dtype(image_input, dtype=tf.float32)
image_prepare = tf.image.resize_images(image_prepare, [image_shape[0], image_shape[1]], method=0)
image_prepare = tf.image.pad_to_bounding_box(image_prepare, 0, 0, 513, 513)
image_prepare = tf.image.convert_image_dtype(image_prepare, dtype=tf.uint8)
image_prepare = tf.cast(tf.cast(image_prepare, tf.int32), tf.float32) - tf.convert_to_tensor(pixels_mean, dtype=tf.float32)
network.build_network(tf.expand_dims(image_prepare, axis=0), is_training=False)
logits = tf.image.crop_to_bounding_box(tf.squeeze(network._logits, axis=0), 0, 0, image_shape[0], image_shape[1])
logits_argmax = tf.argmax(tf.image.resize_images(logits, [512, 1024], method=0), dimension=-1)
saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.run(tf.compat.v1.global_variables_initializer())
saver.restore(sess, "D:/program/deeplab_v3/model_city/model31100.ckpt")
for name in os.listdir(files_path)[:10]:
	result = sess.run(logits_argmax, feed_dict={image_input: cv2.imread(os.path.join(files_path, name))[:, :, (2, 1, 0)]})
	vis_detection(cv2.resize(cv2.imread(os.path.join(files_path, name)), (1024, 512))[:, :, (2, 1, 0)], result)
plt.show()