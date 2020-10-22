import tensorflow as tf
import numpy as np
import cv2
import os


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

images_path_train = "E:/BaiduNetdiskDownload/cityscapes/leftImg8bit/train/"
labels_path_train = "E:/BaiduNetdiskDownload/cityscapes/gtFine/train/"
writer = tf.io.TFRecordWriter("D:/program/deeplab_v3/data/city.tfrecord")

for region in os.listdir(images_path_train):
	the_path = images_path_train + region
	for name in os.listdir(the_path):
		image = cv2.resize(cv2.imread(os.path.join(the_path, name)), (1024, 512))[:, :, (2, 1, 0)]
		name = name.rstrip(name.split("_")[-1]) + "gtFine_labelTrainIds.png"
		label = cv2.resize(cv2.imread(os.path.join(labels_path_train + region, name), 0), (1024, 512), interpolation=cv2.INTER_NEAREST)
		h_list, w_list = np.where(label==255)
		label[h_list, w_list] = 19
		high, width, depth = np.array(np.shape(image)).astype(np.int64)
		image, label = image.tostring(), label.tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
			"image": _bytes_feature(value=image),
			"label": _bytes_feature(value=label),
			"high": _int64_feature(value=high),
			"width": _int64_feature(value=width),
			"depth": _int64_feature(value=depth)
			}))
		writer.write(example.SerializeToString())

images_path_val = "E:/BaiduNetdiskDownload/cityscapes/leftImg8bit/val/"
labels_path_val = "E:/BaiduNetdiskDownload/cityscapes/gtFine/val/"

for region in os.listdir(images_path_val):
	the_path = images_path_val + region
	for name in os.listdir(the_path):
		image = cv2.resize(cv2.imread(os.path.join(the_path, name)), (1024, 512))[:, :, (2, 1, 0)]
		name = name.rstrip(name.split("_")[-1]) + "gtFine_labelTrainIds.png"
		label = cv2.resize(cv2.imread(os.path.join(labels_path_val + region, name), 0), (1024, 512), interpolation=cv2.INTER_NEAREST)
		h_list, w_list = np.where(label==255)
		label[h_list, w_list] = 19
		high, width, depth = np.array(np.shape(image)).astype(np.int64)
		image, label = image.tostring(), label.tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
			"image": _bytes_feature(value=image),
			"label": _bytes_feature(value=label),
			"high": _int64_feature(value=high),
			"width": _int64_feature(value=width),
			"depth": _int64_feature(value=depth)
			}))
		writer.write(example.SerializeToString())

writer.close()