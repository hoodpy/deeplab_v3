import tensorflow as tf
import numpy as np
import cv2
from model import DeepLab_v3


def vis_detection(image, label):
	#0=background 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow
	#11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=monitor
	label_colors = [np.array([[0, 0, 0]]), np.array([[128, 0, 0]]), np.array([[0, 128, 0]]), np.array([[128, 128, 0]]), 
	np.array([[0, 0, 128]]), np.array([[128, 0, 128]]), np.array([[0, 128, 128]]), np.array([[128, 128, 128]]), np.array([[64, 0, 0]]), 
	np.array([[192, 0, 0]]), np.array([[64, 128, 0]]), np.array([[192, 128, 0]]), np.array([[64, 0, 128]]), np.array([[192, 0, 128]]), 
	np.array([[64, 128, 128]]), np.array([[192, 128, 128]]), np.array([[0, 64, 0]]), np.array([[128, 64, 0]]), np.array([[0, 192, 0]]), 
	np.array([[128, 192, 0]]), np.array([[0, 64, 128]])]
	for i in range(1, 21):
		h_list, w_list = np.where(label==i)
		image[h_list, w_list, :] = 0.5 * image[h_list, w_list, :].astype(np.int32).astype(np.float32) + \
		0.5 * label_colors[i].astype(np.float32)
	return image[:, :, (2, 1, 0)].astype(np.uint8)

def vis_detection_city(image, label):
	color_list = [np.array([[128,64,128]]), np.array([[244,35,232]]), np.array([[70,70,70]]), np.array([[102,102,156]]), 
	np.array([[190,153,153]]), np.array([[153,153,153]]), np.array([[250,170,30]]), np.array([[220,220,0]]), np.array([[107,142,35]]), 
	np.array([[152,251,152]]), np.array([[70,130,180]]), np.array([[220,20,60]]), np.array([[255,0,0]]), np.array([[0,0,142]]), 
	np.array([[0,0,70]]), np.array([[0,60,100]]), np.array([[0,80,100]]), np.array([[0,0,230]]), np.array([[119,11,32]])]
	for i in range(19):
		h_list, w_list = np.where(label==i)
		image[h_list, w_list, :] = 0.5 * image[h_list, w_list, :].astype(np.int32).astype(np.float32) + \
		0.5 * color_list[i].astype(np.float32)
	return image[:, :, (2, 1, 0)].astype(np.uint8)


model_path = "D:/program/deeplab_v3/model/model120310.ckpt"
image_size, image_max_size = [int(513 * 480 / 640), 513], 513
pixel_vals = np.array([[123.68, 116.78, 103.94]])
network = DeepLab_v3(num_classes=21)
image_input = tf.placeholder(tf.uint8, [None, None, 3])
image_prepare = tf.image.convert_image_dtype(image_input, dtype=tf.float32)
image_prepare = tf.image.resize_images(image_prepare, image_size, method=0)
image_prepare = tf.image.pad_to_bounding_box(image_prepare, 0, 0, image_max_size, image_max_size)
image_prepare = tf.image.convert_image_dtype(image_prepare, dtype=tf.uint8)
image_prepare = tf.cast(tf.cast(image_prepare, tf.int32), tf.float32) - tf.convert_to_tensor(pixel_vals, dtype=tf.float32)
network.build_network(tf.expand_dims(image_prepare, axis=0), is_training=False)
logits = tf.image.crop_to_bounding_box(tf.squeeze(network._logits, axis=0), 0, 0, image_size[0], image_size[1])
logits = tf.image.resize_images(logits, [480, 640], method=0)
logits_argmax = tf.argmax(logits, dimension=-1)
saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.run(tf.compat.v1.global_variables_initializer())
saver.restore(sess, model_path)
cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow("001")
res, frame = cameraCapture.read()
while res and cv2.waitKey(1) != 27:
	frame = frame[:, :, (2, 1, 0)]
	result = sess.run(logits_argmax, feed_dict={image_input: frame})
	cv2.imshow("001", vis_detection(frame, result))
	res, frame = cameraCapture.read()
cv2.destroyAllWindows()
cameraCapture.release()