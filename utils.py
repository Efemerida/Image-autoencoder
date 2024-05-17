import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import tensorflow as tf

def load_dataset():
	with np.load("mnist.npz") as f:
		# convert from RGB to Unit RGB
		x_train = f['x_train'].astype("float32") / 255

		# reshape from (60000, 28, 28) into (60000, 784)
		x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

		# labels
		y_train = f['y_train']

		# convert to output layer format
		y_train = np.eye(10)[y_train]

		return x_train, y_train


def load_correct():
	image, label = load_dataset()
	map = {}
	for im,lb in zip(image, label):
		if (map.keys().__sizeof__() == 10): break

		elif(not map.keys().__contains__(np.argmax(lb))):
			rr = np.reshape(im, (-1, 1))
			map[np.argmax(lb)] = rr
	return map

def load_test_dataset():
	with np.load("mnist.npz") as f:
		# convert from RGB to Unit RGB
		x_test = f['x_test'].astype("float32") / 255

		# reshape from (60000, 28, 28) into (60000, 784)
		x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

		# labels
		y_test = f['y_test']

		# convert to output layer format
		y_test = np.eye(10)[y_test]

		return x_test, y_test


def load_image_jpg():

	folder_path = 'images'

	file_list = os.listdir(folder_path)

	images = [file for file in file_list if file.endswith('.jpg')]
	x_train = []
	i = 0
	for im in images:
		relative_path = os.path.relpath(os.path.join(folder_path, im))

		image = Image.open(relative_path)
		a = i/len(images)
		print(a)
		try:
			standard_size = (1024, 1024)
			st_img = tf.image.resize(image, standard_size)
			# Преобразование изображения в массив numpy
			arr = np.array(st_img)/255
			x_train.append(np.array(arr))
			i+=1
		except:
			print('error')
			# tensor_images = tf.convert_to_tensor(image, dtype=tf.float32)
			# tensor_images = tf.expand_dims(tensor_images, axis=-1)  # Добавляем измерение для каналов
			# tensor_images = tf.expand_dims(tensor_images, axis=0)
			# standard_size = (1024, 1024)
			# st_img = tf.image.resize(tensor_images, standard_size)
			# # Преобразование изображения в массив numpy
			# arr = np.array(st_img)/255
			# x_train[i] = np.array(arr)
			# i += 1
		if a > 0.05: break
	return np.array(x_train)

# load_image_jpg()