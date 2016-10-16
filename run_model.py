from __future__ import print_function
import tensorflow as tf
from skimage import io, filters
from skimage import transform
import numpy as np
import sys

def get_probability(img_list):	
	#Fized sizes
	input_size = 80*80
	Nlabels = 104
	out_size = Nlabels
	init_var = 1e-2

	#Parameters
	hidden_1_size = 1600
	hidden_2_size = 400


	sess = tf.InteractiveSession()

	def model(W1, b1, W2, b2, Wout, bout):
		h1 = tf.add(tf.matmul(x, W1), b1)
		z1 = tf.nn.relu6(h1)
		h2 = tf.add(tf.matmul(z1, W2), b2)
		z2 = tf.nn.relu6(h2)
		return tf.nn.softmax(tf.matmul(z2,Wout) + bout)

	x = tf.placeholder(tf.float32, shape=[None, input_size])

	W1 = tf.Variable(tf.random_normal([input_size, hidden_1_size], 0, init_var), name="W1")
	W2 = tf.Variable(tf.random_normal([hidden_1_size, hidden_2_size], 0, init_var), name="W2")
	Wout = tf.Variable(tf.random_normal([hidden_2_size, out_size], 0, init_var), name="Wout")

	b1 = tf.Variable(tf.random_normal([hidden_1_size], 0, init_var), name="b1")
	b2 = tf.Variable(tf.random_normal([hidden_2_size], 0, init_var), name="b2")
	bout = tf.Variable(tf.random_normal([out_size], 0, init_var), name="bout")

	saver = tf.train.Saver()

	y = model(W1, b1, W2, b2, Wout, bout)

	saver.restore(sess, "./model/model.ckpt")

	ret_prob = []

	for image_path in img_list :
		image = io.imread(image_path)
		image = np.invert(image)
		image = filters.gaussian(image, 5)
		image = transform.resize(image, (80, 80))

		image = np.reshape(image, (1, input_size))
		image = image.astype(float)
		image = image/np.max(image)

		prob = y.eval(feed_dict={x: image})
		ret_prob.append(prob)

	ret_prob = np.array(ret_prob)

	sess.close()

	return ret_prob

# for i in range(Nlabels):
# 	print(prob[0][i], "",end="")

# print()

