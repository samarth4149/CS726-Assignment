import tensorflow as tf
from skimage import io, filters
import numpy as np
import time 
import math
import sys

with open("devnagri/train/labels.txt") as f:
	labels = []
	for line in f:
		labels.append(int(line))

with open("devnagri/valid/labels.txt") as f:
	test_labels = []
	for line in f:
		test_labels.append(int(line))


#Fized sizes
train_size = 17205
input_size = 80*80
Nlabels = 104
out_size = Nlabels
test_size = 1829
batch_size = 208
init_var = 1e-2

#Parameters
learning_rate = 0.001
training_epochs = 100
hidden_1_size = 1600
hidden_2_size = 400

# print "\n\n\n\n\n pkeep : " + str(sys.argv[1])


def read_train_data(): #starting from image i, read batch_size images and populate y
	# buf = 'devnagri/train/{}.png'.format(start_index)
	# image = io.imread(buf)
	# image = np.reshape(image, (input_size, 1))
	# train_data = image;
	train_dataX = []
	train_dataY = []
	for i in range(0, train_size):
		buf = 'devnagri/train_small/{}.png'.format(i)
		image = io.imread(buf)
		image = np.reshape(image, input_size)
		image = image.astype(float)
		image=image/np.max(image)
		train_dataX.append(image)
		temp = np.zeros(Nlabels)
		temp[labels[i]] = 1
		train_dataY.append(temp);
		if i%1000==0 and i>0:
			print("Read till image:", i)
	return train_dataX, train_dataY

def read_test_data(): 
	test_dataX = []
	test_dataY = []
	for i in range(test_size):
		buf = 'devnagri/valid_small/{}.png'.format(i)
		image = io.imread(buf)
		image = np.reshape(image, input_size)
		image = image.astype(float)
		image=image/np.max(image)
		test_dataX.append(image)
		temp = np.zeros(Nlabels)
		temp[test_labels[i]] = 1
		test_dataY.append(temp);
	return test_dataX, test_dataY

def model(W1, b1, W2, b2, Wout, bout):
	# h = []
	# z = []
	# for i in range(n_layers):
	# 	h.append(tf.add(tf.matmul(x, W[i]), b[i]))
	# 	z.append(tf.nn.relu6(h[i]))
	x_h1 = tf.nn.dropout(x, pkeep1)
	h1 = tf.add(tf.matmul(x_h1, W1), b1)
	z1 = tf.nn.relu6(h1)
	z1_h2 = tf.nn.dropout(z1, pkeep1)
	h2 = tf.add(tf.matmul(z1_h2, W2), b2)
	z2 = tf.nn.relu6(h2)
	z2_out = tf.nn.dropout(z2, pkeep1)
	return tf.nn.softmax(tf.matmul(z2_out,Wout) + bout)


train_dataX, train_dataY = read_train_data()
test_dataX, test_dataY = read_test_data()


sess = tf.InteractiveSession()

#Network variables, weights and biases
x = tf.placeholder(tf.float32, shape=[None, input_size])
y_ = tf.placeholder(tf.float32, shape=[None, Nlabels])

# W = []
# b = []
# for i in range(n_layers):
# 	W.append(tf.Variable(tf.random_normal([input_size, hidden_1_size], 0, init_var)))
# 	b.append(tf.Variable(tf.random_normal([hidden_1_size], 0, init_var)))


W1 = tf.Variable(tf.random_normal([input_size, hidden_1_size], 0, init_var), name="W1")
W2 = tf.Variable(tf.random_normal([hidden_1_size, hidden_2_size], 0, init_var), name="W2")
Wout = tf.Variable(tf.random_normal([hidden_2_size, out_size], 0, init_var), name="Wout")

b1 = tf.Variable(tf.random_normal([hidden_1_size], 0, init_var), name="b1")
b2 = tf.Variable(tf.random_normal([hidden_2_size], 0, init_var), name="b2")
bout = tf.Variable(tf.random_normal([out_size], 0, init_var), name="bout")

pkeep1 = tf.placeholder("float")


saver = tf.train.Saver()
y = model(W1, b1, W2, b2, Wout, bout)


cross_entropy = -tf.reduce_sum(y_*tf.log(y+1e-10))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(Wout) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(bout)
# cross_entropy += regularizers

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)



start_time = time.clock();
sess.run(tf.initialize_all_variables())
# saver.restore(sess, "./model.ckpt")



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_iter = 10
numiters = int(math.ceil(train_size*1.0/batch_size))
for epoch in range(training_epochs):
	cost = 0
	for i in range(numiters):
		if (i+1)*batch_size > train_size :
			batchX = train_dataX[train_size-batch_size : train_size]
			batchY = train_dataY[train_size-batch_size : train_size]
		else :
			batchX = train_dataX[i*batch_size : (i+1)*batch_size]
			batchY = train_dataY[i*batch_size : (i+1)*batch_size]
		# train_step.run(feed_dict={x: batchX, y_: batchY})
		# print(len(400
		# print(batchX[0].shape)
		_, c= sess.run([train_step, cross_entropy], feed_dict={x: batchX, y_: batchY, pkeep1: 0.5})
		# print("Epoch:", epoch, "cross_entropy:", c)
		cost += c
		# print("Epoch number:", epoch, "iteration:", i)

	cost = cost/numiters
	print("Epoch number :", epoch, "Cost=", cost)
	if epoch%accuracy_iter==0 : 
		print("Test accuracy:", accuracy.eval(feed_dict={x: test_dataX, y_: test_dataY, pkeep1: 1.0}))
		save_path = saver.save(sess, "./model.ckpt")
		print "Model saved in file %s" % save_path


acc_val = accuracy.eval(feed_dict={x: test_dataX, y_: test_dataY, pkeep1: 1.0})

# outfile = open("q5-1.txt", 'a')
# outfile.write(str(pkeep1)+", " + str(acc_val) + "\n")
# outfile.close()

print("Final test accuracy:", acc_val)
print("Time taken:", time.clock() - start_time);