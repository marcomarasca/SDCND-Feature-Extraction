import time
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from scipy.misc import imread
from get_data import get_data
get_data()
from alexnet import AlexNet

sign_names = pd.read_csv('signnames.csv')
nb_classes = 43

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# NOTE: By setting `feature_extract` to `True` we return
# the second to last layer.
fc7 = AlexNet(resized, feature_extract=True)

# Replace the last layer of AlexNet (e.g. from 1000 to 43)

shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))

# Define a new fully connected layer followed by a softmax activation to classify
# the traffic signs.

logits = tf.matmul(fc7, fc8W) + fc8b
probs = tf.nn.softmax(logits)

saver = tf.train.Saver()

sess = tf.Session()

if tf.train.checkpoint_exists('./model.ckpt'):
    print('Restoring model.ckpt...')
    saver.restore(sess, './model.ckpt')
else:
    print('No model found')
    sess.run(tf.global_variables_initializer())

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))
