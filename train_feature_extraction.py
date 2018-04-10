import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from get_data import get_data
get_data()
from alexnet import AlexNet

nb_classes = 43
epochs = 10
batch_size = 128
learning_rate = 0.001

# Load traffic sign training data
with open('./train.p', 'rb') as f:
    data = pickle.load(f)

# Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.33)

# Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(features, (227, 227))

fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.matmul(fc7, fc8W) + fc8b

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, var_list=[fc8W, fc8b])

preds = tf.arg_max(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

def evaluate(X, y, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        batch_loss, batch_acc = sess.run([loss, accuracy], feed_dict={features: X_batch, labels: y_batch})
        total_loss += (batch_loss * X_batch.shape[0])
        total_acc += (batch_acc * X_batch.shape[0])

    return total_loss/X.shape[0], total_acc/X.shape[0]

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Training on {} samples (Epochs: {}, Batch size: {}, Learning rate: {})...".format(len(X_train), epochs, batch_size, learning_rate))
    for i in range(epochs):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t_start = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            sess.run(optimizer, feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})

        val_loss, val_acc = evaluate(X_val, y_val, sess)
        t_total = time.time() - t_start
        print("Epoch {} ({:.3f} s) - Validation loss/accuracy: {}/{} ".format(i + 1, t_total, val_loss, val_acc))
    
    model_path = saver.save(sess, './model.ckpt')
    print("Model saved in:", model_path)

