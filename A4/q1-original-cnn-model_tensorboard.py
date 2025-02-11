# COMP 4107
# Fall 2018
# Assignment 4
# Yunkai Wang, student number 100968473
# Jules Kuehn, student number 100661464

"""
1. Modification of the dataset loaded to classify the CIFAR-10, 10 class image data.
2. Changing the number of convolutional layers and sizes of max pooling layers. You must investigate 5 different model scenarios.
3. Provide a PDF image of your computational graph. This must be captured from TensorBoard.
4. Provide a capability to view weight histograms using TensorBoard. You must be able to checkpoint your model during training. See tutorial for details.
5. Provide a chart of the accuracy of your network for 1-15 epochs for the scenarios investigated.
6. Provide the capability to show the top 9 patches. Examples are shown on slides 63-65 of the CNN notes.
"""

import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 128
test_size = 256

def init_weights(shape, name="W"):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


"""
Part 1: Load CIFAR-10
"""
from keras.datasets import cifar10
(trX, trY), (teX, teY) = cifar10.load_data()
trX = trX / 255.0
teX = teX / 255.0

# Transform labels to one bit hot
one_bit_labels = []
for label in trY:
    one_bit_labels.append([1 if j == label[0] else 0 for j in range(10)])
trY = np.array(one_bit_labels[:])

one_bit_labels = []
for label in teY:
    one_bit_labels.append([1 if j == label[0] else 0 for j in range(10)])
teY = np.array(one_bit_labels)

def model():
    with tf.name_scope("convolutional_layer"):
        with tf.name_scope("Conv2D_3x3"):
            w = init_weights([3, 3, 3, 32])
            tf.summary.histogram("w", w)
            l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
        with tf.name_scope("MaxPool_2x2"):
            l2 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.name_scope("Dropout_Conv"):
            l2 = tf.nn.dropout(l2, p_keep_conv)
    
    with tf.name_scope("fully_connected_layer"):
        with tf.name_scope("FC"):
            w_fc = init_weights([32 * 16 * 16, 625])
            tf.summary.histogram("w_fc", w_fc)

            # Transform from CNN to feed forward style
            l5 = tf.reshape(l2, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?, 16x16x32)
            l6 = tf.nn.relu(tf.matmul(l5, w_fc))
        
        with tf.name_scope("Dropout_FC"):
            l6 = tf.nn.dropout(l6, p_keep_hidden)

    
    with tf.name_scope("Output"):
        w_o = init_weights([625, 10])
        tf.summary.histogram("w_o", w_o)
        pyx = tf.matmul(l6, w_o)
    return pyx

X = tf.placeholder("float", [None, 32, 32, 3], name="x")
Y = tf.placeholder("float", [None, 10], name="image_labels")

with tf.name_scope('dropout'):
    p_keep_conv = tf.placeholder("float", name="p_keep_conv")
    p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")

py_x = model()

# Note, this is the same as original MLP.
with tf.name_scope("cost"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

    cost_sum = tf.summary.scalar('cost', cross_entropy)

predict_op = tf.argmax(py_x, 1)

merged_summary = tf.summary.merge_all()

# Store accuracies per epoch for graphing
accuracies = []
saver = tf.train.Saver()
# Launch the graph in a session
with tf.Session() as sess:

    # Python class that writes data for TensorBoard
    writer = tf.summary.FileWriter("./tensorboard/original_cnn/", sess.graph)

    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(50):
        # randomize order of training data
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))

        for j, (start, end) in enumerate(training_batch):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})
            if j % 10 == 0:
                # beholder.update(session=sess)
                s = sess.run(merged_summary, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5})
                writer.add_summary(s, i)
        
        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        
        s = sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                        p_keep_conv: 1.0,
                                                        p_keep_hidden: 1.0})
        
        accuracy = np.mean(np.argmax(teY[test_indices], axis=1) == s)
            
        print('epoch', i+1, 'accuracy', accuracy)
        accuracies.append(accuracy)

        # save the model and data
        saver.save(sess, "session_data/small_model_session.ckpt")