batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

  
def model(X, w, w_2, w_3, w_4, w_fc, w_o, p_keep_conv, p_keep_hidden):
  
    print(X.shape)
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    print(l1a.shape)
    l2a = tf.nn.relu(tf.nn.conv2d(l1a, w_2, strides=[1, 1, 1, 1], padding='SAME'))
    print(l2a.shape)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(l2.shape)
    
    l2 = tf.nn.dropout(l2, p_keep_conv)
    
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w_3, strides=[1, 1, 1, 1], padding='SAME'))
    print(l3a.shape)
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(l3.shape)
    l4a = tf.nn.relu(tf.nn.conv2d(l3, w_4, strides=[1, 1, 1, 1], padding='SAME'))
    print(l4a.shape)
    l4 = tf.nn.max_pool(l4a, ksize=[1, 24, 24, 1], strides=[1, 24, 24, 1], padding='SAME')
    print(l4.shape)
    
    l4 = tf.nn.dropout(l4, p_keep_conv)

    # Transform from CNN to feed forward style
    l5 = tf.reshape(l4, [-1, w_o.get_shape().as_list()[0]])    # reshape to (?, 16x16x32)
#     l6 = tf.nn.relu(tf.matmul(l5, w_fc))
    
#     l6 = tf.nn.dropout(l6, p_keep_hidden)
    
    pyx = tf.matmul(l5, w_o)
    return pyx

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
# teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
trainingData = load_vectorized_data()
word_vectors = [np.array(word_vector) for word_vector, _ in trainingData]
categories = [category for _, category in trainingData]
categories = vectorize_vector(categories)
trX, teX, trY, teY = train_test_split(word_vectors, categories, test_size=0.1)

# print(teY)

trX = np.array(trX).reshape(-1, 96, 1, 1) / 71056
teX = np.array(teX).reshape(-1, 96, 1, 1) / 71056
trY = np.array(trY)
teY = np.array(teY)

# X = tf.placeholder("float", [None, 28, 28, 1])
# Y = tf.placeholder("float", [None, 10])
X = tf.placeholder("float", [None, 96, 1, 1])
Y = tf.placeholder("float", [None, 32])

# w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
# w_fc = init_weights([32 * 14 * 14, 625]) # FC 32 * 14 * 14 inputs, 625 outputs
# w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)
# w = init_weights([3, 1, 1, 64]) 
# w_fc = init_weights([64 * 48, 1024])
# w_o = init_weights([1024, 32])      

# [filter_height, filter_width, in_channels, out_channels]
w = init_weights([3, 1, 1, 32])   
w_2 = init_weights([3, 1, 32, 32])
w_3 = init_weights([3, 1, 32, 64])  
w_4 = init_weights([3, 1, 64, 64])   

w_fc = init_weights([64 * 12, 1024])
w_o = init_weights([64 , 32])         # 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w_2, w_3, w_4, w_fc, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
# train_op = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=1e-6).minimize(cost)
train_op = tf.train.AdamOptimizer().minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Store accuracies per epoch for graphing
accuracies = []

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

#         test_indices = np.arange(len(teX)) # Get A Test Batch
#         np.random.shuffle(test_indices)
#         test_indices = test_indices[0:test_size]

        accuracy = np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX,
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0}))
        print('epoch', i+1, 'accuracy', accuracy)
        accuracies.append(accuracy)
