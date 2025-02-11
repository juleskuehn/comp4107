import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
import random
import math


TARGET_MSE = 0.02


def f(x, y):  # x, y ∈ [-1 1]
    return math.cos(x + 6*0.35*y) + 2*0.35*x*y


def genCoords(n):
    coords = []
    step = 1 / ((n-1) / 2.0)
    x = -1
    for _ in range(n):
        coords.append(x)
        x += step
    return coords


def genGridData(n, f):
    # Function output is scaled to fall between 0 and 1
    coords = genCoords(n)
    dataXY = []
    for x in coords:
        for y in coords:
            dataXY.append([x, y])
    dataZ = [(f(x, y)+1) / 2 for x, y in dataXY]
    dataZ = np.array(dataZ).reshape(n*n, 1)
    return np.concatenate((dataXY, dataZ), axis=1)


def genRandData(n, f):
    # Function output is scaled to fall between 0 and 1
    xy = [(random.random()*2 - 1, random.random()*2 - 1) for _ in range(n)]
    z = [(f(x, y)+1) / 2 for x, y in xy]
    return np.concatenate((np.array(xy), np.array(z).reshape(n, 1)), axis=1)


def pltData(data, color=''):
    g = int(math.sqrt(len(data)))
    plt.contour(
        data[:, 0].reshape((g, g)),
        data[:, 1].reshape((g, g)),
        data[:, 2].reshape((g, g)),
        10,
        colors=color
    )


def trainTest(n_hidden=50, epochs=3000, lr=0.02):
    # Generate training and test data
    # 10x10 and 9x9 grids: x, y ∈ [-1 1]
    trainData = genGridData(10, f)
    testData = genGridData(9, f)

    # 2 input neurons (x, y), 1 output (z)
    n_input = 2
    n_output = 1
    display_step = 1000

    # X: input, Y: output
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    # Weights and biases
    W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
    W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))
    b1 = tf.Variable(tf.zeros([n_hidden]))
    b2 = tf.Variable(tf.zeros([n_output]))

    # Hidden layer with sigmoid activation function
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

    # Output with linear activation function
    hy = tf.matmul(L2, W2) + b2

    # Minimize MSE on training data
    cost = tf.reduce_mean((hy - Y)**2)
    optimizer = tf.train.RMSPropOptimizer(lr).minimize(cost)

    init = tf.global_variables_initializer()

    # Randomize the order of training data
    np.random.shuffle(trainData)
    x = trainData[:,:2]
    y = trainData[:,2].reshape(len(trainData), 1)

    # Keep track of costs so as to graph reduction in cost over epochs
    costs = []

    # At what epoch is the goal reached?
    goal_reached = -1

    # Early stopping
    maxFails = 10
    consecFails = 0
    break_point = -1
    grmse = -1
    bpmse = -1

    with tf.Session() as sess:

        sess.run(init)
        for step in range(epochs):
            _, c = sess.run([optimizer, cost], feed_dict = {X: x, Y: y})
            # Show progress
            if step % display_step == 0:
                print("Cost: ", c)
            costs.append(c)
            if c <= TARGET_MSE and goal_reached == -1:
                goal_reached = step
                pred = sess.run([hy], feed_dict = {X: testData[:,:2]})
                pred = [x[0] for x in pred[0]]
                grmse = np.square(pred - testData[:,2]).mean()
                print("Goal reached at step", step)

            # If MSE on validation data is increasing...
            if step > 1 and costs[-1] > costs[-2]:
                consecFails += 1
                if consecFails >= maxFails and break_point == -1:
                    break_point = step
                    print("Early stop at step", step)
                    pred = sess.run([hy], feed_dict = {X: testData[:,:2]})
                    pred = [x[0] for x in pred[0]]
                    bpmse = np.square(pred - testData[:,2]).mean()
            else:
                consecFails = 0
        
        # Get prediction for testData, for comparison to actual f(x, y)
        pred = sess.run([hy], feed_dict = {X: testData[:,:2]})

    pred = [x[0] for x in pred[0]]
    mse = np.square(pred - testData[:,2]).mean()

    # Reshape prediction into format of testData for comparison [[x, y, z]]
    pred = np.concatenate((testData[:,:2], np.array(pred).reshape(len(pred), 1)),axis=1)
    return pred, mse, costs, goal_reached, break_point, bpmse, grmse


# num_neurons is an array of numbers of neurons in hidden layer
def runExperiment(num_neurons, epochs, lr):
    results = []
    colors = ['red', 'green', 'blue', 'yellow', 'pink', 'purple', 'orange']

    # Run experiments and get results for each number of neurons
    for n in num_neurons:
        pred, mse, c, g, b, bpmse, grmse = trainTest(n, epochs, lr)
        results.append((pred, mse, c, g, b, bpmse, grmse))

    print("\nMSE with LR =", lr)
    for i in range(len(num_neurons)):
        print(f"{num_neurons[i]:3}", "neurons: ")
        print(f"At {epochs} epochs: {results[i][1]:>15.6f}")
        print(f"At goal reached {results[i][3]}: {results[i][6]:>15.6f}")
        print(f"At early stop {results[i][4]}: {results[i][5]:>15.6f}")

    # Re-generate testData, for reference
    testData = genGridData(9, f)

    # Legends
    legends = [mpatches.Patch(color='black', label='target')]
    for i in range(len(num_neurons)):
        legends.append(
          mpatches.Patch(
              color=colors[i] if i < len(colors) else 'grey',
              label=str(num_neurons[i])+' neurons'))
    plt.legend(handles=legends)

    pltData(testData, 'black')
    for r in results:
        pltData(r[0], colors.pop(0) if len(colors) > 0 else 'grey')
    plt.show()

    # Title offsets
    off = 0.7

    plt.figure(1)
    plt.subplot(311)
    plt.title('2 neurons', y=off)
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.plot(range(0, epochs), results[0][2], 'k')
    if results[0][3] != -1:
        plt.plot((results[0][3],results[0][3]),(0,100),'k--')
    if results[0][4] != -1:
        plt.plot((results[0][4],results[0][4]),(0,100),'r--')
    plt.ylim(0,0.06)

    plt.subplot(312)
    plt.title('8 neurons', y=off)
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.plot(range(0, epochs), results[1][2], 'k')
    if results[1][3] != -1:
        plt.plot((results[1][3],results[1][3]),(0,100),'k--')
    if results[1][4] != -1:
        plt.plot((results[1][4],results[1][4]),(0,100),'r--')
    plt.ylim(0,0.06)

    plt.subplot(313)
    plt.title('50 neurons', y=off)
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.plot(range(0, epochs), results[2][2], 'k')
    if results[2][3] != -1:
        plt.plot((results[2][3],results[2][3]),(0,100),'k--')
    if results[2][4] != -1:
        plt.plot((results[2][4],results[2][4]),(0,100),'r--')
    plt.ylim(0,0.06)
    plt.show()


runExperiment([2, 8, 50], 10000, 0.02)