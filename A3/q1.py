# COMP 4107
# Fall 2018
# Assignment 3
# Yunkai Wang, student number 100968473
# Jules Kuehn, student number 100661464

from q1_helpers import *


# calculate the weight based on the given input vectors using Hebbian's rule or
# Storkey's rule based on the choice
def cal_weight(data, use_storkey_rule=False):
    p = len(data)
    W = np.zeros((input_len, input_len))

    for pixels, _ in data:
        # code for the bonus mark, Storkey's rule of learning
        if use_storkey_rule:
            # these variables relate to the terms used in Storkey's learning rule
            local_field = W.dot(pixels.transpose())
            term1 = np.outer(local_field, pixels) / input_len
            term2 = np.outer(pixels, local_field) / input_len
            W -= np.add(term1, term2)

        W += (pixels.transpose()).dot(pixels)
    W -= np.dot(np.identity(input_len), p)
    return W


# feed the input vector to the network with the weight and threshold value
def test(weight, input):
    changed = True # a variable that indicates if there exist any node which changes its state
    
    vector = input[0]
    indices = list(range(0, len(vector))) # do it for every node in the network

    while changed:  # repeat until converge
        changed = False

        # array to store new state after this iteration
        new_vector = np.array(
            [0 for _ in range(len(vector))]
        )
        shuffle(indices) # use different order every time
        
        for index in indices:
            s = compute_sum(weight, vector, index)
            new_vector[index] = 1 if s >= 0 else -1 # new state for the node
        
        changed = not np.allclose(vector, new_vector)
        vector = new_vector

    return vector


# compute the sum by adding up the weights of all active edges that connects to the
# given node
def compute_sum(weight, vector, node_index):
    return sum([weight[node_index][index] for index in range(len(vector)) if vector[index] == 1])


# Among the training data, find the data that's closest to the stable state and
# pick the label that corresponds to that data as the label for the state
def classify(vector, data):
    closestDis = float('inf')
    closestLabel = None

    for pixels, label in data:
        dis = np.linalg.norm(vector - pixels)
        if dis < closestDis:
            closestDis = dis
            closestLabel = label

    # print("Output vector, classified as", str(closestLabel))
    # printVector(vector)

    return closestLabel


def test_network(num_training_data=5, num_testing_data=10, use_storkey_rule=False, reprs='random'):
    if reprs == 'all_centers':
        # pick representative training data (closest to kmeans centers)
        trainingData = getCenterOnesAndFives(num_training_data)
    elif reprs == 'most_similar':
        trainingData = getRepresentativeOnesAndFives(num_training_data)
    else:
        trainingData = getRandomOnesAndFives(num_training_data)
    W = cal_weight(trainingData, use_storkey_rule)
    testData = getRandomOnesAndFives(num_testing_data)
    correct = 0 # number of correct identified image

    for pixels, actual_label in testData:
        # print("Input digit, with actual label", str(actual_label))
        # printVector(pixels)

        vector = test(W, pixels)
        label = classify(vector, trainingData)
        if actual_label == label: # correctly identified one image
            correct += 1

    # (2 * num_testing_data) because num_testing_data is for each of 1 and 5
    return correct / (2 * num_testing_data) # calculate the accuracy


# It seems like feeding the network with 5 of each digit will cause the network
# to forget everything, even if the original training data is tested. If I gave
# only 1 image of each digit, the network will do a relatively good job.
def multi_trial(maxTrain, numTest, numTrials, representative, storkey):
    accuracies = {}

    for _ in range(numTrials):
        for numTrain in range(1, maxTrain + 1):
            accuracy = test_network(numTrain, numTest, use_storkey_rule=storkey, reprs=representative)
            print("number of training data for each digit:", numTrain)
            print("number of test data for each digit:", numTest)
            print("Storkey:", storkey, "; Representative training data:", representative)
            print("accuracy:", accuracy)
            if numTrain in accuracies:
                accuracies[numTrain].append(accuracy)
            else:
                accuracies[numTrain] = [accuracy]
            print("---")

    # for numTrain in accuracies:
    #     accuracies[numTrain] = np.average(accuracies[numTrain])

    print(accuracies)
    return accuracies


maxTrain = 5
numTest = 50
numTrials = 5



f = open('q1_results.txt', 'w')
f.write(f'maxTrain: {maxTrain}, numTest: {numTest}, numTrials: {numTrials}\n')
f.write('Random selection of training data, without Storkey:')
results = multi_trial(maxTrain, numTest, numTrials, 'random', False)
f.write(f'{results}\n')
f.write('K-means centers as training data, without Storkey:')
results = multi_trial(maxTrain, numTest, numTrials, 'all_centers', False)
f.write(f'{results}\n')
f.write('Most similar as training data, without Storkey:')
results = multi_trial(maxTrain, numTest, numTrials, 'most_similar', False)
f.write(f'{results}\n')
f.write('Random selection of training data, with Storkey:')
results = multi_trial(maxTrain, numTest, numTrials, 'random', True)
f.write(f'{results}\n')
f.write('K-means centers as training data, with Storkey:')
results = multi_trial(maxTrain, numTest, numTrials, 'all_centers', True)
f.write(f'{results}\n')
f.write('Most similar as training data, with Storkey:')
results = multi_trial(maxTrain, numTest, numTrials, 'most_similar', True)
f.write(f'{results}\n')
f.close()

# plt.ylim(0.4, 1)
# plt.xticks(range(maxTrain + 1))
# plt.plot(*zip(*sorted(results.items())))
# plt.show()


# ----------------------------------------------------------------
# code for testing the network on the small example given in class
# input_len = 4 # testing small images

# # data found on slide 59 of Hopfield network
# x1 = np.array([1, -1, -1, 1])
# x2 = np.array([1, 1, -1, 1])
# x3 = np.array([-1, 1, 1, -1])
# # testing on the small example to find the problem
# trainingData = [[x1, 1], [x2, 2], [x3, 3]]
# W = cal_weight(trainingData, threshold=0)
# print(W)
# for pixels, actual_label in trainingData:
#     print("Start to test on pixels: ", pixels)
#     vector = test(W, pixels, threshold=0)
#     label = classify(vector, trainingData)
#     print(actual_label, label)