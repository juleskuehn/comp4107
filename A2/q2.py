"""
35 inputs and 31 output neurons (see Fig.11). The network receives 35 Boolean values, which represents one character.
The logsig (Log Sigmoid) is chosen as the transfer function for both hidden and output layers.
softmax function on the output layer.

Part A

1. Run experiments with hidden neuron numbers in the range 5-25.
2. Plot a chart of recognition error against the number of hidden neurons.


- weights normally distributed around zero.

In the first step, the network is trained on the ideal data for zero decision errors (see Fig.13 (a)).
In the second step, the network is trained on noisy data as shown in Fig.10 for several passes (e.g.10 passes) for a proper performance goal (0.01 is used in the program).
In the final step, it is trained again on just ideal data for zero decision errors (see Fig.13 (b)). 

Part B

1. Confirm that Fig.13 is a reasonable representation of performance for the optimal number of hidden layer neurons chosen.
2. Plot a chart in support of (1)


Part C

1. Create testing data that has between 0 and 3 bits noise.
2. Confirm that you can produce the recognition accuracy shown in Fig. 14.
"""