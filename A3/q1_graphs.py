import numpy as np
import matplotlib.pyplot as plt

random = {1: [0.51, 0.5, 0.5, 0.54, 0.5], 2: [0.5, 0.5, 0.5, 0.5, 0.5], 3: [0.5, 0.5, 0.5, 0.5, 0.5], 4: [0.5, 0.5, 0.5, 0.5, 0.5], 5: [0.5, 0.5, 0.5, 0.5, 0.5]}

centers = {1: [0.68, 0.65, 0.67, 0.66, 0.65], 2: [0.5, 0.5, 0.5, 0.5, 0.5], 3: [0.63, 0.5, 0.5, 0.5, 0.5], 4: [0.5, 0.5, 0.5, 0.5, 0.5], 5: [0.5, 0.5, 0.5, 0.5, 0.5]}

similar = {1: [0.65, 0.68, 0.66, 0.65, 0.67], 2: [0.65, 0.66, 0.73, 0.69, 0.65], 3: [0.5, 0.5, 0.5, 0.5, 0.5], 4: [0.5, 0.5, 0.5, 0.5, 0.5], 5: [0.5, 0.5, 0.5, 0.5, 0.5]}

def avgD(d):
    for key in d:
        d[key] = np.average(d[key])
    return d

plt.ylim(0.4, 1)
plt.ylabel('Classification accuracy')
plt.xticks([key for key in random])
plt.xlabel('Number of training images per digit')
l1, = plt.plot(*zip(*sorted(avgD(random).items())))
l2, = plt.plot(*zip(*sorted(avgD(centers).items())))
l3, = plt.plot(*zip(*sorted(avgD(similar).items())))
plt.legend((l1, l2, l3), ('Random images', 'Cluster centers', 'Most similar'))
plt.savefig('document/q1_noStorkey.png')
plt.close()

storky = {1: [0.5, 0.51, 0.51, 0.5, 0.71], 2: [0.51, 0.5, 0.5, 0.54, 0.5], 3: [0.5, 0.77, 0.5, 0.5, 0.64], 4: [0.5, 0.5, 0.5, 0.5, 0.5], 5: [0.5, 0.5, 0.5, 0.5, 0.5]}

plt.ylim(0.4, 1)
plt.ylabel('Classification accuracy')
plt.xticks([key for key in random])
plt.xlabel('Number of training images per digit')
l1, = plt.plot(*zip(*sorted(avgD(random).items())))
l2, = plt.plot(*zip(*sorted(avgD(storky).items())))
plt.legend((l1, l2), ('Normal Hopfield', 'With Storkey'))
plt.savefig('document/q1_withStorkey.png')
plt.close()