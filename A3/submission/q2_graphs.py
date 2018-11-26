import matplotlib.pyplot as plt

# 1000 samples, 100 epochs, lr=0.05
centers = {
    10: 71.02465320827038,
    40: 82.60398837033368,
    80: 84.31194848736659,
    160: 86.4130433253927,
    320: 84.01040230348342,
    640: 42.5716640874277
}

# 160 centers, 1000 samples, and 100 epochs, lr=0.05
dropout1 = {
    0: 86.49364913138595,
    0.1: 86.49095127939911,
    0.2: 84.80501058292278,
    0.4: 83.68496464162325,
    0.6: 80.80279256419999,
    0.8: 77.10251759336823
}

# 320 centers, 5000 samples, and 500 epochs, lr=0.01:
dropout2 = {
    0: 92.25957215181523,
    0.1: 92.41908419990857,
    0.2: 91.77876615314769,
    0.4: 90.65975656015401,
    0.6: 88.81949583464971,
    0.8: 85.61887915837953
}

plt.ylim(70, 100)
plt.ylabel('Classification Accuracy')
plt.xticks([key for key in centers])
plt.xlabel('Number of Centers')
plt.plot(*zip(*sorted(centers.items())))
plt.savefig('document/q2_centers.png')
plt.close()

plt.ylim(70, 100)
plt.ylabel('Classification Accuracy')
plt.xticks([key for key in dropout1])
plt.xlabel('Dropout level (0 = no dropout)')
plt.plot(*zip(*sorted(dropout1.items())))
plt.savefig('document/q2_dropout1.png')
plt.close()

plt.ylim(70, 100)
plt.ylabel('Classification Accuracy')
plt.xticks([key for key in dropout2])
plt.xlabel('Dropout level (0 = no dropout)')
plt.plot(*zip(*sorted(dropout2.items())))
plt.savefig('document/q2_dropout2.png')