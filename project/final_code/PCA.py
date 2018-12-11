import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def display_PCA(trainingData, figure_name="PCA"):
    # unpack the word vector and the categories
    word_vectors = [np.array(word_vector) for word_vector, _ in trainingData]
    categories = [category for _, category in trainingData]

    pca = PCA(n_components=32)
    pca.fit(word_vectors)

    pca_list = pca.transform(word_vectors) # reduce the data to 2D for displaying purpose
    plt.scatter(pca_list[:, 0], pca_list[:, 1], c=categories, s=0.5)
    title = figure_name
    plt.title(title)
    plt.savefig(os.getcwd() + "/document/" + title + ".png")
    plt.clf() # clear current figure for the next plot
