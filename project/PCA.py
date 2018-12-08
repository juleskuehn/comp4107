import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_helper import load_vectorized_data
import os

def display_PCA(trainingData, figure_name="PCA"):
    # unpack the word vector and the categories
    word_vectors = [np.array(word_vector) for word_vector, _ in trainingData]
    categories = [category for _, category in trainingData]

    # convert the categories into an index vector
    categories_set = list(set(categories))
    category_indices = [categories_set.index(category) for category in categories]

    pca = PCA(n_components=32)
    pca.fit(word_vectors)

    pca_list = pca.transform(word_vectors) # reduce the data to 2D for displaying purpose
    plt.scatter(pca_list[:, 0], pca_list[:, 1], c=category_indices, s=0.5)
    title = figure_name
    plt.title(title)
    plt.savefig(os.getcwd() + "/document/" + title + ".png")
    plt.clf() # clear current figure for the next plot
