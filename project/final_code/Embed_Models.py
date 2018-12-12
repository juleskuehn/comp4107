# Code based on
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

from numpy import array
from numpy import asarray
from numpy import zeros
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import metrics
from data_helper import labelsDict
import random
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import urllib

"""
Load the data
"""

try:
    labelsFile = open('labels.txt')
except:
    print("File not found locally, downloading...")
    urllib.request.urlretrieve("https://www.dropbox.com/s/mjk1mum1ycj1rwe/labels.txt?dl=1", "labels.txt")
    labelsFile = open('labels.txt')

try:
    titlesFile = open('titles.txt')
except:
    print("File not found locally, downloading...")
    urllib.request.urlretrieve("https://www.dropbox.com/s/cd3lpllzzo935an/titles.txt?dl=1", "titles.txt")
    titlesFile = open('titles.txt')

labels = [labelsDict[line.rstrip()] for line in labelsFile]
titles = [line.rstrip() for line in titlesFile]
labelsFile.close()
titlesFile.close()

"""
Generate and train the model
"""

max_title_length = 26  # See plot of title lengths. Only tiny % above 26 words

# Do train-test split before encoding, so that we can get title strings later
trsX, tesX, trY, teY = train_test_split(
    array(titles), array(labels), test_size=0.1)

# Tokenizer is used simply to get vocabulary size
t = Tokenizer()
t.fit_on_texts(trsX)
vocab_size = len(t.word_index) + 1

# Keras helpers to convert titles to sequence of integers and pad to same length
trX = [one_hot(d, vocab_size) for d in trsX]
teX = [one_hot(d, vocab_size) for d in tesX]
trX = pad_sequences(trX, maxlen=max_title_length,
                    padding='post', truncating='post')
teX = pad_sequences(teX, maxlen=max_title_length,
                    padding='post', truncating='post')

model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_title_length))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(32, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy', metrics.top_k_categorical_accuracy])

print(model.summary())

history = model.fit(trX, to_categorical(trY), epochs=5, verbose=1, batch_size=32,
                    validation_data=(teX, to_categorical(teY)))

"""
Views into the data
"""

labels = [0 for _ in range(32)]
for label in labelsDict:
    labels[labelsDict[label]] = label


def show_random_top_5():
    # Show top 5 categories for a random title in the test set
    i = random.randint(0, len(teX) - 1)
    class_probs = model.predict(teX[i].reshape(1, 26))
    print(tesX[i])
    print(labels[teY[i]])
    print("-------------------------------------")
    results = {}
    for i, pred in enumerate(class_probs[0]):
        results[labels[i]] = pred
    for w in sorted(results, key=results.get, reverse=True)[:5]:
        print(f"{w:30} {results[w]:6.4f}")
    print('\n')


def pred_from_string(s):
    # Test the model with an arbitrary title
    sBin = [one_hot(s, vocab_size)]
    sBin = pad_sequences(sBin, maxlen=max_title_length,
                         padding='post', truncating='post')
    class_probs = model.predict(sBin)
    results = {}
    for i, pred in enumerate(class_probs[0]):
        results[labels[i]] = pred
    for w in sorted(results, key=results.get, reverse=True)[:5]:
        print(f"{w:30} {results[w]:6.4f}")
    print('\n')


def test_string_loop():
    while True:
        pred_from_string(input("Enter a title: "))


def get_test_indices_by_category():
    test_indices_by_category = [[] for _ in range(32)]
    for i in range(len(teY)):
        test_indices_by_category[teY[i]].append(i)
    return test_indices_by_category


def category_accuracy():
    # Find top 1 and top 5 for each category
    tic = get_test_indices_by_category()
    category_top1_top5 = {}

    for catNum in range(32):
        # One-hot categorical
        testLabels = np.array([[1 if i == catNum else 0 for i in range(32) ] for _ in range(len(tic[catNum]))])
        _, accuracy, topk = model.evaluate(teX[tic[catNum]], testLabels, verbose=0)
        category_top1_top5[labels[catNum]] = [accuracy, topk]
        
    return category_top1_top5


def print_category_accuracy():
    category_top1_top5 = category_accuracy()
    print(f'{"Category":40}{"Top 1":>10}{"Top 5":>10}')
    print('-'*60)
    for w in sorted(category_top1_top5, key=category_top1_top5.get, reverse=True):
        print(f"{w:40}{category_top1_top5[w][0]:10.4f}{category_top1_top5[w][1]:10.4f}")


def plot_category_accuracy():
    # Plot accuracy per category
    category_top1_top5 = category_accuracy()
    n_groups = 32

    top1 = [category_top1_top5[label][0] for label in sorted(
        category_top1_top5, key=category_top1_top5.get, reverse=True)]
    top5 = [category_top1_top5[label][1] for label in sorted(
        category_top1_top5, key=category_top1_top5.get, reverse=True)]

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    rects1 = ax.bar(index, top1, bar_width,
                    alpha=opacity, color='b',
                    label='Top 1')
    rects2 = ax.bar(index + bar_width, top5, bar_width,
                    alpha=opacity, color='r',
                    label='Top 5')
    ax.set_xlabel('Book Category')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Top 1 and Top 5 accuracy by category')
    ax.set_xticks(list(range(32)))
    ax.set_xticklabels(list(range(32)))
    ax.legend()
    fig.tight_layout()
    plt.show()


def get_categorization_matrix():
    # Average of predictions for all test data in each category
    # Returns 32 x 32 matrix of Input category -> Predicted category
    tic = get_test_indices_by_category()
    probs = []
    for catNum in range(32):
        probs.append(model.predict(teX[tic[catNum]]))

    for catNum in range(32):
        probs[catNum] = np.average(probs[catNum], axis=0)
        
    return probs


def plot_title_lengths():
    # Plot distribution of title lengths, and accuracy vs length of title

    class_probs = model.predict(teX)
    correct = [1 if np.argmax(prob) == teY[i] else 0 for i, prob in enumerate(class_probs)]

    # Length of each title
    lengths = [np.count_nonzero(vec) for vec in teX]

    correctByLength = [[] for _ in range(26)]
    for i, length in enumerate(lengths):
        correctByLength[length-1].append(correct[i])
    
    numBooksPerLength = [len(c) for c in correctByLength]
    correctByLength = [np.average(c) for c in correctByLength]

    plt.title('Distribution of title lengths')
    plt.ylabel('Number of books')
    plt.xlabel('Title length (words)')
    plt.plot(numBooksPerLength)
    plt.show()
    
    plt.title('Classification accuracy vs title length')
    plt.ylabel('Accuracy')
    plt.xlabel('Title length (words)')
    plt.plot(correctByLength)
    plt.show()


def plot_accuracy_per_epoch():
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylim(0, 1)
    plt.xticks([epoch for epoch in range(0, len(history.history['acc'])) if epoch % 1 == 0])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_category_means_tsne():
    probs = get_categorization_matrix()
    perplexity = 2

    tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=6000, random_state=2157)
    low_dim_embeds = tsne.fit_transform(probs)

    title = "t-SNE of category means (after training)"
    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = [10,12]
    # plt.rcParams['axes.facecolor'] = 'white'
    plt.grid(c='white')
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])

    ax.scatter(low_dim_embeds[:, 0], low_dim_embeds[:, 1], s=100, c="yellow")

    for i, txt in enumerate(labels):
        ax.annotate(txt, (low_dim_embeds[:, 0][i], low_dim_embeds[:, 1][i]), ha='center')
    plt.title(title)
    plt.show()


