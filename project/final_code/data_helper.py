
import re
import random
import numpy as np

# To show distribution of title lengths
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set(color_codes=True)


def load_vectorized_data(file_path="./book-dataset/vectorized_data.csv", vector_length=96):
    """
    Read the vectorized data instead of the orignal book dataset

    Return split titles and categories
    """
    raw_data = list(open(file_path, "r").readlines())

    data = []
    for row in raw_data:
        book_info = row[:-1].split(" ")  # remove the last newline character
        data.append([[int(index) for index in book_info[:vector_length]], " ".join(
            book_info[vector_length:])])

    return data

def get_lengths():
    books = load_vectorized_data()
    lengths = []
    for book in books:
        i = 0
        for char in book[0]:
            i += 1
            if char < 1:
                break
        lengths.append(i)

    sns.distplot(lengths);
    return lengths

def vectorize_vector(vector):
    """
    Vectorized given vector, for instance, the given input vector is ["a", "b", "a"], then this function will
    convert it into [[1, 0], [0, 1], [0, 1]], the purpose of this function is that if we keep the category as string, it's to feed to the neural network.
    """
    vector_set = list(set(vector))
    vectorized_array = []
    num_categories = len(vector_set)

    print(vector_set)
    for item in vector:
        i = vector_set.index(item)
        vectorized_array.append(
            [1 if j == i else 0 for j in range(num_categories)])

    return vectorized_array


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data(file_path="./book-dataset/book32-listing.csv"):
    """
    Load the book titles and the corresponding category of those books, split the book title into word vectors
    Each row in the orignal csv file looks like
    "[AMAZON INDEX (ASIN)}","[FILENAME]","[IMAGE URL]","[TITLE]","[AUTHOR]","[CATEGORY ID]","[CATEGORY]"
    Here we only need the title and category, so we only take those information

    Return split titles and categories
    """

    # some special character cannot be recognized correctly, so adding the error=ignore to ignore those errors
    # raw data containinng all information
    raw_data = list(open(file_path, "r", errors="ignore").readlines())

    data = []

    for row in raw_data:
        book_info = row.split('","')  # split the data
        # remove the last '"' symbol from category label
        data.append([clean_str(book_info[3]), book_info[-1][:-1]])

    return data


def separate_titles_labels(file_path="./book-dataset/book32-listing.csv"):
    """
    Load the book titles and the corresponding category of those books, split the book title into word vectors
    Each row in the orignal csv file looks like
    "[AMAZON INDEX (ASIN)}","[FILENAME]","[IMAGE URL]","[TITLE]","[AUTHOR]","[CATEGORY ID]","[CATEGORY]"
    Here we only need the title and category, so we only take those information

    Return split titles and categories
    """

    # some special character cannot be recognized correctly, so adding the error=ignore to ignore those errors
    # raw data containinng all information
    raw_data = open(file_path, "r", errors="ignore")

    labels = open('./book-dataset/labels.txt', 'w')
    titles = open('./book-dataset/titles.txt', 'w')

    for line in raw_data:
        book_info = line.split('","')  # split the data
        # remove the last '"' symbol from category label
        titles.write(book_info[3]+'\n')
        labels.write(book_info[-1][:-2]+'\n')

    raw_data.close()
    labels.close()
    titles.close()
    return


# Build a vocabulary index and map each word to an integer between 0 and 71056 (the vocabulary size).
# Each title becomes a vector of integers of the longest title dimension.


def vectorize_data(data):
    max_length = 0  # find the title with the largest length
    vocabulary_list = []
    for title, _ in data:
        word_list = title.split(" ")
        if len(word_list) > max_length:
            max_length = len(word_list)
        vocabulary_list.extend(title.split(" "))

    # create a set of the word so the index for each word in unique
    vocabulary_list = list(set(vocabulary_list))

    for i in range(len(data)):
        # create a vector of dimension max_length
        word_vector = np.array([0 for _ in range(max_length)])
        word_list = data[i][0].split(" ")
        for j in range(len(word_list)):
            word = word_list[j]

            # add 1 to the index, since 0 is left as the empty character(most titles have a smaller length, so all indexes after its maximum length are all 0)
            word_vector[j] = vocabulary_list.index(word.lower()) + 1

        data[i][0] = word_vector
    return data

def generate_vectorized_data():
    """
    Code for generating vectorized data, it will take > 1 hour for the whole data set,
    so we choose to run this code once and remember the results that have been genereted
    into a new csv file and start from there
    """
    data = load_data("./book-dataset/book32-listing.csv")
    vectorized_data = vectorize_data(data)
    with open('./book-dataset/vectorized_data.csv', 'w') as f:
        for row in vectorized_data:
            f.write("%s %s\n" % (" ".join(str(x)
                                          for x in row[0]), row[1][:-1]))


def vectorized_data_to_tsv(thin=1):
    """
    Transform to format expected by TensorFlow embedding projector
    """
    inFile = open('./book-dataset/vectorized_data.csv', 'r')
    outVectorFile = open('./book-dataset/title_vectors.tsv', 'w')
    outLabelFile = open('./book-dataset/labels.tsv', 'w')

    counter = 0
    for line in inFile:
        if counter % thin == 0:
            lineData = line.split()
            vector = ' '.join(lineData[:-1]) + '\n'
            label = lineData[-1] + '\n'
            outVectorFile.write(vector)
            outLabelFile.write(label)
        counter += 1

    inFile.close()
    outVectorFile.close()
    outLabelFile.close()

labelsDict = {
    'Science & Math': 0,
    'Engineering & Transportation': 1,
    'Christian Books & Bibles': 2,
    'Travel': 3,
    'Literature & Fiction': 4,
    'Sports & Outdoors': 5,
    'Computers & Technology': 6,
    'Parenting & Relationships': 7,
    'Religion & Spirituality': 8,
    'Self-Help': 9,
    "Children's Books": 10,
    'Biographies & Memoirs': 11,
    'Reference': 12,
    'Cookbooks, Food & Wine': 13,
    'Arts & Photography': 14,
    'Education & Teaching': 15,
    'Law': 16,
    'Comics & Graphic Novels': 17,
    'Science Fiction & Fantasy': 18,
    'Medical Books': 19,
    'Health, Fitness & Dieting': 20,
    'Gay & Lesbian': 21,
    'History': 22,
    'Calendars': 23,
    'Mystery, Thriller & Suspense': 24,
    'Politics & Social Sciences': 25,
    'Business & Money': 26,
    'Test Preparation': 27,
    'Humor & Entertainment': 28,
    'Teen & Young Adult': 29,
    'Crafts, Hobbies & Home': 30,
    'Romance': 31
    }
