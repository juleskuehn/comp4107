# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam

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


vocab_size = 71057
max_title_length = 26 # See plot of title lengths. Only tiny % above 26 words
actual_title_length = 96

labels = open('labels.txt').readlines()
labels = [labelsDict[label] for label in labels]
titles = open('titles.txt').readlines()

trX, teX, trY, teY = train_test_split(array(titles), array(labels), test_size=0.1)

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(titles)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(titles)
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = max_title_length
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt', encoding="utf-8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
# define model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))