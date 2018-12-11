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
import matplotlib.pyplot as plt
from data_helper import labelsDict

# Download glove dataset if necessary:
# https://www.dropbox.com/s/wy8dfpz6ox66h8v/glove.6B.100d.txt?dl=1

max_title_length = 26 # See plot of title lengths. Only tiny % above 26 words
actual_title_length = 96

labels = [line.rstrip() for line in open('labels.txt')]
labels = [labelsDict[label] for label in labels]
titles = [line.rstrip() for line in open('titles.txt')]

# Prepare tokenizer
t = Tokenizer()
t.fit_on_texts(titles)
vocab_size = len(t.word_index) + 1

# Integer encode the documents
encoded_docs = t.texts_to_sequences(titles)
print(encoded_docs[:10])

# Pad documents to a max length
max_length = max_title_length
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs[:10])

# Load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt', encoding="utf-8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# Create pre-trained weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

trX, teX, trY, teY = train_test_split(padded_docs, array(labels), test_size=0.1)

model = Sequential()
model.add(Embedding(vocab_size, 100, weights=[embedding_matrix],
                input_length=max_title_length, trainable=False))
model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(300, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(32, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy'])

print(model.summary())

history = model.fit(trX, to_categorical(trY), epochs=15, verbose=1,
          validation_data=(teX, to_categorical(teY)))

loss, accuracy = model.evaluate(teX, to_categorical(teY), verbose=1)
print('Accuracy: %f' % (accuracy*100))

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
