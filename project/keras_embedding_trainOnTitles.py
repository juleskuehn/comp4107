# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from data_helper import labelsDict

vocab_size = 71057
max_title_length = 26 # See plot of title lengths. Only tiny % above 26 words
actual_title_length = 96

labels = [line.rstrip() for line in open('labels.txt')]
labels = [labelsDict[label] for label in labels]
titles = [line.rstrip() for line in open('titles.txt')]

encoded_docs = [one_hot(d, vocab_size) for d in titles]
padded_docs = pad_sequences(encoded_docs, maxlen=max_title_length, padding='post')
print(padded_docs[:10])

trX, teX, trY, teY = train_test_split(padded_docs, array(labels), test_size=0.1)

model = Sequential()
model.add(Embedding(vocab_size, 16, input_length=max_title_length))
model.add(Flatten())
# model.add(Dense(300, activation='relu'))
model.add(Dropout(0.5))
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