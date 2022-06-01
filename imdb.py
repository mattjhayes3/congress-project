# LSTM with Dropout for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras.layers as layers
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 20000
# top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 128
# model = Sequential()
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# model.add(Dropout(0.2))
# model.add(LSTM(100))
# model.add(Dropout(0.2))
# model.add(Dense(50, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
model = Sequential([Embedding(top_words, embedding_vecor_length, input_length=max_review_length), 
                    layers.Dropout(0.5),
                    layers.Conv1D(128, 7, padding="valid", activation="relu"),
                    # layers.Conv1D(32, 3, padding="valid", activation="relu"),
                    # layers.MaxPooling1D(pool_size=2),
                    layers.GlobalMaxPooling1D(),
                    layers.Dense(128, activation="relu"),
                    layers.Dropout(0.5),
                    layers.Dense(1, activation='sigmoid')])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))