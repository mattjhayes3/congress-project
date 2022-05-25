import numpy as np
import keras
import keras.layers as layers
from .common_fns import loadEmbeddings
from .model import SequenceModel 
import gensim.downloader as api

class LSTMWord2VecModel(SequenceModel):
    def __init__(self, instance_name=None):
        super().__init__(instance_name)

    def name(self):
        return 'lstm_word2vec' if not self.instance_name else f"lstm_word2vec_{self.instance_name}"

    # inside, save the trained model to the corresponding folder - might be needed in the future
    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):
        embedding_size = 300
        embedding_index = api.load('word2vec-google-news-300')
        dictionary_size = np.max(training_matrix) + 1
        print(f"dictionary_size =  {dictionary_size}")
        embedding_matrix = np.zeros((dictionary_size, embedding_size))
        hits = 0
        misses = 0
        for word, i in dictionary.items():
            if word == '*':
                word = '.'
            try:
                vector = embedding_index[word]
                embedding_matrix[i, :] = vector
                hits +=1
            except KeyError:
                misses += 1
        print("Converted %d words (%d misses)" % (hits, misses))

        training_matrix = training_matrix.toarray()
        print(f"shape training matrix {training_matrix.shape}")
        # print("example:", training_matrix[-1, :])
        validation_matrix = validation_matrix.toarray()

        # define the early stopping criteria
        es = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)
        self.model = keras.models.Sequential([layers.Embedding(dictionary_size, embedding_size, input_length= np.shape(training_matrix)[1], embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=True),
                                            # layers.Dropout(0.2),
                                            layers.LSTM(128),
                                            # layers.Dropout(0.2),
                                            keras.layers.Dense(2, activation='softmax'),
                                            ])

        # with sgd optimizer, the result was 0.74, i just replaced it with adam and got 0.88 - the highest performance so far
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                        metrics=['sparse_categorical_crossentropy', 'accuracy'])
        self.model.fit(training_matrix, training_labels, epochs=200, batch_size=128,
                    validation_data=(validation_matrix, validation_labels), callbacks=[es])

