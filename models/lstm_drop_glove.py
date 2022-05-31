import numpy as np
import keras
import keras.layers as layers
from .common_fns import loadEmbeddings
from .model import SequenceModel 
import tensorflow as tf
import os
import shutil

class LSTMDropGloveModel(SequenceModel):
    def __init__(self, embedding_size, instance_name=None):
        super().__init__(instance_name)
        self.embedding_size = embedding_size

    def name(self):
        return f'lstm_drop_glove_32_{self.embedding_size}' if not self.instance_name else f"lstm_drop_glove_32_{self.embedding_size}_{self.instance_name}"

    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):
        embedding_index = loadEmbeddings(f"../glove.6B/glove.6B.{self.embedding_size}d.txt")
        dictionary_size = np.max(training_matrix) + 1
        print(f"dictionary_size =  {dictionary_size}")
        embedding_matrix = np.zeros((dictionary_size, self.embedding_size))
        hits = 0
        misses = 0
        for word, i in dictionary.items():
            if word == '*':
                word = '.'
            vector = embedding_index.get(word)
            if vector is None:
                misses += 1
            else:
                hits += 1
                embedding_matrix[i, :] = vector
        print("Converted %d words (%d misses)" % (hits, misses))

        training_matrix = training_matrix.toarray()
        print(f"shape training matrix {training_matrix.shape}")
        # print("example:", training_matrix[-1, :])
        validation_matrix = validation_matrix.toarray()

        es = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)
        self.model = keras.models.Sequential([layers.Embedding(dictionary_size, self.embedding_size, input_length= np.shape(training_matrix)[1], embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=True),
                                            # layers.Dropout(0.2),
                                            layers.Dropout(0.5),
                                            layers.LSTM(32, dropout=0.5),
                                            layers.Dropout(0.5),
                                            # layers.Dropout(0.2),
                                            keras.layers.Dense(1, activation='sigmoid'),
                                            ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                        metrics=['binary_crossentropy', 'accuracy'])
        logdir = f"./logs/{self.name()}"
        shutil.rmtree(logdir, ignore_errors=True)
        os.makedirs(logdir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        self.model.fit(training_matrix, training_labels, epochs=200, batch_size=128,
                    validation_data=(validation_matrix, validation_labels), callbacks=[es, tensorboard_callback])
