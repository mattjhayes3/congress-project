import numpy as np
import keras
from tensorflow.keras import layers
# from common_fns import *
from .model import SequenceModel 
import os
import shutil
import tensorflow as tf

class LSTMBiDiModel(SequenceModel):
    def use_gpu(self):
        return True

    def __init__(self, instance_name=None):
        super().__init__(instance_name)
        self.embedding_size = 128

    def name(self):
        return 'lstm2_bidi_128' if not self.instance_name else f"lstm2_bidi_128_{self.instance_name}"

    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):

        training_matrix = training_matrix.toarray()
        print(f"shape training matrix {training_matrix.shape}")
        print("example:", training_matrix[-1, :])
        dictionary_size = int(np.max(training_matrix) + 2)
        print(f"dictionary_size =  {dictionary_size}")

        validation_matrix = validation_matrix.toarray()

        es = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)
        self.model = keras.models.Sequential([layers.Embedding(dictionary_size, self.embedding_size, input_length= np.shape(training_matrix)[1]),
                                            # layers.Dropout(0.5),
                                            # layers.LSTM(128, dropout=0.75, return_sequences=True),
                                            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
                                            layers.Bidirectional(layers.LSTM(128)),
                                            # layers.Dropout(0.5),
                                            # layers.Dropout(0.2),
                                            keras.layers.Dense(1, activation='sigmoid'),
                                            # keras.layers.Dense(2, activation='softmax'),
                                            ])

        logdir = f"./logs/{self.name()}"
        shutil.rmtree(logdir, ignore_errors=True)
        os.makedirs(logdir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                        metrics=['binary_crossentropy', 'accuracy'])
        self.model.fit(training_matrix, training_labels, epochs=200, batch_size=128,
                    validation_data=(validation_matrix, validation_labels), callbacks=[es, tensorboard_callback])

