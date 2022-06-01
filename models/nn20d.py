# original author: Ulya Bayram, adapted by: Matthew Hayes
# Here are the used classification methods are defined as separate functions
# calling them simply will provide necessary information for different feature types' classification in parallel
# ulyabayram@gmail.com, mattjhayes3@gmail.com
import numpy as np
import keras
# from common_fns import *
from .model import Model
import tensorflow as tf
import os
import shutil

class NN20DModel(Model):
    def name(self):
        return 'nn20d' if not self.instance_name else f"nn20d_{self.instance_name}"

    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):

        training_matrix = training_matrix.toarray()
        validation_matrix = validation_matrix.toarray()

        num_features = np.shape(training_matrix)[1]

        es = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=25, verbose=0, mode='auto', restore_best_weights=True)
        self.model = keras.models.Sequential([ # feed forward NN with a single hidden layer
                                            keras.layers.Dense(20, input_dim=num_features, activation='tanh'),
                                            #keras.layers.Dense(1000, input_dim=num_features, activation='tanh'),
                                            #keras.layers.Dense(2, input_dim=num_features, activation='softmax'),
                                            #keras.layers.Dropout(0.5),
                                            #keras.layers.Dense(20, activation='tanh'),
                                            keras.layers.Dropout(0.98),
                                            keras.layers.Dense(2, activation='softmax')
                                            ])

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                        metrics=['sparse_categorical_crossentropy', 'accuracy'])
        logdir = f"./logs/{self.name()}"
        shutil.rmtree(logdir, ignore_errors=True)
        os.makedirs(logdir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        self.model.fit(training_matrix, training_labels, epochs=200, batch_size=10,
                    validation_data=(validation_matrix, validation_labels), callbacks=[es, tensorboard_callback])

