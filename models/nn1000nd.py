# author: Ulya Bayram
# Here are the used classification methods are defined as separate functions
# calling them simply will provide necessary information for different feature types' classification in parallel
# ulyabayram@gmail.com
import numpy as np
import keras
# from common_fns import *
from .model import Model
from tensorflow.keras import regularizers


class NN1000NDModel(Model):
    def use_gpu(self):
        return True
        
    def name(self):
        return 'nn1000nd' if not self.instance_name else f"nn1000nd_{self.instance_name}"

    # inside, save the trained model to the corresponding folder - might be needed in the future
    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):

        training_matrix = training_matrix.toarray()
        validation_matrix = validation_matrix.toarray()

        num_features = np.shape(training_matrix)[1]

        # define the early stopping criteria
        es = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=25, verbose=0, mode='auto', restore_best_weights=True)
        self.model = keras.models.Sequential([ # feed forward NN with a single hidden layer
                                            #keras.layers.Dense(20, input_dim=num_features, activation='tanh'),
                                            keras.layers.Dense(1000, input_dim=num_features, activation='tanh'),
                                            #keras.layers.Dense(2, input_dim=num_features, activation='softmax'),
                                            #keras.layers.Dropout(0.5),
                                            #keras.layers.Dense(20, activation='tanh'),
                                            #keras.layers.Dropout(0.98),
                                            keras.layers.Dense(2, activation='softmax')
                                            ])

        # with sgd optimizer, the result was 0.74, i just replaced it with adam and got 0.88 - the highest performance so far
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                        metrics=['sparse_categorical_crossentropy', 'accuracy'])
        self.model.fit(training_matrix, training_labels, epochs=200, batch_size=64,
                    validation_data=(validation_matrix, validation_labels), callbacks=[es])
