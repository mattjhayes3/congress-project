# author: Ulya Bayram
# Here are the used classification methods are defined as separate functions
# calling them simply will provide necessary information for different feature types' classification in parallel
# ulyabayram@gmail.com
import numpy as np
import keras
from .model import Model


class LogisticSigModel(Model):
    def name(self):
        return 'logistic_sig' if not self.instance_name else f"logistic_sig_{self.instance_name}"


    # inside, save the trained model to the corresponding folder - might be needed in the future self,
    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):
        # print(f"type(training_matrix)={type(training_matrix)}, type(training_labels)={type(training_labels)}")
        # print(f"type(validation_matrix)={type(validation_matrix)}, type(validation_labels)={type(validation_labels)}")
        training_matrix = training_matrix.toarray()
        validation_matrix = validation_matrix.toarray()

        num_features = np.shape(training_matrix)[1]

        # define the early stopping criteria
        es = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=100, verbose=0, mode='auto', restore_best_weights=True) # 100
        # with tf.Session() as sess:
        # self.
        self.model = keras.models.Sequential([ # feed forward NN with a single hidden layer
                                        #keras.layers.Dense(20, input_dim=num_features, activation='tanh'),
                                        #keras.layers.Dense(1000, input_dim=num_features, activation='tanh'),
                                        keras.layers.Dense(1, input_dim=num_features, activation='sigmoid'),
                                        #keras.layers.Dropout(0.5),
                                        #keras.layers.Dense(20, activation='tanh'),
                                        #keras.layers.Dropout(0.98),
                                        #keras.layers.Dense(2, activation='softmax')
                                        ])

        # with sgd optimizer, the result was 0.74, i just replaced it with adam and got 0.88 - the highest performance so far
        # self.
        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                            metrics=['binary_crossentropy', 'accuracy'])
        # self.
        self.model.fit(training_matrix, training_labels, epochs=200, batch_size=10,
                        validation_data=(validation_matrix, validation_labels), callbacks=[es])
