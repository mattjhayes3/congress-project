import numpy as np
import keras
# from common_fns import *
from .model import Model
import tensorflow as tf
import os
import shutil
#  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ as the ? elected chairman of the democratic congressional campaign ? * i rise to inform my leadership and my democratic colleagues that i shall never ? my tenure put their name to any document they ? not seen * i understand that president reagan has ? the republican ? letter dated january 21 * 1981 ? bears his signature * i believe the president and i can well imagine his embarrassment and his ? for ? is to blame for ? of his signature * i would also like to address my republican colleagues * by definition we are all politicians and we all know that the practice of politics can be tough * but there are also ? that should not be ? * after all we must be able to put our differences aside in order to serve the people we represent * you have my pledge that while we will contest you with ? and ? it will be done ? * i trust we can put this ? incident behind us and move on to address the problems facing this country *

#  _ _ that is right * all it does * i will tell my colleague is request the department of defense to notify the committees of congress when this is going to happen *

class NNMultiModel(Model):
    def name(self):
        return 'nnmulti2_tanh' if not self.instance_name else f"nnmulti2_tanh_{self.instance_name}"

    # def use_gpu(self):
    #     return True

    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):

        training_matrix = training_matrix.toarray()
        validation_matrix = validation_matrix.toarray()

        num_features = np.shape(training_matrix)[1]

        es = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15, verbose=0, mode='auto', restore_best_weights=True)
        self.model = keras.models.Sequential([
                                            keras.layers.Dense(2048, input_dim=num_features, activation='tanh'),
                                            # keras.layers.Dropout(0.15),
                                            keras.layers.Dropout(0.5),
                                            keras.layers.Dense(256, input_dim=num_features, activation='tanh'),
                                            # keras.layers.Dropout(0.15),
                                            keras.layers.Dropout(0.5),
                                            # keras.layers.Dense(40, input_dim=num_features, activation='relu'),
                                            # keras.layers.Dropout(0.15),
                                            # keras.layers.Dropout(0.5),
                                            # keras.layers.Dense(8, input_dim=num_features, activation='relu'),
                                            # keras.layers.Dropout(0.15),
                                            # keras.layers.Dropout(0.5),
                                            keras.layers.Dense(1, activation='sigmoid')
                                            ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                        metrics=['binary_crossentropy', 'accuracy'])
        logdir = f"./logs/{self.name()}"
        shutil.rmtree(logdir, ignore_errors=True)
        os.makedirs(logdir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        self.model.fit(training_matrix, training_labels, epochs=200, batch_size=64,
                    validation_data=(validation_matrix, validation_labels), callbacks=[es, tensorboard_callback])

