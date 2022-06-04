import numpy as np
import keras
import keras.layers as layers
# from common_fns import *
from .model import SequenceModel 
import tensorflow as tf
import os
import shutil

class CNN3AvgDropModel(SequenceModel):
    def __init__(self, embedding_size, drate, lsize, pretrained=None, trainable=False, instance_name=None):
        super().__init__(embedding_size, pretrained, trainable, instance_name)
        self.drate = drate
        self.lsize = lsize

    def name(self):
        trainable_part = 'trainable' if self.trainable else "not_trainable"
        glove_part = "" if not self.pretrained else f"_{self.pretrained}_{trainable_part}"
        dint = int(self.drate * 100)
        return f'cnn3_avg_drop_128_7_l{self.lsize}_s1_d4_{dint}_e{self.embedding_size}{glove_part}' if not self.instance_name else f"cnn3_avg_drop_128_7_l{self.lsize}_s1_d4_{dint}_e{self.embedding_size}{glove_part}_{self.instance_name}"

    # inside, save the trained model to the corresponding folder - might be needed in the future
    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):

        training_matrix = training_matrix.toarray()
        print(f"shape training matrix {training_matrix.shape}")
        print("example:", training_matrix[-1, :])
        dictionary_size = int(np.max(training_matrix) + 2)
        print(f"dictionary_size =  {dictionary_size}")

        validation_matrix = validation_matrix.toarray()

        embedding_layer = layers.Embedding(dictionary_size, self.embedding_size, input_length= np.shape(training_matrix)[1]) if not self.pretrained else layers.Embedding(dictionary_size, self.embedding_size, input_length=np.shape(training_matrix)[1], embeddings_initializer=keras.initializers.Constant(self.load_glove_embeddings(self.pretrained, training_matrix, dictionary)), trainable=self.trainable)

        # define the early stopping criteria
        es = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)
        self.model = keras.models.Sequential([embedding_layer,
                                            layers.Conv1D(self.lsize, 7, padding="valid", activation="relu", strides=1),
                                            layers.Dropout(self.drate),
                                            layers.Conv1D(self.lsize, 7, padding="valid", activation="relu", strides=1),
                                            layers.Dropout(self.drate),
                                            layers.Conv1D(self.lsize, 7, padding="valid", activation="relu", strides=1),
                                            # layers.Conv1D(128, 7, padding="valid", activation="relu", strides=1),
                                            layers.GlobalAveragePooling1D(),
                                            layers.Dropout(self.drate),
                                            layers.Dense(self.lsize, activation="relu"),
                                            layers.Dropout(self.drate),
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

