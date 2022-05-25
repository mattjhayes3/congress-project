import numpy as np
from sklearn.naive_bayes import MultinomialNB
from .model import Model

class DemBaselineModel(Model):
    def name(self):
        return 'dem_baseline' if not self.instance_name else f"dem_baseline_{self.instance_name}"

    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):
        # print(f"type(training_matrix)={type(training_matrix)}, type(training_labels)={type(training_labels)}")
        # print(f"type(validation_matrix)={type(validation_matrix)}, type(validation_labels)={type(validation_labels)}")
        pass

    def predict(self, input_matrix, trained_model):
        return np.ones(input_matrix.shape[0])

    def is_baseline(self):
        return True