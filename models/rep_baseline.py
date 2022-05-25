import numpy as np
from sklearn.naive_bayes import MultinomialNB
from .model import Model


class RepBaselineModel(Model):
    def name(self):
        return 'rep_baseline' if not self.instance_name else f"rep_baseline_{self.instance_name}"

    # inside, save the trained model to the corresponding folder - might be needed in the future
    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):
        # print(f"type(training_matrix)={type(training_matrix)}, type(training_labels)={type(training_labels)}")
        # print(f"type(validation_matrix)={type(validation_matrix)}, type(validation_labels)={type(validation_labels)}")
        pass

    def predict(self, input_matrix, trained_model):
        return np.zeros(input_matrix.shape[0])

    def is_baseline(self):
        return True