import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .model import Model

class LDAModel(Model):
    def name(self):
        return 'lda' if not self.instance_name else f"lda_{self.instance_name}"

    # inside, save the trained model to the corresponding folder - might be needed in the future
    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):
        # print(f"type(training_matrix)={type(training_matrix)}, type(training_labels)={type(training_labels)}")
        # print(f"type(validation_matrix)={type(validation_matrix)}, type(validation_labels)={type(validation_labels)}")
        training_matrix = training_matrix.toarray()
        validation_matrix = validation_matrix.toarray()

        num_features = np.shape(training_matrix)[1]
        # self.model = LinearDiscriminantAnalysis(solver='lsqr')
        self.model = LinearDiscriminantAnalysis(solver='svd')
        self.model.fit(training_matrix, training_labels)
