import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .model import Model
from .common_fns import applyLogNormalizationNoUnit

class LDAModel(Model):
    def name(self):
        return 'lda_noscale_lsqr' if not self.instance_name else f"lda_noscale_lsqr_{self.instance_name}"

    def preprocess(self, matrix):
        return applyLogNormalizationNoUnit(matrix)

    # inside, save the trained model to the corresponding folder - might be needed in the future
    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):
        # print(f"type(training_matrix)={type(training_matrix)}, type(training_labels)={type(training_labels)}")
        # print(f"type(validation_matrix)={type(validation_matrix)}, type(validation_labels)={type(validation_labels)}")
        training_matrix = training_matrix.toarray()
        validation_matrix = validation_matrix.toarray()

        num_features = np.shape(training_matrix)[1]
        self.model = LinearDiscriminantAnalysis(solver='lsqr')
        # self.model = LinearDiscriminantAnalysis(solver='svd')
        self.model.fit(training_matrix, training_labels)
