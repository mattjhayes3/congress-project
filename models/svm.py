# original author: Ulya Bayram, adapted by: Matthew Hayes
# Here are the used classification methods are defined as separate functions
# calling them simply will provide necessary information for different feature types' classification in parallel
# ulyabayram@gmail.com, mattjhayes3@gmail.com
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np
# from .common_fns import *
from .model import Model
from scipy import sparse


class SVMModel(Model):

    def name(self):
        return 'svm_npr' if not self.instance_name else f"svm_npr_{self.instance_name}"

    def getClassifierParams(self, training_matrix, training_labels, validation_matrix,  validation_labels, dictionary):
        # training_matrix = training_matrix.toarray()
        # validation_matrix = validation_matrix.toarray()
        merged_matrix = np.vstack((training_matrix, validation_matrix))
        merged_labels = np.append(training_labels, validation_labels, axis=0)

        merged_matrix = sparse.csr_matrix(merged_matrix)
        print('Merged matrix size ' + str(np.shape(merged_matrix)))
        print('Merged labels ' + str(len(merged_labels)))
        gamma_ = ['scale']  # , pow(2, -8), pow(2, -4), 1, pow(2, 4), pow(2, 8)
        # pow(2, -10), pow(2, -5), 1024
        cost_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        # gamma_ = [1/64, 1/16, 1/4, 1, 4, 16, 64] # pow(2, -8), , pow(2, 8)
        # cost_range = [ 1/32,  pow(2, -2.5), 1, pow(2, 2.5), 32] #  pow(2, -10),  pow(2, 10)

        # kernels = ['linear', 'rbf', ] # 'sigmoid'
        shrinking = [False, True]
        linear_grid = {'kernel': ['linear'], 'C': cost_range,
                    'gamma': ['scale'], 'shrinking': shrinking}
        rbf_grid = {'kernel': ['rbf'], 'C': cost_range,
                    'gamma': gamma_, 'shrinking': shrinking}
        param_grid = [rbf_grid]  # linear_grid

        self.grid = GridSearchCV(SVC(), param_grid=param_grid, scoring='accuracy',
                                cv=4, verbose=2, n_jobs=-1)  # parallel search on , factor=2
        self.grid.fit(training_matrix, training_labels)
        print("The best SVM parameters are %s with a score of %0.2f" %
            (self.grid.best_params_, self.grid.best_score_))
        return self.grid


    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):
        training_matrix = sparse.csr_matrix(training_matrix)
        validation_matrix = sparse.csr_matrix(validation_matrix)
        if self.grid is None:
            self.model = SVC(verbose=True, cache_size=6000, kernel='rbf', probability=False,
                            C=16, gamma='scale', shrinking=False)
        else:
            self.model = SVC(kernel=self.grid.best_params_['kernel'], probability=False, C=self.grid.best_params_[
                            'C'], gamma=self.grid.best_params_['gamma'], shrinking=self.grid.best_params_['shrinking'])
        # classifier = SVC(kernel=grid.best_params_['kernel'], probability=True, C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], shrinking=False)
        # fit the model - train the classifier
        self.model.fit(training_matrix.toarray(), 2 * training_labels - 1)


    def predict(self, input_matrix):
        # return self.model.predict_proba(input_matrix)
        return (self.model.predict(input_matrix) + 1) / 2
