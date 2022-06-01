import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from .model import Model


class SKLogisticModel(Model):
    def name(self):
        return 'sk_logistic' if not self.instance_name else f"sk_logistic_{self.instance_name}"

    def getClassifierParams(self, training_matrix, training_labels, validation_matrix,  validation_labels, dictionary):

        training_matrix = training_matrix.toarray()
        validation_matrix = validation_matrix.toarray()
        merged_matrix = np.vstack( (training_matrix, validation_matrix) )
        merged_labels = np.append(training_labels, validation_labels, axis=0)

        print('Merged matrix size ' + str(np.shape(merged_matrix)))
        print('Merged labels ' + str(len(merged_labels)))
        none_grid = {'penalty':['none']}
        l1_grid = {'penalty':['l1'], 'C': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5, 10], 'solver':['saga']}
        l2_grid = {'penalty':['l2'], 'C': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5, 10], 'solver':['lbfgs', 'newton-cg']}
        elasticnet_grid = {'penalty':['elasticnet'], 'l1_ratio':[0.1, 0.3, 0.5, 0.7, 0.9], 'C': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5, 10], 'solver':['saga']}
        param_grid = [none_grid, l1_grid, l2_grid, elasticnet_grid]

        self.grid = HalvingGridSearchCV(LogisticRegression(max_iter=5000), param_grid=param_grid, scoring= 'accuracy', cv=4, verbose = 2, n_jobs = -1, factor=2)
        self.grid.fit(training_matrix, training_labels)
        print("The best parameters are %s with a score of %0.2f" % (self.grid.best_params_, self.grid.best_score_))
        return self.grid

    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):
        if self.grid is None:
            self.model = LogisticRegression(max_iter=5000, n_jobs=-1)
        else:
            C = self.grid.best_params_['C'] if 'C' in self.grid.best_params_ else 1
            solver = self.grid.best_params_['solver'] if 'solver' in self.grid.best_params_ else 'libfgs'
            l1_ratio = self.grid.best_params_['l1_ratio'] if 'l1_ratio' in self.grid.best_params_ else None
            self.model = LogisticRegression(max_iter=5000, penalty=self.grid.best_params_['penalty'], C=C, solver=solver, l1_ratio=l1_ratio)
        self.model.fit(training_matrix.toarray(), training_labels)
