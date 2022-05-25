import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from .model import Model

class BoostModel(Model):

    def name(self):
        return "boost" if not self.instance_name else f"boost_{self.instance_name}"

    def getClassifierParams(self, training_matrix, training_labels, validation_matrix,  validation_labels, dictionary):
        training_matrix = training_matrix.toarray()
        validation_matrix = validation_matrix.toarray()
        merged_matrix = np.vstack( (training_matrix, validation_matrix) )
        merged_labels = np.append(training_labels, validation_labels, axis=0)

        print('Merged matrix size ' + str(np.shape(merged_matrix)))
        print('Merged labels ' + str(len(merged_labels)))
        param_grid = {'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3)], 'n_estimators':[10, 30, 50, 75, 100, 150, 200]}

        self.grid = HalvingGridSearchCV(AdaBoostClassifier(), param_grid=param_grid, scoring= 'accuracy', cv=4, verbose = 2, n_jobs = -1, factor=2)
        self.grid.fit(training_matrix, training_labels)
        return self.grid

    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):
        if self.grid is None:
            self.model = AdaBoostClassifier()
        else:
            self.model = AdaBoostClassifier(base_estimator=self.grid.best_params_['base_estimator'], n_estimators=self.grid.best_params_['n_estimators'])
        self.model.fit(training_matrix.toarray(), training_labels)
