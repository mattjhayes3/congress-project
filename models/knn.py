import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np
from .model import Model

class KNNModel(Model):

    def name(self):
        return 'knn' if not self.instance_name else f"knn_{self.instance_name}"

    def getClassifierParams(self, training_matrix, training_labels, validation_matrix,  validation_labels, dictionary):

        training_matrix = training_matrix.toarray()
        validation_matrix = validation_matrix.toarray()
        merged_matrix = np.vstack( (training_matrix, validation_matrix) )
        merged_labels = np.append(training_labels, validation_labels, axis=0)

        print('Merged matrix size ' + str(np.shape(merged_matrix)))
        print('Merged labels ' + str(len(merged_labels)))
        n_neighbors = [1, 3, 5, 10, 20]
        p = [1, 2]
        param_grid = {'n_neighbors': n_neighbors, 'p':p}

        self.grid = HalvingGridSearchCV(KNeighborsClassifier(), param_grid=param_grid, scoring= 'accuracy', cv=4, verbose = 2, n_jobs = -1, factor=2)
        self.grid.fit(training_matrix, training_labels)

        print("The best parameters are %s with a score of %0.2f" % (self.grid.best_params_, self.grid.best_score_))
        return self.grid

    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):
        if self.grid is None:
            self.model = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
        else:
            self.model = KNeighborsClassifier(n_neighbors=self.grid.best_params_['n_neighbors'], p=self.grid.best_params_['p'])
        self.model.fit(training_matrix.toarray(), training_labels)
