import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np
from .model import Model


class RFCVModel(Model):
    def name(self):
        return 'rf_cv_per4e' if not self.instance_name else f"rf_cv_per4e_{self.instance_name}"

    def getClassifierParams(self, training_matrix, training_labels, validation_matrix,  validation_labels, dictionary):

        training_matrix = training_matrix.toarray()
        validation_matrix = validation_matrix.toarray()
        merged_matrix = np.vstack( (training_matrix, validation_matrix) )
        merged_labels = np.append(training_labels, validation_labels, axis=0)

        print('Merged matrix size ' + str(np.shape(merged_matrix)))
        print('Merged labels ' + str(len(merged_labels)))

        n_estimators = [ 300,275, 250, 225, 200,175, 150] #25 90,, 125, 145,  300 10, 30, 55, 75, 100, 110, 170, 200, 230 , 200  125, 50,
        max_depth = [None] # , , 20, 50 5, 10 , 15
        criterion = ["entropy"] # , "entropy" gini
        ccp_alpha = [0.0] # , 1e-2,  1e-5
        oob_score = [ False] # True
        max_samples = [None] #  0.125, 0.25, 0.5 , 0.8
        max_features = ['sqrt', 0.25] # 0.5**3, 0.5**2, 0.5, 'log2'
        param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth, 'oob_score': oob_score, 'max_samples': max_samples, 'max_features': max_features, 'criterion': criterion, 'ccp_alpha': ccp_alpha}

        self.grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, scoring= 'accuracy', cv=4, verbose = 2, n_jobs = -1) #factor=3
        self.grid.fit(training_matrix, training_labels)

        print("The best RF parameters are %s with a score of %0.2f" % (self.grid.best_params_, self.grid.best_score_))
        return self.grid

    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):
        if self.grid is None:
            self.model = RandomForestClassifier(n_estimators=250, max_depth=None, oob_score=False, max_samples=None, max_features=0.25, criterion='gini', ccp_alpha=0.0, n_jobs=-1)
        else: 
            self.model = RandomForestClassifier(n_estimators=grid.best_params_['n_estimators'], max_depth=grid.best_params_['max_depth'], oob_score=grid.best_params_['oob_score'], max_samples=grid.best_params_['max_samples'], max_features=grid.best_params_['max_features'], criterion=grid.best_params_['criterion'], ccp_alpha=grid.best_params_['ccp_alpha'])
        self.model.fit(training_matrix.toarray(), training_labels)
