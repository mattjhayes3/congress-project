import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from .model import Model
import xgboost as xgb

class XGBoostModel(Model):

    def name(self):
        return "xg_boost_fullgird" if not self.instance_name else f"xg_boost_fullgrid_{self.instance_name}"

    def getClassifierParams(self, training_matrix, training_labels, validation_matrix,  validation_labels, dictionary):
        training_matrix = training_matrix.toarray()
        validation_matrix = validation_matrix.toarray()
        merged_matrix = np.vstack( (training_matrix, validation_matrix) )
        merged_labels = np.append(training_labels, validation_labels, axis=0)

        print('Merged matrix size ' + str(np.shape(merged_matrix)))
        print('Merged labels ' + str(len(merged_labels)))
        param_grid = {'max_depth': [3, 4, 6], 'n_estimators':[ 50, 100, 200]} # 10, , 200
 
        self.grid = GridSearchCV(xgb.XGBClassifier(n_jobs=2), param_grid=param_grid, scoring= 'accuracy', cv=4, verbose = 2, n_jobs = -1) # factor=2
        self.grid.fit(training_matrix, training_labels)
        print("The best parameters are %s with a score of %0.2f" %
            (self.grid.best_params_, self.grid.best_score_))
        return self.grid

    def fit(self, training_matrix, training_labels, validation_matrix, validation_labels, dictionary):
        if self.grid is None:
            self.model = xgb.XGBClassifier(n_jobs=-1)
        else:
            self.model = xgb.XGBClassifier(n_jobs=-1, n_estimators=self.grid.best_params_['n_estimators'], max_depth=self.grid.best_params_['max_depth'])
        self.model.fit(training_matrix.toarray(), training_labels)
