# Ulya Bayram
# Classify unigram features
# ulyabayram@gmail.com
import math
import numpy as np
#import time
import classifier_functions_nn as cf
import scipy.stats
from scipy import stats

###########################################################################

def readTheClassifier(read_dir, feature_type, i_split):

    # load json and create model
    json_file = open(read_dir + 'models/'+feature_type+'_trained_model_h'+i_split+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(read_dir + 'models/'+feature_type+'_trained_model_h'+i_split+'.h5')

    # compile the model  using the same methods we used for training it
    loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_crossentropy', 'accuracy'])
    
    return loaded_model

if __name__ == "__main__":
    # Read the current classifier's save directory
    read_dir = cf.getSaveDirectory()

    for congress in range(97, 115):
        fmt_congress = "%03d" % congress
        print('Processing congress %s' $ fmt_congress)

        # read the features corresponding to the columns in these feature matrices
        fo_columns = open('../features'+congress+'/'+feature_type+'_matrix_columnames.txt', 'r')
        feature_names = fo_columns.read().split('\n')[:-1]
        fo_columns.close()

        # read the previously trained and pickled classifier
        model = cf.readTheClassifier(read_dir, feature_type, congress)

        # read the feature importances from the trained model, and save them
        savefilename_ = save_feature_dir + feature_type + '_feature_importance_h'+congress+'.txt'

        cf.saveFeatureImportances(savefilename_, model, feature_names)
