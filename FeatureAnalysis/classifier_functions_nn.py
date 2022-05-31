# author: Ulya Bayram
# Here are the used classification methods are defined as separate functions
# calling them simply will provide necessary information for different feature types' classification in parallel
# ulyabayram@gmail.com
import numpy as np
import math
from scipy import sparse
import keras
import compute_AUC_functions as auc
import scipy.stats
from scipy import stats
import pickle

def getReadDirectory():
    return '../Logistic_results/'

def getFeatureSaveDirectory():
    return 'Logistic/'

# inside, read the trained model to the corresponding folder - might be needed in the future
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

def saveFeatureImportances(savefilename_, trained_model, feature_names):

    # get the input-output layer weights
    layer_weights = []
    for layer in trained_model.layers:
        layer_weights.append( layer.get_weights() ) # list of numpy arrays

    #print(layer_weights[0][1][0])
    #print(layer_weights[0][1][1])
    #bias_ = layer_weights[0][1]
    btw_input_hidden_weights = layer_weights[0][0] # don't ignore the bias array

    #print(bias_)
    f_importance = open(savefilename_, 'w')
    c = 0
    for word_i in range(len(feature_names)):
        curr_uni_weights = btw_input_hidden_weights[c]
        f_importance.write(feature_names[word_i] + '\t' + str(curr_uni_weights[0]) + '\t' + str(curr_uni_weights[1]) +
                                     '\t' + str(curr_uni_weights[0] - curr_uni_weights[1]) + '\n')
        c += 1
    f_importance.close()

    f_importance = open(savefilename_[:-4] + '_bias.txt', 'w')
    f_importance.write(str(layer_weights[0][1][0]) + '\t' + str(layer_weights[0][1][1]))
    f_importance.close()
