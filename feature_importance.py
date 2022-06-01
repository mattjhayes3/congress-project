# Ulya Bayram
# Classify unigram features
# ulyabayram@gmail.com
import math
import numpy as np
#import time
import scipy.stats
from scipy import stats
from models.logistic import LogisticModel
import models.common_fns as com
import keras
import json
import os

###########################################################################

def readTheClassifier(read_dir, congress, style):

    # load json and create model
    with open(f'{read_dir}models/unigram_trained_model_House_{congress}_{style}.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f'{read_dir}models/unigram_trained_model_House_{congress}_{style}.h5')

    # compile the model  using the same methods we used for training it
    loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_crossentropy', 'accuracy'])
    
    return loaded_model

if __name__ == "__main__":
    # Read the current classifier's save directory
    model = LogisticModel("dict")
    read_dir = "models/" + model.getSaveDirectory()
    # save_feature_dir = cf.getFeatureSaveDirectory()

    for congress in range(97, 115):
        fmt_congress = "%03d" % congress
        print('Processing congress %s' % fmt_congress)

        # read the features corresponding to the columns in these feature matrices
        with open(f"matricies/dicts/House_{fmt_congress}_max_balanced_0_10_50.json", "r") as j:
            dictionary = json.load(j)
        rev_dict = {v: k for k, v in dictionary.items()}

        # read the previously trained and pickled classifier
        trained = readTheClassifier(read_dir, fmt_congress, "3gram_max_balanced_0")

        # read the feature importances from the trained model, and save them
        os.makedirs(f"{read_dir}models/features/", exist_ok=True)
        savefile = f"{read_dir}models/features/House_{fmt_congress}_3gram_max_balanced_0_10_50.txt"

        com.saveFeatureImportances(savefile, trained, dictionary)
