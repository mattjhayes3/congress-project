# Ulya Bayram
# Classify unigram features
# ulyabayram@gmail.com
import math
import numpy as np
#import time
import models.logistic as logistic
import models.logistic_reg as logistic_reg
import models.nn20d as nn20d
import models.nn20nd as nn20nd
import models.nn20nd_reg as nn20nd_reg
import models.nn1000d as nn1000d
import models.nn1000nd as nn1000nd
import models.mnnb as mnnb
import models.rep_baseline as rep_baseline
import models.dem_baseline as dem_baseline
import models.lda as lda
import models.sk_logistic as sk_logistic
import models.logistic_old as logistic_old
import models.boost as boost
import models.knn as knn
import models.rf as rf
import models.rf_cv as rf_cv
from models.common_fns import *
import models.svm as svm
import pickle
import scipy.stats
import json
import os
import tensorflow as tf
from keras import backend as K
import glob
import pandas as pd

# print(f'gpus= {K.tensorflow_backend._get_available_gpus()}')
###########################################################################

# Read the current classifier's save directory


feature_type = 'unigram'
baselines = [rep_baseline, dem_baseline]
# tf.debugging.set_log_device_placement(True)

def run(save_dir, cf, i_split):
    # 3) read the training, validation, and test set document lists
    with open('splits/train'+ str(i_split) + '.txt', 'r') as f_test:
        all_training_files = f_test.read().split()
    with open('splits/validation'+ str(i_split) + '.txt', 'r') as f_test:
        all_validation_files = f_test.read().split()
    with open('splits/test'+ str(i_split) + '.txt', 'r') as f_test:
        all_test_files = f_test.read().split()

    # separate the training and validation set files into their respective classes
    training_d_files, training_r_files = separateGroupFiles(all_training_files)
    validation_d_files, validation_r_files = separateGroupFiles(all_validation_files)

    # read the complete feature matrices
    feature_matrix = np.loadtxt(f"matricies/{i_split}_old matrix.txt")
    feature_matrix = np.loadtxt(f"matricies/{i_split}_matrix.txt")

    if not cf in [mnnb]:
        feature_matrix = applyFeaturePreprocessing(feature_matrix)
  
    # read the filenames corresponding to the rows of above feature matrix
    f_rownames = open(f"matricies/{i_split}_row_files.txt")
    filenames_rows = f_rownames.read().split('\n')[:-1]
    f_rownames.close()

    # split the feature matrix into training, validation and test, and collect their class labels
    matrices, labels, row_filenames = splitFeatureMatrix(feature_matrix, filenames_rows, training_d_files, training_r_files,
                                                                validation_d_files, validation_r_files, all_test_files)
    training_matrix = matrices['train']
    validation_matrix = matrices['validation']
    test_matrix = matrices['test']

    training_labels = labels['train']
    validation_labels = labels['validation']
    test_labels = labels['test']

    training_row_filenames = row_filenames['train']
    validation_row_filenames = row_filenames['validation']
    test_matrix_row_filenames = row_filenames['test']

    # some classifiers might have some parameter selection (classifier model building) requirement
    # if so, do it here, now - and use the selected parameters
    # if classifier has no such dependence, params_ will be an empty list [] and classifier functions will ignore it
    #params_ = cf.getClassifierParams(training_matrix, validation_matrix, training_labels, validation_labels)

    # train the classifier - save the trained model - in case it might be necessary to re-use it in the future
    #model = cf.fit(training_matrix, training_labels, params_)
    #filename = str(save_dir + 'models/'+feature_type+'_trained_model_h'+ str(i_split) + '.sav')
    #pickle.dump(model, open(filename, 'wb'))

    # print(f"type(training_matrix)={type(training_matrix)}, type(training_labels)={type(training_labels)}")
    # print(f"type(validation_matrix)={type(validation_matrix)}, type(validation_labels)={type(validation_labels)}")
    sk_models = [mnnb, rf, lda, sk_logistic, boost, knn]
    cv_models = [svm, rf_cv, knn]
    os.makedirs(save_dir + 'models/', exist_ok=True)
    if cf in cv_models:
        grid = cf.getClassifierParams(training_matrix, validation_matrix, training_labels, validation_labels)
        pd.DataFrame.from_dict(grid.cv_results_).to_csv(f'{save_dir}models/cv_results.csv')

        model = cf.fit(training_matrix, training_labels, grid)
        # model = cf.fitNoGrid(training_matrix, training_labels)
        model_json = json.dumps(model.get_params())
    else:
        # print(f"type(training_matrix)={type(training_matrix)}, type(training_labels)={type(training_labels)}")
        # print(f"type(validation_matrix)={type(validation_matrix)}, type(validation_labels)={type(validation_labels)}")
        model = cf.fit(training_matrix, training_labels, validation_matrix, validation_labels)
        model_json = ""
        if not cf in baselines:
            model_json = json.dumps(model.get_params()) if cf in sk_models else model.to_json()
            if not cf in sk_models:
                model.save_weights(save_dir + 'models/'+feature_type+'_trained_model_h'+ str(i_split) + '.h5')
    if not cf in baselines:
        with open(save_dir + 'models/'+feature_type+'_trained_model_h'+ str(i_split) + '.json', "w") as json_file:
            json_file.write(model_json)
    # serialize weights to HDF5

    # perform the classifications using the same trained classifier
    training_classification_results_prob = cf.predict(training_matrix.toarray(), model)
    validation_classification_results_prob = cf.predict(validation_matrix.toarray(), model)
    test_classification_results_prob = cf.predict(test_matrix.toarray(), model)

    # save the file level feature vector classification results
    train_classifications_path = save_dir + 'Training/training_classified_'+feature_type+'_matrix_results_house'+ str(i_split) + '.txt'
    saveFileLevelClassificationResults(training_row_filenames, training_labels, training_classification_results_prob,
                                              train_classifications_path, np.shape(training_matrix)[1])
    val_classifications_path = save_dir + 'Validation/validation_classified_'+feature_type+'_matrix_results_house'+ str(i_split) + '.txt'
    saveFileLevelClassificationResults(validation_row_filenames,validation_labels, validation_classification_results_prob,
                                              val_classifications_path, np.shape(validation_matrix)[1])
    test_classifications_path = save_dir + 'Test/test_classified_'+feature_type+'_matrix_results_house'+ str(i_split) + '.txt'
    saveFileLevelClassificationResults(test_matrix_row_filenames, test_labels, test_classification_results_prob,
                                              test_classifications_path, np.shape(test_matrix)[1])

    # save the overall, split level results for the validation, test, and shor sets
    saveSplitStats(save_dir, cf.__name__, i_split, "train", training_classification_results_prob, training_labels)
    saveSplitStats(save_dir, cf.__name__, i_split, "validation", validation_classification_results_prob, validation_labels)
    test_stats = saveSplitStats(save_dir, cf.__name__, i_split, "test", test_classification_results_prob, test_labels)
    print(f"test accuracy {test_stats['accuracy']}")

if __name__ == "__main__":
    np.random.seed(0)
    tf.random.set_seed(0)
    # for i_split in ['98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113']: #, 
    # for cf in [logistic, nn20d, nn1000d, nn1000nd]: #svm
        # for i_split in ['103']: #, 
    # for cf in [svm]: logistic,
    # for cf in [mnnb]:
    for cf in [logistic]:
    # for cf in [svm]:
    # for cf in [ nn20d, nn20nd, nn1000d, nn1000nd, svm]:
        save_dir = "models/" + cf.getSaveDirectory()
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir + "models/", exist_ok=True)
        os.makedirs(save_dir + "Training/", exist_ok=True)
        os.makedirs(save_dir + "Validation/", exist_ok=True)
        os.makedirs(save_dir + "Test/", exist_ok=True)
        for i_split in ['100',]: #, '097',   '103', '106', '109', '112', '114'
            print(f"*** Running for {cf.__name__}, {i_split} ***")
            if cf in [logistic, logistic_old, nn20d, nn20nd, nn20nd_reg, logistic_reg]:
                with tf.device("/cpu:0"):
                    run(save_dir, cf, i_split)
            else:
                run(save_dir, cf, i_split)
    
    # print(( * df[df['split']=='test'])['dataset', 'accuracy'])
