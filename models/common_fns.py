import numpy as np
import math
from scipy import sparse
import keras
import compute_AUC_functions as auc
import scipy.stats
from scipy import stats
import time
import pickle
import json

def loadEmbeddings(path):
    embeddings_index = {}
    with open(path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            # if word == ".":
            #     print("found period")
            # if not word.isalpha():
            #     print(f"non-alpha '{word}'")
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    return embeddings_index


def separateGroupFiles(list_of_files):

    d_list = []
    r_list = []

    if '' in list_of_files:
        del list_of_files[list_of_files.index('')]

    for filename in list_of_files:

        if 'd' in filename[0]:
            d_list.append(filename)
        elif 'r' in filename[0]:
            r_list.append(filename)
        else:
            print('Skipping last - empty - line!')

    return d_list, r_list

def rowNormalizeMatrix(matrix_):
    row_wise_sqrts = np.sqrt(np.sum(np.square(matrix_), axis=1))
    matrix_2 = np.copy(matrix_)

    # now divide each row items with the corresponding square root values
    for i_row in range(np.shape(matrix_2)[0]):
        if row_wise_sqrts[i_row] != 0:
            matrix_2[i_row, :] = matrix_2[i_row, :] / row_wise_sqrts[i_row]

    return matrix_2

def applyLogNormalizationNoUnit(feature_matrix):
    return np.sign(feature_matrix)*np.log( np.abs(feature_matrix) + 1 )

def applyLogNormalization(feature_matrix):
    # assert np.allclose(np.abs(feature_matrix), feature_matrix)
    # our default scaling method is the conversion of feature values to log domain
    # this shrinks the magnitude-related differences between feature types
    # also, keeps the feature value sign intact - which standard scaling methods fail to do so
    feature_matrix = np.sign(feature_matrix)*np.log( np.abs(feature_matrix) + 1 )
    feature_matrix = rowNormalizeMatrix(feature_matrix)
    return feature_matrix

def applyTFIDFNormalization(feature_matrix):
    tf = np.sign(feature_matrix)*np.log( np.abs(feature_matrix) + 1 )
    idf = np.log(1+1/np.mean(feature_matrix>0, axis=0))
    return rowNormalizeMatrix(tf * idf)
    

def splitFeatureMatrix(feature_matrix, feature_filenames, training_dem_files, training_rep_files,
                           validation_dem_files, validation_rep_files, test_files):

    # initialize corresponding future feature matrices
    num_features = np.shape(feature_matrix)[1]
    training_files = training_dem_files + training_rep_files
    validation_files = validation_dem_files + validation_rep_files

    # doesn't hurt to shrink the memory usage - should increase the runtime speed
    # training_matrix = sparse.lil_matrix( np.zeros( (len(training_files), num_features) ) )
    # validation_matrix = sparse.lil_matrix( np.zeros( (len(validation_files), num_features) ) )
    # test_matrix = sparse.lil_matrix( np.zeros( (len(test_files), num_features) ) )

    # split the filenames by their classes and initialize the class label lists
    test_dem_files, test_rep_files = separateGroupFiles(test_files)

    training_labels = []
    validation_labels = []
    test_labels = []

    training_rows = []
    validation_rows = []
    test_rows = []
    training_indices = []
    validation_indices = []
    test_indices = []
    # start = time.time()
    for file_index in range(len(feature_filenames)):
        filename = feature_filenames[file_index]
        # feature_vector = feature_matrix[file_index, :]

        if filename in training_files:
            training_indices.append(file_index)
            # training_matrix[len(training_labels), :] = feature_vector
            training_rows.append(filename)

            if filename in training_dem_files:
                training_labels.append(1)
            elif filename in training_rep_files:
                training_labels.append(0)
            else:
                print('Error in training filenames')
        elif filename in validation_files:
            validation_indices.append(file_index)
            # validation_matrix[len(validation_labels), :] = feature_vector
            validation_rows.append(filename)

            if filename in validation_dem_files:
                validation_labels.append(1)
            elif filename in validation_rep_files:
                validation_labels.append(0)
            else:
                print('Error in validation filenames')
        elif filename in test_files:
            test_indices.append(file_index)
            # test_matrix[len(test_labels), :] = feature_vector
            test_rows.append(filename)

            if filename in test_dem_files:
                test_labels.append(1)
            elif filename in test_rep_files:
                test_labels.append(0)
            else:
                print('Error in test filenames')
        else:
            raise Exception(f"{filename} not in training (size {len(training_files)}) test (size {len(test_files)}) or valid (({len(validation_files)}))")
    # print(f"slow took {time.time() - start}")

    # start = time.time()
    training_matrix_fast = sparse.csr_matrix(feature_matrix[training_indices, :])
    validation_matrix_fast = sparse.csr_matrix(feature_matrix[validation_indices, :])
    test_matrix_fast = sparse.csr_matrix(feature_matrix[test_indices, :])
    # print(f"fast took {time.time() - start}")
    # assert np.allclose(training_matrix_fast.toarray(), training_matrix.toarray())
    # assert np.allclose(validation_matrix_fast.toarray(), validation_matrix.toarray())
    # assert np.allclose(test_matrix_fast.toarray(), test_matrix.toarray())

    matrices = {}
    matrices['train'] = training_matrix_fast
    matrices['validation'] = validation_matrix_fast
    matrices['test'] = test_matrix_fast
    for name, matrix in matrices.items():
        print(f"{name} has shape {matrix.shape}")

    labels = {}
    labels['train'] = np.array(training_labels)
    labels['validation'] = np.array(validation_labels)
    labels['test'] = np.array(test_labels)

    row_filenames = {}
    row_filenames['train'] = training_rows
    row_filenames['validation'] = validation_rows
    row_filenames['test'] = test_rows
    return matrices, labels, row_filenames

def saveFileLevelClassificationResults(congress, input_files, true_labels, predicted_probs,
                                  save_path_pattern, input_numfeatures):
    f_results_save_path = save_path_pattern % "matrix_reuslts"
    by_length_path = save_path_pattern % "by_length"
    top_path = save_path_pattern % "top"
    length_counts = {"<150":0, "<300": 0, "<500": 0, "<1000": 0, "<2000": 0, '2k+': 0}
    length_correct = {"<150":0, "<300": 0, "<500": 0, "<1000": 0, "<2000": 0, '2k+': 0}
    correct_list = []
    incorrect_list = []
    unsure_list = []
    with open(f_results_save_path, 'w') as f_results_save_input:
        f_results_save_input.write('filename\ttrueclass\tpredclass_prob\tnumfeatures\n')
        ic = 0
        # print(f"shape results = {predicted_probs.shape}")
        if len(predicted_probs.shape)>1:
            # print(f"len x shape = {len(predicted_probs.shape)}")
            if len(predicted_probs[0]) > 1: # if there are more than two values in the first item of the probs list
                predicted_probs = predicted_probs[:, 1] # take the second columns only
            else:
                predicted_probs = predicted_probs[:, 0]

        for file_ in input_files:
            with open(f"../processed_data/House{congress}_unigrams/{file_}") as fo:
                content = fo.read().splitlines()[0]
            length = len(content.split())
            predicted_pr = float(predicted_probs[ic])
            prediction = 1 if predicted_pr >= 0.5 else 0
            true_label = int(true_labels[ic])
            correct = prediction==true_label
            unsure_list.append((abs(predicted_pr-0.5), predicted_pr,  true_label, content))
            if correct:
                correct_list.append((abs(predicted_pr-true_label), predicted_pr,  true_label, content))
            else:
                incorrect_list.append((abs(predicted_pr-true_label), predicted_pr, true_label, content))
            if length < 150:
                length_counts["<150"]+=1
                length_correct["<150"]+= int(correct)
            if length < 300:
                length_counts["<300"]+=1
                length_correct["<300"]+= int(correct)
            elif length < 500:
                length_counts["<500"]+= 1
                length_correct["<500"]+= int(correct)
            elif length < 1000:
                length_counts["<1000"]+= 1
                length_correct["<1000"]+= int(correct)
            elif length < 2000:
                length_counts["<2000"]+= 1
                length_correct["<2000"]+= int(correct)
            else:
                length_counts["2k+"]+= 1
                length_correct["2k+"]+= int(correct)
            # 'filename\tsplitnum\ttrueclass\tewd_result\tpredclass_prob\tnumfeatures\n'
            # 'filename\tsplitnum\ttrueclass\tpredclass_prob\tnumfeatures\n'
            f_results_save_input.write(file_ + '\t' + str(true_labels[ic]) +
                                           '\t' + str(predicted_probs[ic]) + '\t' + str(input_numfeatures) + '\n')
            ic += 1
        length_acc = {k:(v/length_counts[k] if length_counts[k]>0 else "N/A") for k,v in length_correct.items()}
        with open(by_length_path, "w") as lo:
            json.dump(length_acc, lo)
        correct_list.sort()
        incorrect_list.sort(reverse=True)
        unsure_list.sort()
        top_correct = correct_list[:min(len(correct_list), 10)]
        top_incorrect = incorrect_list[:min(len(incorrect_list), 10)]
        top_unsure = unsure_list[:min(len(unsure_list), 10)]
        with open(top_path, "w") as lo:
            json.dump({'top_correct':top_correct, 'top_incorrect':top_incorrect, 'top_unsure': top_unsure}, lo)



def saveSplitStats(save_dir, model_name, congress, chamber, style, split, predicted_probs, true_labels):
    path = f"{save_dir}{split}_stats_{chamber}_{congress}_{style}.json"
    print(f"saving to {path}")
    # print(f"shape results = {predicted_probs.shape}")
    if len(predicted_probs.shape)>1:
        # print(f"len x shape = {len(predicted_probs.shape)}")
        if len(predicted_probs[0]) > 1: # if there are more than two values in the first item of the probs list
            predicted_probs = predicted_probs[:, 1] # take the second columns only
        else:
            predicted_probs = predicted_probs[:, 0]

    result = {'model': model_name, 'dataset':f'h{congress}_{style}', 'congress': congress, 'chamber':chamber, 'style':style, 'split': split}

    positive_indices = true_labels == 1
    negative_indices = ~positive_indices
    positive_predictions = predicted_probs[positive_indices]
    negative_predictions = predicted_probs[negative_indices]
    result['n'] = int(true_labels.shape[0])
    result['TP'] = int(np.sum(positive_predictions >= 0.5))
    result['FP'] = int(np.sum(negative_predictions >= 0.5))
    result['TN'] = int(np.sum(negative_predictions < 0.5))
    result['FN'] = int(np.sum(positive_predictions < 0.5))
    assert result['TP'] + result['FP'] + result['TN'] + result['FN'] == result['n']

    result['precision'] = result['TP'] / (result['TP'] + result['FP']) if result['TP'] + result['FP'] > 0 else None
    result['recall'] = result['TP'] / (result['TP'] + result['FN']) if result['TP'] + result['FN'] > 0 else None
    result['accuracy'] = (result['TP'] + result['TN']) / result['n']
    result['f1'] = 2 * result['precision'] * result['recall'] / (result['precision'] + result['recall']) if result['precision'] and result['recall'] else None

    result['negative_accuracy'] = result['TN'] / (result['TN'] + result['FP']) if result['TN'] + result['FP'] > 0 else None
    result['balanced_accuracy'] = (result['recall'] + result['negative_accuracy']) / 2

    # compute AUC and the 95% confidence interval
    auc_score, auc_cov_score = auc.delong_roc_variance(np.array(true_labels), predicted_probs.reshape(-1,))
    auc_std = np.sqrt(auc_cov_score)
    alpha = 0.95 # selected confidence level
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci_score = stats.norm.ppf(lower_upper_q, loc=auc_score, scale=auc_std)
    ci_score[ci_score > 1] = 1
    result['AUC'] = auc_score
    result[f'CI_{100* alpha}'] = (ci_score[0], ci_score[1])

    with open(path, 'w') as f:
        json.dump(result, f)
    return result
