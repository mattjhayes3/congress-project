# ILLUSTRATIONS.-Maps. diagrams. or illustrations may not be inserted in the RECORD without the approval of the Joint Committee on Printing. (Oct. 22. 1968. c. 9. 82 Stat. 1256.) To provide for the prompt publication and delivery on Printing has adopted the following rules. to which the attention of Senators. Representatives. and Delegates is respectfully invited: 1. Arrangement of the daily Congressional Record.The Public Printer shall arrange the contents of the proceedings shall alternate with the House proceedings in order of placement in consecutive issues insofar as such an arrangement is feasible. and Extensions of Remarks and Daily Digest shall follow: Provided. That the makeup of the CONGRESSIONAL RECORD shall proceed without regard to alternation whenever the Public Printer deems It necessary in order to meet production and delivery schedules. 2. Type and style.-The Public Printer shall print the report of the proceedings and debates of the Senate and House of Representatives. as furnished by In 8point type. and all matter included In the remarks or speeches of Members of Congress. other than their own words. and all reports. documents. and other matter authorized to be inserted in the type. and all rollcalls shall be printed in 6point type. No Italic or black type nor words in capitals or small capitals shall be used for emphasis or prominence. nor will unusual Indentions be permitted. These restrictions do not apply to the printing of or quotations from historical. official. or legal documents or papers of which a literal reproduction is necessary. 3. Only as an aid in distinguishing the manner of delivery In order to contribute to the historical accuracy of the RECORD. statements or insertions in the RECORD where no part of them was spoken will be preceded and followed by a "bullet" symbol. i.e.. 0. 4. Return of manuscript.-When manuscript is submitted to Members for revision it should be returned to the Government Printing Office not later than 9 oclock p.m. in order to insure publication in the morning. and if all of the manuscript is not furnished at the time specified. the Public Printer is authorized to withhold it from the CONGRESSIONAL RECORD for 1 day. In no case will a speech be printed ery if the manuscript is furnished later than 12 oclock midnight. 5. Tabular matter.-The manuscript of speeches containing tabular statements to be published in the Public Printer not later than 7 oclock p.m.. to insure publication the following morning. When possible. manuscript copy for tabular matter should be sent to the Government Printing Office 2 or more days in advance of the date of publication in the promptly to the Member of Congress to be submitted by him instead of manuscript copy when he of6. Proof furnished.-Proofs or "leave to print" and advance speeches will not be furnished the day the manuscript is received but will be submitted the following day. whenever possible to do so without causing delay in the publication of the regular proceedings of Congress. Advance speeches shall be set in more than six sets of proofs may be furnished to Members without charge. 7. Notation of withheld remarks.-If manuscript or proofs have not been returned in time for publication in the proceedings. the Public Printer will insert the words "Mr. - addressed the Senate (House or Committee). His remarks will appear hereafter in Extensions of Remarks" and proceed with 8. Thirtyday limitThe Public Printer shall not extension of remarks which has been withheld for a period exceeding 30 calendar days from the date when its printing was authorized: Provided. That at the expiration of each session of Congress the time limit herein fixed shall be 10 days. unless otherwise ordered by the committee. 9. Corrections.-The permanent CONGRESSIONAL RECORD is made up for printing and binding 30 days after each daily publication is issued. therefore all corrections must be sent to the Public Printer within that time: Provided. That upon the final adjournment of each session of Congress the time limit shall be 10 days. unless otherwise ordered by the committee: Provided further. That no Member of Congress shall be entitled to make more than one revision. Any revision shall consist only of corrections of the original copy and shall not include deletions of correct material. substitutions for correct material. or additions of new subject matter. 10. The Public Printer shall not publish in the any committee or subcommittee when the report or print has been previously printed. This rule shall not be construed to apply to conference reports. However. inasmuch as House of Representatives Rule XXVIII. Section 912. provides that conference reports be printed in the daily edition of the CONGRESSIONAL RECORD. they shall not be printed therein a second time. 11. Makeup of the Extensions of Remarks.-Extenshall be made up by successively taking first an extension from the copy submitted by the official reporters of one House and then an extension from the copy of the other House. so that Senate and House extensions appear alternately as far as possible. The sequence for each House shall follow as closely as possible the order or arrangement in which the copy comes from the official reporters of the respective Houses. The official reporters of each House shall designate and distinctly mark the lead item among their extensions. When beth Houses are in session and submit extensions. the lead item shall be changed from one House to the other in alternate issues. with the indicated lead item of the other House appearing in second place. When only one House is in session. the lead item shall be an extension submitted by a Member of the House in session. printed after the sine die adjournment of the Congress. 12. Official reporters.-The official reporters of each House shall indicate on the manuscript and prepare headings for all matter to be printed in Extensions of Remarks and shall make suitable reference thereto at the proper place in the proceedings. 13. Twopage ruleCost estimate from Public Printer.--(1) No extraneous matter in excess of two printed RECORD pages. whether printed in its entirety In one daily Issue or in two or more parts in one or more issues. shall be printed in the CONGRESSIONAL RECORD unless the Member announces. coincident with the request for leave to print or extend. the estimate in writing from the Public Printer of the probable cost of publishing the same. (2) No extraneous matter shall be printed in the House proceedings or the Senate proceedings. with the following exceptions: (a) Excerpts from letters. telegrams. or articles presented in connection with a speech delivered in the course of debate. (b) communications from State legislatures. (c) addresses or articles by the President and the Members of his Cabinet. the Vice President. or a Member of Congress. (3) The official reporters of the House or Senate or the Public Printer shall return to the Member of the respective House any matter submitted for the CONGRESSIONAL RECORD which is in contravention of these provisions.


import math
import numpy as np
#import time
from models.logistic import LogisticModel
from models.logistic_reg import LogisticRegModel
from models.nn20d import NN20DModel
from models.nn20nd import NN20NDModel
from models.nn20nd_reg import NN20NDRegModel
from models.nn1000d import NN1000DModel
from models.nn1000nd import NN1000NDModel
from models.mnnb import MNNBModel
from models.nn_multi import NNMultiModel
from models.rep_baseline import RepBaselineModel
from models.dem_baseline import DemBaselineModel
from models.lda import LDAModel
from models.sk_logistic import SKLogisticModel
from models.lstm_glove import LSTMGloveModel
from models.lstm_drop_bidi import LSTMDropBiDiModel
from models.lstm_bidi import LSTMBiDiModel
from models.xgboost import XGBoostModel
from models.lstm_drop import LSTMDropModel
from models.lstm_word2vec import LSTMWord2VecModel
from models.lstm_drop_glove import LSTMDropGloveModel
from models.cnn import CNNModel
from models.cnn2 import CNN2Model
from models.boost import BoostModel
from models.knn import KNNModel
from models.rf import RFModel
from models.rf_cv import RFCVModel
from models.lstm import LSTMModel
from models.nn_multi import NNMultiModel
from models.transformer import TransformerModel
from models.transformer_hd import TransformerHDModel
from models.transformer_max import TransformerMaxModel
from models.cnn2_avg import CNN2AvgModel
from models.cnn2_avg_drop import CNN2AvgDropModel
from models.logistic_sig import LogisticSigModel
from models.transformer_nd import TransformerNDModel
import models.common_fns as com
from models.svm import SVMModel
import pickle
import scipy.stats
import json
import os
import tensorflow as tf
from keras import backend as K
import glob
import pandas as pd
from scipy import sparse

# print(f'gpus= {K.tensorflow_backend._get_available_gpus()}')
# tf.debugging.set_log_device_placement(True)

# baselines = [rep_baseline, dem_baseline]

def run(save_dir, m, style, style_w_count, congress, chamber):
    style_wo_gram = style.replace('2gram_', '').replace('3gram_', '')
    # split_name = f"{congress}_{style.replace('2gram_', '')}"
    # 3) read the training, validation, and test set document lists
    with open(f'splits/{chamber}{congress}_{style_wo_gram}_train.txt', 'r') as f:
        all_training_files = f.read().split()
    with open(f'splits/{chamber}{congress}_{style_wo_gram}_valid.txt', 'r') as f:
        all_validation_files = f.read().split()
    with open(f'splits/{chamber}{congress}_{style_wo_gram}_test.txt', 'r') as f:
        all_test_files = f.read().split()

    ccs = f"{chamber}_{congress}_{style}"
    ccss = f"{chamber}_{congress}_{style_w_count}"
    styles = f"{style_w_count}" # _1_1 
    # read the complete feature matrices
    print(f"loading matrix... ")
    if m.is_sequence():
        print(f"reading from matricies/{ccss}_sequence_matrix.txt.npz")
        feature_matrix = sparse.load_npz(
            f"matricies/{ccss}_sequence_matrix.txt.npz").toarray()
        with open(f"matricies/{ccss}_sequence_row_files.txt") as f_rownames:
            filenames_rows = f_rownames.read().split('\n')[:-1]
    else:
        feature_matrix = sparse.load_npz(
            f"matricies/{ccss}_matrix.txt.npz").toarray()
        with open(f"matricies/{ccss}_row_files.txt") as f_rownames:
            filenames_rows = f_rownames.read().split('\n')[:-1]
    # read the filenames corresponding to the rows of above feature matrix
    print(f"files loaded")
    # print(f"feature matrix: {feature_matrix}")

    feature_matrix = m.preprocess(feature_matrix)
    print(f"preprocessing complete")

    # separate the training and validation set files into their respective classes
    training_d_files, training_r_files = com.separateGroupFiles(
        all_training_files)
    validation_d_files, validation_r_files = com.separateGroupFiles(
        all_validation_files)
    print(f"files separated")

    # split the feature matrix into training, validation and test, and collect their class labels
    matrices, labels, row_filenames = com.splitFeatureMatrix(feature_matrix, filenames_rows, training_d_files, training_r_files,
                                                             validation_d_files, validation_r_files, all_test_files)
    print(f"matricies split")
    training_matrix = matrices['train']
    validation_matrix = matrices['validation']
    test_matrix = matrices['test']
    print(f"training_matrix.shape={training_matrix.shape}")
    print(f"validation_matrix.shape={validation_matrix.shape}")
    print(f"test_matrix.shape={test_matrix.shape}")
    dictionary = dict()
    if m.is_sequence():
        print(f"reading from matricies/dicts/{ccss}_sequence.json")
        with open(f"matricies/dicts/{ccss}_sequence.json", "r") as j:
            dictionary = json.load(j)
            rev_dict = {v: k for k, v in dictionary.items()}
            # print(rev_dict)
        for i in [ 10]:
            for n, a in matrices.items():
                z = a.toarray()
                s = z[i, :]
                l = [rev_dict[w] if w in rev_dict else (
                    "_" if w == 0 else "__oov__") for w in s]
                # print(l)
                print(f"example from {n}: {' '.join(l)}")

    training_labels = labels['train']
    validation_labels = labels['validation']
    test_labels = labels['test']
    print(f"len(training_labels)={len(training_labels)}")
    print(f"len(validation_labels)={len(validation_labels)}")
    print(f"len(test_labels)={len(test_labels)}")

    training_row_filenames = row_filenames['train']
    validation_row_filenames = row_filenames['validation']
    test_matrix_row_filenames = row_filenames['test']
    print(f"len(validation_row_filenames)={len(validation_row_filenames)}")
    print(f"len(training_row_filenames)={len(training_row_filenames)}")
    print(f"len(test_matrix_row_filenames)={len(test_matrix_row_filenames)}")

    # some classifiers might have some parameter selection (classifier model building) requirement
    # if so, do it here, now - and use the selected parameters
    # if classifier has no such dependence, params_ will be an empty list [] and classifier functions will ignore it
    #params_ = cf.getClassifierParams(training_matrix, validation_matrix, training_labels, validation_labels)

    #pickle.dump(model, open(filename, 'wb'))

    # print(f"type(training_matrix)={type(training_matrix)}, type(training_labels)={type(training_labels)}")
    # print(f"type(validation_matrix)={type(validation_matrix)}, type(validation_labels)={type(validation_labels)}")

    os.makedirs(save_dir + 'models/', exist_ok=True)
    print(f"fitting...")
    # grid = m.getClassifierParams(training_matrix, training_labels, validation_matrix, validation_labels, dictionary)
    # if grid is not None:
    #    df = pd.DataFrame.from_dict(grid.cv_results_)
    #    df.sort_values('rank_test_score')
    #    df.to_csv(f'{save_dir}models/{chamber}{congress}_{style}_cv_results.csv')
    m.fit(training_matrix, training_labels,
          validation_matrix, validation_labels, dictionary)
    if not m.is_baseline():
        with open(f"{save_dir}models/unigram_trained_model_{ccs}.json", "w") as json_file:
            json_file.write(m.get_json())
    if m.is_h5():
        m.model.save_weights(f"{save_dir}models/unigram_trained_model_{ccs}.h5")

    # perform the classifications using the same trained classifier
    training_classification_results_prob = m.predict(training_matrix.toarray())
    validation_classification_results_prob = m.predict(
        validation_matrix.toarray())
    test_classification_results_prob = m.predict(test_matrix.toarray())

    # save the file level feature vector classification results
    train_classifications_path = f"{save_dir}Training/training_classified_unigram_%s_{ccss}.txt"
    com.saveFileLevelClassificationResults(congress, training_row_filenames, training_labels, training_classification_results_prob,
                                           train_classifications_path, np.shape(training_matrix)[1], save_details=False)
    val_classifications_path = f"{save_dir}Validation/validation_classified_unigram_%s_{ccss}.txt"
    com.saveFileLevelClassificationResults(congress, validation_row_filenames, validation_labels, validation_classification_results_prob,
                                           val_classifications_path, np.shape(validation_matrix)[1], save_details=False)
    test_classifications_path = f"{save_dir}Test/test_classified_unigram_%s_{ccss}.txt"
    com.saveFileLevelClassificationResults(congress, test_matrix_row_filenames, test_labels, test_classification_results_prob,
                                           test_classifications_path, np.shape(test_matrix)[1], save_details=False)

    # save the overall, split level results for the validation, test, and shor sets
    com.saveSplitStats(save_dir, m.name(), congress, chamber, styles,  "train",
                       training_classification_results_prob, training_labels)
    # com.saveSplitStats(save_dir, cf.name(), i_split, "validation", validation_classification_results_prob, validation_labels)
    com.saveSplitStats(save_dir, m.name(), congress, chamber, styles,  "valid",
                       validation_classification_results_prob, validation_labels)
    test_stats = com.saveSplitStats(save_dir, m.name(
    ), congress, chamber, styles, "test", test_classification_results_prob, test_labels)
    print(f"test accuracy {test_stats['accuracy']}")

# i stand in opposition to the republican affordable care repeal act because it is an irresponsible approach that does nothing to address the rising cost of health care that our families and our businesses are facing today * it is a fact that the __oov__ cost for most unitedstates companies is health care * without the affordable care act overall health care costs will continue to rise even faster costs that will be borne by both the public and private sector * it is important to note that voting for this repeal bill will eliminate the small business health care tax credit * this tax credit currently allows small businesses to offset up to 35 percent of their health care insurance cost * starting in 2014 the credit will increase to 50 percent of premium co

if __name__ == "__main__":
    np.random.seed(0)
    tf.random.set_seed(0)
    # models = [CNN2AvgDropModel(300, pretrained='glove840'), 
    #         CNN2AvgModel(300, pretrained='glove840'),
    #         CNN2AvgDropModel(300, pretrained='glove840', trainable=True), 
    #         CNN2AvgModel(300, pretrained='glove840', trainable=True), 
    #         TransformerModel(50, 128, pretrained='glove'), 
    #         TransformerModel(100, 128, pretrained='glove'), 
    #         TransformerModel(200, 128, pretrained='glove'), 
    #         TransformerModel(300, 128, pretrained='glove'),
    #         TransformerModel(300, 128, pretrained='glove840'),
    #         TransformerModel(50, 128, pretrained='glove', trainable=True),
    #         TransformerModel(100, 128, pretrained='glove', trainable=True), 
    #         TransformerModel(200, 128, pretrained='glove', trainable=True), 
    #         TransformerModel(300, 128, pretrained='glove', trainable=True),
    #         TransformerModel(300, 128, pretrained='glove840', trainable=True), 
    #         TransformerModel(50, 128), 
    #         TransformerModel(100, 128), 
    #         TransformerModel(200, 128), 
    #         TransformerModel(300, 128)]
    # models = [LDAModel(), SVMModel()]
    # models = [RFCVModel(), KNNModel()]
    # models = [LDAModel('6-3'), SVMModel('6-3')]
    # models = [BoostModel('6-3'), XGBoostModel('6-3')]
    models = [CNN2AvgDropModel(300, pretrained='glove840'), 
        CNN2AvgDropModel(300, pretrained='glove840', trainable=True), 
        CNN2AvgDropModel(300, pretrained='glove'), 
        CNN2AvgDropModel(300, pretrained='glove', trainable=True),
        CNN2AvgDropModel(200, pretrained='glove'), 
        CNN2AvgDropModel(200, pretrained='glove', trainable=True),
        ]
    for m_num, m in enumerate(models): 
        print(f"### model number {m_num}/{len(models)} ###")
        #
        # for m in [CNN2AvgModel(200, glove=True), CNN2AvgModel(300, glove=True), CNN2AvgDropModel(200, glove=True), CNN2AvgDropModel(300, glove=True), CNN2AvgModel(200, glove=True, trainable=True), CNN2AvgModel(300, glove=True, trainable=True), CNN2AvgDropModel(200, glove=True, trainable=True), CNN2AvgDropModel(300, glove=True, trainable=True), CNN2AvgModel(200, glove=False), CNN2AvgModel(300, glove=False), CNN2AvgDropModel(200, glove=False), CNN2AvgDropModel(300, glove=False)]: 
        # for m in [LogisticModel()]: 
        save_dir = "models/" + m.getSaveDirectory()
        for subdir in ["", "models/", "Training/", "Validation/", "Test/"]:
            print(f"makedir", save_dir+subdir)
            os.makedirs(save_dir + subdir, exist_ok=True)
        # for i_split in [ '100', '103', '106', '109', '112', '114']:  '097',
        for style, style_w_count in [('max_balanced_0', 'max_balanced_0_1_1')]: #
        # for style, style_w_count in [('bayram', 'bayram'), ('bayram', 'bayram_3_7'), ('bayram', 'bayram_1_1'), ('max_balanced_0', 'max_balanced_0'), ('3gram_max_balanced_0', '3gram_max_balanced_0'), ('2gram_max_balanced_0', '2gram_max_balanced_0')]: #
        # for style, style_w_count in [('max_balanced_0', 'max_balanced_0_10_50')]: #
        # for style, style_w_count in [('bayram', 'bayram')]: #
            for chamber in ['House']: # , 'Senate'
                for congress in [97, 100, 103, 106, 109, 112, 114]:
                    fmt_congress = "%03d" % congress
                    np.random.seed(0)
                    tf.random.set_seed(0)
                    print(f"*** Running for {m.name()}, {congress} ***")
                    if not m.use_gpu():
                        with tf.device("/cpu:0"):
                            run(save_dir, m, style, style_w_count, fmt_congress, chamber)
                    else:
                        run(save_dir, m, style, style_w_count, fmt_congress, chamber)

        # print(( * df[df['split']=='test'])['dataset', 'accuracy'])
