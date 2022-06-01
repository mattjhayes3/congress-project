import glob
import os
import re
import random

def getBasenames(filename_list):
    return [os.path.basename(f) for f in filename_list]

def cleanSpeechData(txt_):
    # delete some unnecessary stuff
    txt_ = txt_.replace('\n.', '')
    txt_ = txt_.replace('-', " ")
    #print(txt_)
    # delete some symbols if they exist within the text
    txt_ = re.sub(r'\[(.*?)\]', '', txt_)

    txt_ = re.sub(r'\,|\(|\)|\[|\]|\*|\&|\$|\@|\%|\-|\:|\"|\_|\+|\#|\;|\\|\/', ' ', txt_)

    txt_ = txt_.replace('..', '.') # replace multiple periods with a single one

    # replace periods with our sentence boundary signs
    txt_ = txt_.replace('.', ' * ')
    txt_ = txt_.replace('?', ' * ')
    txt_ = txt_.replace('!', ' * ')
    
    txt_ = re.sub(r' +', ' ', txt_)
    txt_ = txt_.lower()

    # clean the numericals from this text - convert them to 17 - my constant
    text_ = txt_.split(' ')
    # for word_i in range(len(text_)):
    #     if text_[word_i].isdigit():
    #         # convert all digits within text to 17
    #         text_[word_i] = '17'

    return text_

def makeProperForSentences(words_list):

    complete_text = " ".join(words_list)
    del words_list
    complete_text = complete_text.replace('\n', '') # delete newline characters if there're any
    complete_text = complete_text.replace(" ' ", '') # delete if there're aposthropes hanging out alone
    complete_text = re.sub(r'[^\x00-\x7f]',r'', complete_text) # delete any non-ascii characters if there're any

    # now split the whole text into a list of words
    words_list = complete_text.split()
    words_list = removeSpaceFromWords(words_list)

    all_text = " ".join(words_list)

    return all_text

def makeProperForUnigrams(words_list):

    complete_text = " ".join(words_list)
    del words_list
    complete_text = complete_text.replace('\n', '') # delete newline characters if there're any
    complete_text = complete_text.replace(' *', '') # delete the end of sentence marks
    complete_text = complete_text.replace(" ' ", '') # delete if there're aposthropes hanging out alone
    complete_text = re.sub(r'[^\x00-\x7f]',r'', complete_text) # delete any non-ascii characters if there're any

    # now split the whole text into a list of words
    words_list = complete_text.split()
    words_list = removeSpaceFromWords(words_list)

    all_text = " ".join(words_list)

    return all_text

def removeSpaceFromWords(tlist):

    for wordi in range(len(tlist)):
        word = tlist[wordi]
        assert ' ' not in word:
        # if ' ' in word:
            # word = word.replace(' ', '')

        # tlist[wordi] = word
    return tlist
#--------------------------------------------------------------------------
# i_list = ['097', '100', '103', '106']#['098', '099', '101', '102', '104', '105', '107', '108', '110', '111']#['100', '103', '106', '109', '112']
# i_list = ['103']#['098', '099', '101', '102', '104', '105', '107', '108', '110', '111']#['100', '103', '106', '109', '112']
# i_list = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111' '112', '113' '114']
# i_list = ['097', '098', '099', '111', '112', '113']
# i_list = ['114']
processed_dir = '../../processed_data/'
# selected_congresses = range(43, 97)
for chamber in [ "Senate", "House"]: #
    # for i in selected_congresses:
    for i in [100]:
        # if chamber== "House" and i in [97, 100, 103, 106, 109, 112, 114]:
        #     continue
        fmt_congress = "%03d" % i
        print(f'Processsing {chamber} {i}')
        files_dem = getBasenames(glob.glob(f'{processed_dir}{chamber}_{fmt_congress}/d_*.txt'))
        files_rep = getBasenames(glob.glob(f'{processed_dir}{chamber}_{fmt_congress}/r_*.txt'))
        files_dem.sort()
        files_rep.sort()

        print(f"size dem = {len(files_dem)}, size_rep={len(files_rep)}")

        # select 3000 speeches from each set at random
        # random.seed(0)
        # random.shuffle(files_dem)
        # random.shuffle(files_rep)

        # d_files, r_files = saveTrainingTestSplit(files_dem, files_rep, i)
        # print(f"d_files={d_files}")
        # print(f"r_files={r_files}")
        # del files_dem, files_rep

        # selected_files = []
        # for split in ["train", "test", "validation"]:
        #     with open(f'{split}{i}.txt') as f:
        #         paths = f.readlines()
        #         selected_files += [path.replace('\n','') for path in paths]


        os.makedirs(f"{processed_dir}{chamber}{fmt_congress}_sentences", exist_ok=True)
        os.makedirs(f"{processed_dir}{chamber}{fmt_congress}_unigrams", exist_ok=True)

        # for file_ in selected_files:
        # for file_ in d_files + r_files:
        for file_ in files_dem + files_rep:
            if file_ == "..":
                continue

            fo = open(f'{processed_dir}{chamber}_{fmt_congress}/{file_}', 'r')
            text_ = fo.read()
            fo.close()
            text_ = cleanSpeechData(text_) # returns a list of words

            uni_text = makeProperForUnigrams(text_)
            sent_text = makeProperForSentences(text_)
            if len(uni_text.strip())==0:
                print(f"skipping empty speech: {text_}")
                continue
            if 'd_dale_bumpers_19871211_1000139461' in file_:
                print(f'processing file {file_}')
                print(f"uni text: {uni_text}")
                print(f"sent text: {sent_text}")

            fo = open(f'{processed_dir}{chamber}{fmt_congress}_sentences/{file_}', 'w')
            fo.write(sent_text)
            fo.close()

            fo = open(f'{processed_dir}{chamber}{fmt_congress}_unigrams/{file_}', 'w')
            fo.write(uni_text)
            fo.close()
        
