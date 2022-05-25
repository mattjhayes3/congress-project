import random
import os
import glob

def getBasenames(filename_list):
    return [os.path.basename(f) for f in filename_list]

if __name__ == "__main__":
    # select 3000 speeches from each set at random
    seed = 0
    style = "max_balanced"
    random.seed(seed)
    processed_dir = '../../processed_data/'
    bayram_congresses = [97, 100, 103, 106, 109, 112, 114]
    # selected_congresses = [97, 100, 103, 106, 109, 112, 114]#range(43, 115)
    selected_congresses = [44]#range(43, 115)  # [100]
    # selected_congresses = range(97, 115)
    for chamber in ["House"]: # , , 
        for i in selected_congresses:
            fmt_congress = "%03d" % i
            print(f'Processsing {chamber} {i} {style}')
            files_dem = getBasenames(glob.glob(f'{processed_dir}{chamber}{fmt_congress}_unigrams/d_*.txt'))
            files_rep = getBasenames(glob.glob(f'{processed_dir}{chamber}{fmt_congress}_unigrams/r_*.txt'))
            files_dem.sort()
            files_rep.sort()            
            all_files = files_dem + files_rep
            all_files.sort()
            existing_test = []
            if chamber == "House" and i in bayram_congresses:
                with open(f'../splits/test{fmt_congress}.txt', 'r') as existing_test_f:
                    existing_test = existing_test_f.read().splitlines()
                assert len(existing_test) > 0
                print(f"found {len(existing_test)} exisitng test cases")
                for t in existing_test:
                    assert t in all_files, t
                    all_files.remove(t)
                    assert t not in all_files
                    if t.startswith("d_"):
                        assert t in files_dem
                        files_dem.remove(t)
                        assert t not in files_dem
                    elif t.startswith("r_"):
                        assert t in files_rep
                        files_rep.remove(t)
                        assert t not in files_rep
                    else:
                        raise Exception(f"Unexpected path: '{t}'")
            # print(f"all files: {all_files}")
            print(f"len allfiles={len(all_files)}, len depfiles={len(files_dem)}, len repfiles={len(files_rep)},")
            random.shuffle(files_dem)
            random.shuffle(files_rep)
            random.shuffle(all_files)
            if style == "small":
                num_valid = 500
                num_test = 500 - int(len(existing_test) /2)
                num_train = 2000
            elif style == "max_balanced":
                n = min(len(files_dem), len(files_rep))
                num_valid = int(n * 0.15)
                num_test = int(n * 0.15) - int(len(existing_test)  /2)
                num_train = int(n * 0.7)
            elif style == "unbalanced":
                n = min(len(all_files))
                num_valid = int(n * 0.15)
                num_test = int(n * 0.15) -len(existing_test)
                num_train = int(n * 0.7)
            else:
                raise Exception('unexpected style '+style)

            print(f"num_train={num_train}, num_valid={num_valid}, num_test={num_test}")
            train_stop = num_test + num_train
            valid_stop = train_stop + num_valid

            selected_test_files = [] + existing_test
            # selected_test_files_dem = []
            # selected_test_files_rep = []
            selected_train_files = []
            # selected_train_files_dem = []
            # selected_train_files_rep = []
            selected_valid_files = []
            # selected_valid_files_dem = []
            # selected_valid_files_rep = []
            if style != "unbalanced":
                selected_test_files += files_dem[:num_test] + files_rep[:num_test]
                selected_train_files += files_dem[num_test : train_stop] + files_rep[num_test : train_stop]
                selected_valid_files += files_dem[train_stop : valid_stop] + files_rep[train_stop : valid_stop]
            else:
                selected_test_files += all_files[:num_test]
                selected_train_files += all_files[num_test : train_stop]
                selected_valid_files += all_files[train_stop : valid_stop]

            # print(f"selected_valid_files_rep={len(selected_valid_files_rep)}, selected_valid_files_dem={len(selected_valid_files_rep)}, selected_train_files_rep={len(selected_train_files_rep)}, selected_train_files_dem={len(selected_train_files_dem)}, selected_test_files_dem={len(selected_test_files_dem)}, selected_test_files_rep={len(selected_test_files_rep)}")

            dataset = f"{chamber}{fmt_congress}_{style}"
            fo_train = open(f'../splits/{dataset}_{seed}_train.txt', 'w')
            fo_test = open(f'../splits/{dataset}_{seed}_test.txt', 'w')
            fo_valid = open(f'../splits/{dataset}_{seed}_valid.txt', 'w')
            all_files += existing_test
            all_files.sort()
            for file_ in all_files:
                if file_ in selected_test_files:
                    fo_test.write(file_ + '\n')
                if file_ in selected_train_files:
                    fo_train.write(file_ + '\n')
                if file_ in selected_valid_files:
                    fo_valid.write(file_ + '\n')

            fo_train.close()
            fo_test.close()
            fo_valid.close()

            # dem_files = selected_train_files_dem+selected_test_files_dem+selected_valid_files_dem
            # rep_files = selected_train_files_rep+selected_test_files_rep+selected_valid_files_rep
            print(f"num_train={len(selected_train_files)}, num_valid={len(selected_valid_files)}, num_test ={len(selected_test_files)}")
