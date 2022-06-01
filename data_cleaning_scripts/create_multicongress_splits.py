import random
import os
import glob

def getBasenames(filename_list):
    return [f.replace('../../','') for f in filename_list]

if __name__ == "__main__":
    # select 3000 speeches from each set at random
    seed = 0
    style = "max_balanced"
    random.seed(seed)
    processed_dir_no_dots = 'processed_data/'
    processed_dir = f'../../{processed_dir_no_dots}'
    bayram_congresses = [97, 100, 103, 106, 109, 112, 114]
    # selected_congresses = [97, 100, 103, 106, 109, 112, 114]#range(43, 115)
    selected_congresses = range(114, 115)  # [100]
    all_files = []
    selected_test_files = []
    selected_train_files = []
    selected_valid_files = []
    # selected_congresses = range(97, 115)
    for chamber in ["House"]: # , ,  , "Senate"
        for i in selected_congresses:
            fmt_congress = "%03d" % i
            print(f'Processsing {chamber} {i} {style}')
            files_dem = set(getBasenames(glob.glob(f'{processed_dir}{chamber}{fmt_congress}_unigrams/d_*.txt')))
            files_rep = set(getBasenames(glob.glob(f'{processed_dir}{chamber}{fmt_congress}_unigrams/r_*.txt')))

            current_files = files_dem.union(files_rep)
            # basename_to_current = {os.path.basename(f):f for f in current_files}
            existing_test = []
            existing_train = []
            existing_valid = []
            if chamber == "House" and i in bayram_congresses:
                with open(f'../splits/House{fmt_congress}_bayram_test.txt', 'r') as existing_test_f:
                    existing_test = existing_test_f.read().splitlines()
                assert len(existing_test) > 0
                print(f"found {len(existing_test)} exisitng test cases")
                
                with open(f'../splits/House{fmt_congress}_bayram_train.txt', 'r') as existing_train_f:
                    existing_train = existing_train_f.read().splitlines()
                assert len(existing_train) > 0
                print(f"found {len(existing_train)} exisitng train cases")

                with open(f'../splits/House{fmt_congress}_bayram_valid.txt', 'r') as existing_valid_f:
                    existing_valid = existing_valid_f.read().splitlines()
                assert len(existing_valid) > 0
                print(f"found {len(existing_valid)} exisitng valid cases")

                existing_test = [f"{processed_dir_no_dots}{chamber}{fmt_congress}_unigrams/{t}" for t in existing_test]
                existing_train = [f"{processed_dir_no_dots}{chamber}{fmt_congress}_unigrams/{t}" for t in existing_train]
                existing_valid = [f"{processed_dir_no_dots}{chamber}{fmt_congress}_unigrams/{t}" for t in existing_valid]

                for existing in [existing_train, existing_valid, existing_test]:
                    for current in existing:
                        assert current in current_files, f"{current}"
                        current_files.remove(current)
                        assert current not in current_files
                        basename = os.path.basename(current)
                        if basename.startswith("d_"):
                            assert current in files_dem
                            files_dem.remove(current)
                            assert current not in files_dem
                        elif basename.startswith("r_"):
                            assert current in files_rep
                            files_rep.remove(current)
                            assert current not in files_rep
                        else:
                            raise Exception(f"Unexpected path: '{current}'")
            print(f"len allfiles={len(current_files)}, len demfiles={len(files_dem)}, len repfiles={len(files_rep)},")
            current_files = list(current_files)
            current_files.sort()
            files_dem = list(files_dem)
            files_dem.sort()
            files_rep = list(files_rep)
            files_rep.sort()
            random.shuffle(files_dem)
            random.shuffle(files_rep)
            random.shuffle(current_files)
            if style == "small":
                num_valid = 500 - int(len(existing_valid) /2)
                num_test = 500 - int(len(existing_test) /2)
                num_train = 2000 - int(len(existing_train) /2)
            elif style == "max_balanced":
                n = min(len(files_dem), len(files_rep))
                num_valid = int(n * 0.15)  - int(len(existing_valid) /2)
                num_test = int(n * 0.15) - int(len(existing_test)  /2)
                num_train = int(n * 0.7) - int(len(existing_train) /2)
            elif style == "unbalanced":
                n = min(len(current_files))
                num_valid = int(n * 0.15) -len(existing_valid)
                num_test = int(n * 0.15) -len(existing_test)
                num_train = int(n * 0.7) -len(existing_train)
            else:
                raise Exception('unexpected style '+style)

            print(f"num_train={num_train}, num_valid={num_valid}, num_test={num_test}")
            train_stop = num_test + num_train
            valid_stop = train_stop + num_valid

            current_files += existing_test + existing_valid + existing_train
            all_files += current_files
            selected_test_files += existing_test
            selected_train_files += existing_train
            selected_valid_files += existing_valid
            if style != "unbalanced":
                selected_test_files += files_dem[:num_test] + files_rep[:num_test]
                selected_train_files += files_dem[num_test : train_stop] + files_rep[num_test : train_stop]
                selected_valid_files += files_dem[train_stop : valid_stop] + files_rep[train_stop : valid_stop]
            else:
                selected_test_files += current_files[:num_test]
                selected_train_files += current_files[num_test : train_stop]
                selected_valid_files += current_files[train_stop : valid_stop]

            # print(f"selected_valid_files_rep={len(selected_valid_files_rep)}, selected_valid_files_dem={len(selected_valid_files_rep)}, selected_train_files_rep={len(selected_train_files_rep)}, selected_train_files_dem={len(selected_train_files_dem)}, selected_test_files_dem={len(selected_test_files_dem)}, selected_test_files_rep={len(selected_test_files_rep)}")

    dataset = f"House114_114_{style}"
    fo_train = open(f'../splits/{dataset}_{seed}_train.txt', 'w')
    fo_test = open(f'../splits/{dataset}_{seed}_test.txt', 'w')
    fo_valid = open(f'../splits/{dataset}_{seed}_valid.txt', 'w')
    all_files.sort()
    selected_test_files = set(selected_test_files)
    selected_train_files = set(selected_train_files)
    selected_valid_files = set(selected_valid_files)
    print("writing...")
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
