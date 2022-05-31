import numpy as np
import pickle
from scipy import sparse
import json

# for fmt_congress in ['97', '100', '103', '106', '109']: #,
# for fmt_congress in []: #,
n_range = [1,2,3]
for style in ['max_balanced_0']:  # 'bayram'
    for chamber in ["House"]:  # "House", , "Senate"
        distinct_count_min = 10 
        absolute_count_min = 50
        out_style = f'{style}_{distinct_count_min}_{absolute_count_min}'
        # for i in range(100, 115): # stopped 2gram at 100
        for i in range(97, 115):
        # for i in [ 97, 100, 103, 106, 109, 112, 114]: #, 100,
            # if i in [ 103, 106, 109, 112, 114] and chamber =="House":
            # 	continue
            fmt_congress = "%03d" % i
            print(f'Processsing {chamber} {fmt_congress}')
            row_files = []
            absolute_counts = dict()
            number_of_speeches = dict()
            dictionary = dict()
            # for split in ["train", "test", "validation"]:
            for split in ["train", "test", "valid"]:
                print(f"processing {fmt_congress}, {split}")
                # with open(f"splits/{split}{fmt_congress}.txt") as f:
                with open(f"splits/{chamber}{fmt_congress}_{style}_{split}.txt") as f:
                    paths = f.readlines()
                print(f"found {len(paths)} paths")
                if split == "train":
                    index = len(dictionary)
                    for path in paths:
                        path = path.replace('\n', '')
                        with open(f"../processed_data/{chamber}{fmt_congress}_sentences/{path}") as f:
                            lines = f.readlines()
                            assert len(
                                lines) > 0, f"empty file: {path}, {lines}"
                        tokens = []
                        words = lines[0].split()
                        for n in n_range:
                            if len(words) < n:
                                continue
                            for start in range(len(words)-n+1):
                                tokens.append(
                                    " ".join(words[start:start+n]))
                        for token in set(tokens):
                            # if token in dictionary:
                            #     continue
                            if token not in number_of_speeches:
                                number_of_speeches[token] = 1
                            else:
                                number_of_speeches[token] += 1
                        for token in tokens:
                            # if token in dictionary:
                            #     continue
                            if token not in absolute_counts:
                                absolute_counts[token] = 0
                            absolute_counts[token] += 1
                            if token not in dictionary and absolute_counts[token] >= absolute_count_min and number_of_speeches[token] >= distinct_count_min:
                                dictionary[token] = index
                                index += 1
                    print(f"computed dictionary of size {len(dictionary)}")
                    with open(f"matricies/dicts/{chamber}_{fmt_congress}_{out_style}.json", "w") as dict_file:
                        json.dump(dictionary, dict_file)
                    stats = dict()
                    for token in dictionary.keys():
                        stats[token] = {
                            'total': absolute_counts[token], 'distinct': number_of_speeches[token]}
                    # rev_dict = {v:k for k,v in dictionary.items()}
                    with open(f"matricies/stats/{chamber}_{fmt_congress}_{n}gram_{out_style}_counts.json", "w") as dict_file:
                        json.dump(stats, dict_file)
                row_files += paths
            result = sparse.lil_matrix((len(row_files), len(dictionary)), dtype=np.float64)
            for (i, path) in enumerate(row_files):
                newrow = np.zeros((result.shape[1]))
                path = path.replace('\n', '')
                with open(f"../processed_data/{chamber}{fmt_congress}_sentences/{path}") as f:
                    lines = f.readlines()
                    assert len(lines) > 0, f"empty file! {f}"
                tokens = []
                words = lines[0].split()
                for n in n_range:
                    if len(words)< n:
                        continue
                    for start in range(len(words)-n+1):
                        tokens.append(" ".join(words[start:start+n]))
                for token in tokens:
                    if token in dictionary:
                        newrow[dictionary[token]] += 1
                result[i, :] = newrow
            # np.savetxt(
            sparse.save_npz(
                f"matricies/{chamber}_{fmt_congress}_{n}gram_{out_style}_matrix.txt", sparse.csr_matrix(result))
            # sparse.save_npz(f"matricies/{fmt_congress}_matrix.txt", sparse.csr_matrix(result))
            with open(f"matricies/{chamber}_{fmt_congress}_{n}gram_{out_style}_row_files.txt", 'w') as f:
                # with open(f"matricies/{fmt_congress}_row_files.txt", 'w') as f:
                for path in row_files:
                    f.write("%s" % path)
