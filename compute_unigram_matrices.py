import numpy as np
import pickle
import json
from scipy import sparse

# for fmt_congress in ['97', '100', '103', '106', '109']: #, 
# for fmt_congress in []: #, 
for style in ['max_balanced_0']: # '' 'bayram'
	for chamber in [ "House", "Senate" ]: # "House", "Senate"
		distinct_count_min = 10
		absolute_count_min = 50
		out_style = f'{style}_{distinct_count_min}_{absolute_count_min}'
		# a = list(range(44, 42, -1)) # 97, 43
		# for i in [  100]: #, 100,
		# for i in range(114, 42, -1):
		# for i in a:
		for congress in range(97, 115):
		# for congress in [ 97, 100, 103, 106, 109, 112, 114]: #, 100,
			# if i in [ 103, 106, 109, 112, 114] and chamber =="House":
			# 	continue
			fmt_congress = "%03d" % congress
			print(f'Processsing {chamber} {fmt_congress}')
			row_files = []
			lengths = []
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
					index = 0
					for path in paths:
						path = path.replace('\n', '')
						with open(f"../processed_data/{chamber}{fmt_congress}_unigrams/{path}") as f:
							lines = f.readlines()
							assert len(lines) > 0, f"empty file: {path}, {lines}"
							speech = lines[0].split()
						lengths.append(len(speech))
						for word in set(speech):
							if word in dictionary:
								continue
							if word not in number_of_speeches:
								number_of_speeches[word] = 1
							else:
								number_of_speeches[word] += 1
						for word in speech:
							if word in dictionary:
								continue
							if word not in absolute_counts:
								absolute_counts[word] = 0
							absolute_counts[word] += 1
							if absolute_counts[word] >= absolute_count_min and number_of_speeches[word] >= distinct_count_min:
								dictionary[word] = index
								index += 1
					print(f"computed dictionary of size {len(dictionary)}")
					quantiles = np.quantile(lengths, np.arange(0, 1.1, 0.1))
					print(f"speech length quantiles", quantiles)
					with open(f"matricies/dicts/{chamber}_{fmt_congress}_{out_style}.json", "w") as dict_file:
						json.dump(dictionary, dict_file)
					np.savetxt(f"matricies/lengths/{chamber}_{fmt_congress}_{out_style}.txt", quantiles)
					# rev_dict = {v:k for k,v in dictionary.items()}
					# with open("dict.pkl", "wb") as dict_file:
					# 	pickle.dump(dictionary, dict_file)
				row_files += paths
			result = np.zeros((len(row_files), len(dictionary)))
			for (i, path) in enumerate(row_files):
				path = path.replace('\n', '')
				with open(f"../processed_data/{chamber}{fmt_congress}_unigrams/{path}") as f:
					lines = f.readlines()
					assert len(lines) > 0, f"empty file! {f}"
					speech = lines[0].split()
				for word in speech:
					if word in dictionary:
						result[i, dictionary[word]] += 1
			# np.savetxt(
			# sparse.save_npz(f"matricies/{chamber}_{fmt_congress}_max_balanced_0_matrix.txt", sparse.csr_matrix(result))
			sparse.save_npz(f"matricies/{chamber}_{fmt_congress}_{out_style}_matrix.txt", sparse.csr_matrix(result))
			# with open(f"matricies/{chamber}_{fmt_congress}_max_balanced_0_row_files.txt", 'w') as f:
			with open(f"matricies/{chamber}_{fmt_congress}_{out_style}_row_files.txt", 'w') as f:
				for path in row_files:
					f.write("%s"%path)

