import numpy as np
import pickle
from scipy import sparse
from keras.preprocessing import sequence
import json

# for i_split in ['97', '100', '103', '106', '109']: #, 
# for i_split in []: #, 
for style in ['max_balanced_0' ]: # 'max_balanced_0' 
	for chamber in ["House"]: # "House", "Senate"  "House",
		distinct_count_min = 1 
		absolute_count_min = 1
		out_style = f'{style}_{distinct_count_min}_{absolute_count_min}'
		# a = list(range(44, 42, -1)) # 97, 43
		# for i in [  100]: #, 100,
		# for i in range(114, 42, -1):
		# for i in a:
		for congress in [ 97, 100, 103, 106, 109, 112, 114]: #,
		# for congress in range(97, 115): #, 100,
			fmt_congress = "%03d" % congress
			print(f'Processsing {chamber} {fmt_congress}')
			row_files = []
			absolute_counts = dict()
			number_of_speeches = dict()
			dictionary = dict()
			rev_dict = dict()
			# for split in ["train", "test", "validation"]:
			lengths = []
			for split in ["train", "test", "valid"]:
				print(f"processing {fmt_congress}, {split}")
				# with open(f"splits/{split}{i_split}.txt") as f:
				with open(f"splits/{chamber}{fmt_congress}_{style}_{split}.txt") as f:
					paths = f.readlines()
				print(f"found {len(paths)} paths")
				if split == "train":
					index = 1
					for path in paths:
						if "d_timothy_penny_19870128_1000004209.txt" in path:
							print("found the file")
						path = path.replace('\n', '')
						with open(f"../processed_data/{chamber}{fmt_congress}_sentences/{path}") as f:
							lines = f.readlines()
							assert len(lines) > 0, f"empty file: {path}, {lines}"
							speech = lines[0].split()
							lengths.append(len(speech))
						f.close()
						for word in set(speech):
							if word == "revenuesin":
								print(f"found the word")
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
					# print(f"count revenuesin: {absolute_counts['revenuesin']}")
					# print(f"distinct revenuesin: {number_of_speeches['revenuesin']}")
					print(f"computed dictionary of size {len(dictionary)}")
					rev_dict = {v:k for k,v in dictionary.items()}
					with open(f"matricies/dicts/{chamber}_{fmt_congress}_{out_style}_sequence.json", "w") as dict_file:
						json.dump(dictionary, dict_file)
				row_files += paths

			print(f"length stats: max={np.max(lengths)} min={np.min(lengths)} 50%={np.quantile(lengths, 0.5)} 75%={np.quantile(lengths, 0.75)} 95%={np.quantile(lengths, 0.95)} 99%={np.quantile(lengths, 0.99)}")
			# max_length = min(2048, np.quantile(lengths, 0.99))
			max_length = 1024
			print(f"using max_length={max_length}")
			sequences = []
			for (congress, path) in enumerate(row_files):
				path = path.replace('\n', '')
				with open(f"../processed_data/{chamber}{fmt_congress}_sentences/{path}") as f:
					lines = f.readlines()
					assert len(lines) > 0, f"empty file! {f}"
					speech = lines[0].split()
				sequences.append([dictionary[word] if word in dictionary else index for word in speech])
				# if len(sequences) % 1000 ==0:
				# 	print(sequences[-1])

			sequences = sequence.pad_sequences(sequences, maxlen=max_length, padding='pre', truncating='post')
			# print("sparse", sparse.csr_matrix(sequences[:10, :]))
			# np.savetxt(
			path = f"matricies/{chamber}_{fmt_congress}_{out_style}_sequence_matrix.txt"
			print(f"writing to {path}")
			sparse.save_npz(path, sparse.csr_matrix(sequences))
			# back = sparse.load_npz(path).toarray()
			# assert np.allclose(sequences, back)
			# with open(f"matricies/{chamber}_{i_split}_max_balanced_0_row_files.txt", 'w') as f:
			with open(f"matricies/{chamber}_{fmt_congress}_{out_style}_sequence_row_files.txt", 'w') as f:
				for path in row_files:
					f.write("%s"%path)

