import numpy as np
import pickle
import json
from scipy import sparse

# for fmt_congress in ['97', '100', '103', '106', '109']: #, 
# for fmt_congress in []: #, 
for style in ['bayram' ]: # '' 'bayram'
	for chamber in [  "House"]: # "House", "Senate" , 'small_0'
		distinct_count_min = 3
		absolute_count_min = 7
		# a = list(range(44, 42, -1)) # 97, 43
		# for i in [  100]: #, 100,
		# for i in range(114, 42, -1):
		# for i in a:
		# for congress in range(43, 115):
		for congress in [ 97, 100, 103]: #, 100,
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
			for split in ["test"]:
				print(f"processing {fmt_congress}, {split}")
				# with open(f"splits/{split}{fmt_congress}.txt") as f:
				with open(f"splits/{chamber}{fmt_congress}_{style}_{split}.txt") as f:
					paths = f.readlines()
				for (i, path) in enumerate(paths):
						path = path.replace('\n', '')
						with open(f"../processed_data/{chamber}{fmt_congress}_unigrams/{path}") as f:
							lines = f.readlines()
							assert len(lines) > 0, f"empty file! {f}"
							speech = lines[0]
							wordcount = len(speech.split())
							if wordcount < 175:							print(f'{path}:\n"{speech}"\n')
