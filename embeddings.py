import numpy as np

glove_path = "../glove.6B/glove.6B.50d.txt"

embeddings_index = {}
with open(glove_path) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        # if word == ".":
        #     print("found period")
        # if not word.isalpha():
        #     print(f"non-alpha '{word}'")
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))