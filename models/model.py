import numpy as np
from .common_fns import *
import json

class Model:
	def __init__(self, instance_name=None):
		self.instance_name = instance_name
		self.model = None
		self.grid = None

	def getClassifierParams(self, training_matrix, training_labels, validation_matrix,  validation_labels, dictionary):
		return None

	def name(self):
		raise Exception("Unimplemented!")

	def getSaveDirectory(self):
		return f"results/{self.name()}/"

	def preprocess(self, matrix):
		return applyLogNormalization(matrix)
		# return applyTFIDFNormalization(matrix)

	def fit(self, training_input, training_labels, validation_input, validation_labels):
		raise Exception("Unimplemented!")

	def predict(self, matrix):
		return self.model.predict(matrix)

	def use_gpu(self):
		return False

	def is_baseline(self):
		return False

	def get_json(self):
		if not self.model:
			raise Exception("Not trained!")
		to_json = getattr(self.model, "to_json", None)
		if callable(to_json):
			return to_json()
		get_params = getattr(self.model, "get_params", None)
		if callable(get_params):
			try:
				return json.dumps(get_params())
			except TypeError:
				print(f"Json TypeError save failed for {self.name()}")
				return ""
		raise Exception(f"Don't know how to save for {self.name()}")
		
	def is_h5(self):
		if not self.model:
			raise Exception("Not trained!")
		weights_op = getattr(self.model, "save_weights", None)
		return callable(weights_op)

	def is_sequence(self):
		return False

class SequenceModel(Model):
	def __init__(self, embedding_size, pretrained=None, trainable=False, instance_name=None):
		super().__init__(instance_name)
		self.embedding_size = embedding_size
		self.pretrained = pretrained
		self.trainable = trainable

	def is_sequence(self):
		return True
	
	def use_gpu(self):
		return True

	def preprocess(self, matrix):
		return matrix

	def load_glove_embeddings(self, model, training_matrix, dictionary):
		if model == 'glove':
			embedding_index = loadEmbeddings(f"../glove.6B/glove.6B.{self.embedding_size}d.txt")
		elif model == 'glove840':
			embedding_index = loadEmbeddings(f"../glove.840B.{self.embedding_size}d.txt")
		else:
			raise Exception(f"Unexpected model {model}")
		dictionary_size = np.max(training_matrix) + 2
		print(f"dictionary_size =  {dictionary_size}")
		embedding_matrix = np.zeros((dictionary_size, self.embedding_size))
		hits = 0
		misses = 0
		for word, i in dictionary.items():
			if word == '*':
				word = '.'
			vector = embedding_index.get(word)
			if vector is None:
				misses += 1
			elif vector.shape[0] != self.embedding_size:
				print(f"unexpected embedding size {vector.shape}")
				misses += 1
			else:
				hits += 1
				embedding_matrix[i, :] = vector
		print("Converted %d words (%d misses)" % (hits, misses))
		return embedding_matrix
