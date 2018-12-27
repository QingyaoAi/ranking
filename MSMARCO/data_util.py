import os,sys
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import tensorflow as tf
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
import numpy as np
import string

class MsMarcoData:
	settings = {}
	vocabulary = {}
	max_list_length = 0
	max_doc_length = 0
	max_query_length = 0
	pad_tensor = {}
	ps = PorterStemmer()
	is_sparse_feature = False

	def create_feature_columns(self):
		"""Returns the example feature columns."""
		doc_unigram_column = tf.feature_column.categorical_column_with_vocabulary_list(
						"doc_unigrams",
						vocabulary_list=self.vocabulary.keys(),
						num_oov_buckets=self.settings["num_oov_buckets"]
					)
		query_unigram_column = tf.feature_column.categorical_column_with_vocabulary_list(
						"query_unigrams",
						vocabulary_list=self.vocabulary.keys(),
						num_oov_buckets=self.settings["num_oov_buckets"]
					)
		doc_embed_column, query_embed_column = tf.feature_column.shared_embedding_columns(
				[doc_unigram_column, query_unigram_column],
				dimension=self.settings["embedding_size"],
				shared_embedding_collection_name="words")
		context_feature_columns = {"query_unigrams" : query_embed_column}
		example_feature_columns = {"doc_unigrams" : doc_embed_column}

		return context_feature_columns, example_feature_columns

	def _tokenize(self, line):
		return [self.ps.stem(w.lower()) for w in word_tokenize(line) if w not in string.punctuation]


	def __init__(self, data_json_file_path):
		self.settings = json.load(open(data_json_file_path))

		# Build vocabulary
		print('Start loading data')
		self.vocabulary = {}
		vocab_file = self.settings["WORKING_PATH"] + '/vocab_min_count_%d.txt' % self.settings["word_min_count"]
		if os.path.isfile(vocab_file):
			print('Load vocabulary')
			with open(vocab_file) as fin:
				for line in fin:
					arr = line.strip().split(' ')
					self.vocabulary[arr[0]] = arr[1:]
		else:
			print('Build vocabulary')
			if not os.path.exists(self.settings["WORKING_PATH"]):
				os.makedirs(self.settings["WORKING_PATH"])
			def _add_vocab_from_file(file_path):
				with open(file_path) as fin:
					for line in fin:
						arr = line.strip().split('\t')
						words = self._tokenize(arr[1])
						for word in words:
							if word not in self.vocabulary:
								self.vocabulary[word] = [0,0] #cf, df
							self.vocabulary[word][0] += 1 #cf
						for word in set(words):
							self.vocabulary[word][1] += 1 #df
			for key in ["COLLECTION_PATH", "TRAIN_QUERY_PATH", "DEV_QUERY_PATH", "TEST_QUERY_PATH"]:
				_add_vocab_from_file(self.settings[key])

			# remove words with low frequency
			words = self.vocabulary.keys()
			del_words = set()
			for w in words:
				if self.vocabulary[w][0] < self.settings['word_min_count']:
					del_words.add(w)
			for w in del_words:
				self.vocabulary.pop(w, None)
			with open(vocab_file, 'w') as fout:
				for w in self.vocabulary:
					fout.write('%s %d %d\n' % (w, self.vocabulary[w][0], self.vocabulary[w][1]))
		print('Vocabulary loading finished')

		# Load passages and build example features
		self.example_pid_feature_map = {}
		self.max_doc_length = 0
		with open(self.settings["COLLECTION_PATH"]) as fin:
			for line in fin:
				arr = line.strip().split('\t')
				pid = int(arr[0])
				words = self._tokenize(arr[1])
				if pid not in self.example_pid_feature_map:
					self.example_pid_feature_map[pid] = {}
				self.example_pid_feature_map[pid]["doc_unigrams"] = words
				if len(words) > self.max_doc_length:
					self.max_doc_length = len(words)
		

		# Load queries and build context features
		self.context_qid_feature_map = {}
		self.max_query_length = 0
		for key in ["TRAIN_QUERY_PATH", "DEV_QUERY_PATH", "TEST_QUERY_PATH"]:
			with open(self.settings[key]) as fin:
				for line in fin:
					arr = line.strip().split('\t')
					qid = int(arr[0])
					words = self._tokenize(arr[1])
					if qid not in self.context_qid_feature_map:
						self.context_qid_feature_map[qid] = {}
					self.context_qid_feature_map[qid]["query_unigrams"] = words
					if len(words) > self.max_query_length:
						self.max_query_length = len(words)

		tf.logging.info("Collection size {}".format(str(len(self.example_pid_feature_map))))
		tf.logging.info("Max doc length {}".format(str(self.max_doc_length)))
		tf.logging.info("Max query length {}".format(str(self.max_query_length)))
		print('Load finish')

	def load_sparse_feature_from_data(self, set_name, list_size = -1):
		"""Returns features and labels in numpy.array."""

		
		if not self.is_sparse_feature:
			for pid in self.example_pid_feature_map:
				for k in self.example_pid_feature_map[pid]:
					words = self.example_pid_feature_map[pid][k]
					self.example_pid_feature_map[pid][k] = sparse_tensor_lib.SparseTensor(
						indices=[[i] for i in range(len(words))],
						values=words,
						dense_shape=[self.max_doc_length])
			self.pad_tensor['doc_unigrams'] = sparse_tensor_lib.SparseTensor(
						indices=[[0]],
						values=[','],
						dense_shape=[self.max_doc_length])
			
			for qid in self.context_qid_feature_map:
				for k in self.context_qid_feature_map[qid]:
					words = self.context_qid_feature_map[qid][k]
					self.context_qid_feature_map[qid][k] = sparse_tensor_lib.SparseTensor(
						indices=[[i] for i in range(len(words))],
						values=words,
						dense_shape=[self.max_query_length])
			self.pad_tensor['query_unigrams'] = sparse_tensor_lib.SparseTensor(
						indices=[[0]],
						values=[','],
						dense_shape=[self.max_query_length])
			self.is_sparse_feature = True
		


		def _parse_line(line):
			"""Parses a single line in LibSVM format."""
			tokens = line.strip().split()
			assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
			qid = int(tokens[0])
			pid = int(tokens[1])
			return qid, pid

		tf.logging.info("Loading data from {}".format(set_name))

		# Read labels
		qrel_map = {}
		with open(self.settings["%s_QRELS_FILE" % set_name]) as fin:
			for line in fin:
				qid, pid = _parse_line(line)
				if qid not in qrel_map:
					qrel_map[qid] = set()
				qrel_map[qid].add(pid)
		
		# Read data
		qid_to_doc = {} # The list of docs seen so far for a query.
		max_list_length = 0
		total_docs = 0
		with open(self.settings["%s_DATA_PATH" % set_name]) as fin:
			for line in fin:
				for line in fin:
					qid, pid = _parse_line(line)
					total_docs += 1
					label = 0
					if qid in qrel_map and pid in qrel_map[qid]: 
						label = 1
					if qid not in qid_to_doc:
						qid_to_doc[qid] = []
					qid_to_doc[qid].append((pid, label))
					if len(qid_to_doc[qid]) > max_list_length:
						max_list_length = len(qid_to_doc[qid])
		list_size = list_size if list_size > 0 else max_list_length
		if max_list_length > self.max_list_length:
			self.max_list_length = max_list_length	

		# Build feature map
		context_feature_columns, example_feature_columns = self.create_feature_columns()
		feature_map = {k: [] for k in context_feature_columns}
		for k in example_feature_columns:
			feature_map[k] = []
		label_list = []
		discarded_docs = 0
		# Each feature is mapped an array with [num_queries, list_size, 1]. Label has
		# a shape of [num_queries, list_size]. We use list for each of them due to the
		# unknown number of quries.
		for qid in qid_to_doc:
			label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
			# build context feature, which is shared among the whole list
			for k in context_feature_columns:
				#feature_map[k][-1].append(self.context_qid_feature_map[qid][k])
				feature_map[k].append(self.context_qid_feature_map[qid][k])
			# build example feature, which is arranged in a list
			for k in example_feature_columns:
				feature_map[k].append([self.pad_tensor[k] for _ in range(list_size)])
				index = 0
				for pid, label in qid_to_doc[qid]:
					if index < list_size:
						for k in example_feature_columns:
							feature_map[k][-1][index] = self.example_pid_feature_map[pid][k]
							#feature_map[k][-1].append(self.example_pid_feature_map[pid][k])
						label_list[-1][index] = label
					else:
						discarded_docs += 1
					index += 1
					total_docs += 1

		tf.logging.info("Number of queries: {}".format(len(qid_to_doc)))
		tf.logging.info("Number of documents in total: {}".format(total_docs))
		tf.logging.info("Number of documents discarded: {}".format(discarded_docs))

		'''
		indices = []
		values = []
		for i in range(len(feature_map['doc_unigrams'])):
			for j in range(len(feature_map['doc_unigrams'][i])):
				words = feature_map['doc_unigrams'][i][j]
				for k in range(len(words)):
					indices.append([i,j,k])
					values.append(words[k])
		feature_map['doc_unigrams'] = sparse_tensor_lib.SparseTensor(
					indices=indices,
					values=values,
					dense_shape=[len(feature_map['doc_unigrams']), list_size, self.max_doc_length])
		indices = []
		values = []
		for i in range(len(feature_map['query_unigrams'])):
			for j in range(len(feature_map['query_unigrams'][i])):
				words = feature_map['query_unigrams'][i][j]
				for k in range(len(words)):
					indices.append([i,j,k])
					values.append(words[k])
		feature_map['query_unigrams'] = sparse_tensor_lib.SparseTensor(
					indices=indices,
					values=values,
					dense_shape=[len(feature_map['query_unigrams']), list_size, self.max_query_length])
		'''	
		'''
			
		# pad list
		for k in feature_map:
			for l in feature_map[k]:
				if len(l) < list_size:
					l.extend([self.pad_tensor[k] for _ in range(list_size - len(l))])
		'''

		#print('feature shape')
		#for k in feature_map:
		#	print(feature_map[k].shape.as_list())
		#print(np.array(label_list).shape)
		return feature_map, np.array(label_list)

	def load_feature_from_data(self, set_name, list_size = -1):
		"""Returns features and labels in numpy.array."""

		def _parse_line(line):
			"""Parses a single line in LibSVM format."""
			tokens = line.strip().split()
			assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
			qid = int(tokens[0])
			pid = int(tokens[1])
			return qid, pid

		tf.logging.info("Loading data from {}".format(set_name))

		# Read labels
		qrel_map = {}
		with open(self.settings["%s_QRELS_FILE" % set_name]) as fin:
			for line in fin:
				qid, pid = _parse_line(line)
				if qid not in qrel_map:
					qrel_map[qid] = set()
				qrel_map[qid].add(pid)
		
		# Read data
		qid_to_doc = {} # The list of docs seen so far for a query.
		max_list_length = 0
		total_docs = 0
		with open(self.settings["%s_DATA_PATH" % set_name]) as fin:
			for line in fin:
				for line in fin:
					qid, pid = _parse_line(line)
					total_docs += 1
					label = 0
					if qid in qrel_map and pid in qrel_map[qid]: 
						label = 1
					if qid not in qid_to_doc:
						qid_to_doc[qid] = []
					qid_to_doc[qid].append((pid, label))
					if len(qid_to_doc[qid]) > max_list_length:
						max_list_length = len(qid_to_doc[qid])
		list_size = list_size if list_size > 0 else max_list_length
		if max_list_length > self.max_list_length:
			self.max_list_length = max_list_length	

		# Build feature map
		context_feature_columns, example_feature_columns = self.create_feature_columns()
		feature_map = {k: [] for k in context_feature_columns}
		for k in example_feature_columns:
			feature_map[k] = []
		label_list = []
		discarded_docs = 0
		# Each feature is mapped an array with [num_queries, list_size, 1]. Label has
		# a shape of [num_queries, list_size]. We use list for each of them due to the
		# unknown number of quries.
		for qid in qid_to_doc:
			label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
			# build context feature, which is shared among the whole list
			for k in context_feature_columns:
				if len(self.context_qid_feature_map[qid][k]) < self.max_query_length:
					padding = ['$' for _ in range(self.max_query_length - len(self.context_qid_feature_map[qid][k]))]
					self.context_qid_feature_map[qid][k].extend(padding)
				feature_map[k].append(self.context_qid_feature_map[qid][k])
			# build example feature, which is arranged in a list
			for k in example_feature_columns:
				feature_map[k].append([])
				index = 0
				for pid, label in qid_to_doc[qid]:
					if index < list_size:
						if len(self.example_pid_feature_map[pid][k]) < self.max_doc_length:
							padding = ['$' for _ in range(self.max_doc_length - len(self.example_pid_feature_map[pid][k]))]
							self.example_pid_feature_map[pid][k].extend(padding)
						#feature_map[k][-1][index] = self.example_pid_feature_map[pid][k]
						feature_map[k][-1].append(self.example_pid_feature_map[pid][k])
						label_list[-1][index] = label
					else:
						discarded_docs += 1
					index += 1
					total_docs += 1

		# padding
		for k in example_feature_columns:
			for l in feature_map[k]:
				if len(l) < list_size:
					l.extend([['$' for x in range(len(l[0]))]  for _ in range(list_size - len(l))])
				#for x in l:
				#	print('%d, %d, %d' % (len(feature_map[k]), len(l), len(x)))


		tf.logging.info("Number of queries: {}".format(len(qid_to_doc)))
		tf.logging.info("Number of documents in total: {}".format(total_docs))
		tf.logging.info("Number of documents discarded: {}".format(discarded_docs))

		#for k in feature_map:
		#	feature_map[k] = np.chararray(feature_map[k])
		#print(np.array(label_list).shape)
		return feature_map, np.array(label_list)






