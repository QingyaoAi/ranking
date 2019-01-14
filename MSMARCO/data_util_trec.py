import os,sys
import json
import tensorflow as tf
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
import numpy as np
import string
import gzip
from nltk.stem import PorterStemmer

class MsMarcoData:
	settings = None
	vocabulary = None
	vocab_idxs = None
	max_list_length = 0
	max_doc_length = 0
	max_query_length = 0
	pad_tensor = None
	is_sparse_feature = False
	ps = PorterStemmer()
	SET_LIST = ['train', 'dev', 'eval']

	def create_feature_columns(self):
		"""Returns the example feature columns."""
		
		'''
		doc_unigram_column = tf.feature_column.categorical_column_with_vocabulary_list(
						"doc_unigrams",
						vocabulary_list=self.vocabulary.keys(),
						default_value=-1
						#num_oov_buckets=self.settings["num_oov_buckets"]
					)
		query_unigram_column = tf.feature_column.categorical_column_with_vocabulary_list(
						"query_unigrams",
						vocabulary_list=self.vocabulary.keys(),
						default_value=-1
						#num_oov_buckets=self.settings["num_oov_buckets"]
					)
		'''
		vocabulary_size = len(self.vocabulary.keys())
		doc_unigram_column = tf.feature_column.categorical_column_with_identity(
								key="doc_unigrams", num_buckets=vocabulary_size)
		query_unigram_column = tf.feature_column.categorical_column_with_identity(
								key="query_unigrams", num_buckets=vocabulary_size)

		doc_embed_column, query_embed_column = tf.feature_column.shared_embedding_columns(
				[doc_unigram_column, query_unigram_column],
				dimension=self.settings["embedding_size"],
				shared_embedding_collection_name="words")
		context_feature_columns = {"query_unigrams" : query_embed_column}
		example_feature_columns = {"doc_unigrams" : doc_embed_column}

		return context_feature_columns, example_feature_columns

	def _tokenize(self, line):
		text = line.strip()
		for ch in string.punctuation:
			text = text.replace(ch, '')
		return [self.ps.stem(w.lower()) for w in text.split(' ')]

	def __init__(self, data_json_file_path):
		self.settings = json.load(open(data_json_file_path))

		# Build vocabulary
		print('Start loading data')
		self.vocabulary = {}
		self.vocab_idxs = {}
		vocab_file = self.settings["WORKING_PATH"] + '/vocab_min_count_%d.txt' % self.settings["word_min_count"]
		if os.path.isfile(vocab_file):
			print('Load vocabulary')
			with open(vocab_file) as fin:
				idx = 0
				for line in fin:
					arr = line.strip().split(' ')
					self.vocabulary[arr[0]] = arr[1:]
					self.vocab_idxs[arr[0]] = idx
					idx += 1
					if idx != len(self.vocabulary):
						print(idx)
		else:
			print('Build vocabulary')
			if not os.path.exists(self.settings["WORKING_PATH"]):
				os.makedirs(self.settings["WORKING_PATH"])
			
			def _add_words_to_vocab(raw_word_list):
				word_list = [w.strip() for w in raw_word_list]
				word_list = [w.lower() for w in word_list if len(w) > 0]
				for word in word_list:
					if word not in self.vocabulary:
						self.vocabulary[word] = [0,0] #cf, df
					self.vocabulary[word][0] += 1 #cf
				for word in set(word_list):
					self.vocabulary[word][1] += 1 #df
			# add collection words		
			with open(self.settings["COLLECTION_PATH"] + '/corpus_text.txt') as text_fin:
				for line in text_fin:
					words = self._tokenize(line)
					_add_words_to_vocab(words)

			# add query words
			query_files = { x: self.settings['QUERY_PATH'] + '/queries.%s.json' %x for x in self.SET_LIST}
			for set_name in self.SET_LIST:
				if not os.path.exists(query_files[set_name]):
					continue
				with open(query_files[set_name]) as fin:
					data = json.load(fin)
					for query in data['queries']:
						qid = int(query['number'])
						query_text = query['text'].replace('#combine( ', '').replace(' )', '')
						_add_words_to_vocab(self._tokenize(query_text))

			# remove words with low frequency
			words = self.vocabulary.keys()
			del_words = set()
			for w in words:
				if self.vocabulary[w][0] < self.settings['word_min_count']:
					del_words.add(w)
			for w in del_words:
				self.vocabulary.pop(w, None)
			with open(vocab_file, 'w') as fout:
				idx = 0
				for w in self.vocabulary:
					self.vocab_idxs[w] = idx
					fout.write('%s %d %d\n' % (w, self.vocabulary[w][0], self.vocabulary[w][1]))
					idx += 1
		print('Vocabulary loading finished, size %d' % (len(self.vocabulary)))

		# Load passages and build example features
		doc_file = self.settings["WORKING_PATH"] + '/collection_min_count_%d.txt.gz' % self.settings["word_min_count"]
		self.example_pid_feature_map = {}
		self.max_doc_length = 0
		self.context_qid_feature_map = {}
		self.max_query_length = 0
		if os.path.isfile(doc_file): # if the processed collection exists, read it
			print('Load passage features...')
			with gzip.open(doc_file, 'rt') as fin:
				for line in fin:
					arr = line.strip().split('\t')
					pid = int(arr[0])
					words = []
					if len(arr) > 1:
						words = [int(x) for x in arr[1].split(' ')]
					if pid not in self.example_pid_feature_map:
						self.example_pid_feature_map[pid] = {}
					self.example_pid_feature_map[pid]["doc_unigrams"] = words
					if len(words) > self.max_doc_length:
						self.max_doc_length = len(words)
					if len(self.example_pid_feature_map) % 10000 == 0:
						print('Read %d docs' % len(self.example_pid_feature_map))

		else: # if the collection hasn't been processed yet, process it and store the file.
			# load raw collection
			print('Create passage features...')
			def _get_word_idxs(words):
				word_idxs = []
				for i in range(len(words)):
					if words[i] in self.vocab_idxs:
						word_idxs.append(self.vocab_idxs[words[i]])
				return word_idxs

			pid_list = []
			with open(self.settings["COLLECTION_PATH"] + '/corpus_text.txt') as text_fin:
				with open(self.settings["COLLECTION_PATH"] + 'corpus_id.txt') as id_fin:
					for line in id_fin:
						text = text_fin.readline().strip()
						pid = int(line.strip())
						words = self._tokenize(text)
						pid_list.append(pid)
						if pid not in self.example_pid_feature_map:
							self.example_pid_feature_map[pid] = {}
						word_idxs = _get_word_idxs(words)
						self.example_pid_feature_map[pid]["doc_unigrams"] = word_idxs
						if len(word_idxs) > self.max_doc_length:
							self.max_doc_length = len(word_idxs)
						if len(self.example_pid_feature_map) % 10000 == 0:
							print('Read %d docs' % len(self.example_pid_feature_map))
			# write processed collection
			with gzip.open(doc_file, 'wt') as fout:
				for pid in pid_list:
					word_idxs = self.example_pid_feature_map[pid]["doc_unigrams"]
					fout.write('%d\t%s\n' % (pid, ' '.join([str(x) for x in word_idxs])))
		
		for set_name in self.SET_LIST:
			print('Read %s queries' % set_name)
			query_file = self.settings["WORKING_PATH"] + '/%s_QUERY_min_count_%d.txt.gz' % (set_name, self.settings["word_min_count"])
			if os.path.isfile(query_file):
				with gzip.open(query_file, 'rt') as fin:
					for line in fin:
						arr = line.strip().split('\t')
						qid = int(arr[0])
						words = []
						if len(arr) > 1:
							words = [int(x) for x in arr[1].split(' ')]
						if qid not in self.context_qid_feature_map:
							self.context_qid_feature_map[qid] = {}
						self.context_qid_feature_map[qid]["query_unigrams"] = words
						if len(words) > self.max_query_length:
							self.max_query_length = len(words)
			else:
				# Load queries and build context features
				print('Create %s query files' % set_name)
				qid_list = []
				with open(self.settings['QUERY_PATH'] + '/queries.%s.json' % set_name) as fin:
					data = json.load(fin)
					for query in data['queries']:
						qid = int(query['number'])
						query_text = query['text'].replace('#combine( ', '').replace(' )', '')
						words = self._tokenize(query_text)
						qid_list.append(qid)
						if qid not in self.context_qid_feature_map:
							self.context_qid_feature_map[qid] = {}
						word_idxs = _get_word_idxs(words)
						self.context_qid_feature_map[qid]["query_unigrams"] = word_idxs
						if len(word_idxs) > self.max_query_length:
							self.max_query_length = len(word_idxs)
				# write processed collection
				with gzip.open(query_file, 'wt') as fout:
					for qid in qid_list:
						word_idxs = self.context_qid_feature_map[qid]["query_unigrams"]
						fout.write('%d\t%s\n' % (qid, " ".join([str(x) for x in word_idxs])))

		tf.logging.info("Collection size {}".format(str(len(self.example_pid_feature_map))))
		tf.logging.info("Max doc length {}".format(str(self.max_doc_length)))
		tf.logging.info("Max query length {}".format(str(self.max_query_length)))
		print('Load finish')

		#print(self.context_qid_feature_map[524699]["query_unigrams"])
		#print(self.example_pid_feature_map[10009]["doc_unigrams"])


	def load_feature_from_data(self, set_name, list_size = -1):
		"""Returns features and labels in numpy.array."""

		tf.logging.info("Loading data from {}".format(set_name))

		# Read labels
		qrel_map = {}
		with open(self.settings['QRELS_PATH'] + '%s.qrels' % set_name) as fin:
			for line in fin:
				arr = line.strip().split(' ')
				qid = int(arr[0])
				pid = int(arr[2])
				label = int(arr[3])
				if qid not in qrel_map:
					qrel_map[qid] = set()
				if label > 0:
					qrel_map[qid].add(pid)
		
		# Read data
		qid_to_doc = {} # The list of docs seen so far for a query.
		max_list_length = 0
		total_docs = 0
		def _create_line_parser(data_type):
			if data_type == 'pair':
				def pair_line_parser(line):
					arr = line.strip().split('\t')
					qid = int(arr[0])
					pid = int(arr[1])
					return qid, pid
				return pair_line_parser
			elif data_type == 'trec_ranklist':
				def trec_ranklist_line_parser(line):
					arr = line.strip().split(' ')
					qid = int(arr[0])
					pid = int(arr[2])
					return qid, pid
				return trec_ranklist_line_parser

		line_parser = _create_line_parser(self.settings["%s_data_path" % set_name]['type'])
		with open(self.settings["%s_data_path" % set_name]['path']) as fin:
			for line in fin:
				qid, pid = line_parser(line)
				if qid not in self.context_qid_feature_map or pid not in self.example_pid_feature_map:
					continue
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
			#label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
			label_list.append(np.zeros([list_size], dtype=np.float32))
			# build context feature, which is shared among the whole list
			#if qid not in self.context_qid_feature_map:
			#	continue
			for k in context_feature_columns:
				if len(self.context_qid_feature_map[qid][k]) < self.max_query_length:
					padding = [-1 for _ in range(self.max_query_length - len(self.context_qid_feature_map[qid][k]))]
					self.context_qid_feature_map[qid][k].extend(padding)
				feature_map[k].append(self.context_qid_feature_map[qid][k])
			# build example feature, which is arranged in a list
			for k in example_feature_columns:
				feature_map[k].append([])
				index = 0
				for pid, label in qid_to_doc[qid]:
					#if pid not in self.example_pid_feature_map:
					#	continue
					if index < list_size:
						if len(self.example_pid_feature_map[pid][k]) < self.max_doc_length:
							padding = [-1 for _ in range(self.max_doc_length - len(self.example_pid_feature_map[pid][k]))]
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
					l.extend([[-1 for x in range(len(l[0]))] for _ in range(list_size - len(l))])
				#for x in l:
				#	print('%d, %d, %d' % (len(feature_map[k]), len(l), len(x)))


		tf.logging.info("Number of queries: {}".format(len(qid_to_doc)))
		tf.logging.info("Number of documents in total: {}".format(total_docs))
		tf.logging.info("Number of documents discarded: {}".format(discarded_docs))

		for k in feature_map:
			feature_map[k] = np.array(feature_map[k])
		#print(np.array(label_list).shape)
		#print(label_list)
		return feature_map, np.array(label_list)




