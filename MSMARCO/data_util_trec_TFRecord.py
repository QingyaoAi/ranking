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
	example_pid_feature_map = None
	context_qid_feature_map = None

	smoothing = 10e-9
	padding_number = -10e9
	mu = 200
	jm_lambda = 0.5
	b = 0.75
	k = 0.6
	theta = 0.5
	vocab_stat_map = {}
	corpus_length = 0.0
	doc_count = 0.0
	q_term_map = {}
	numerical_feature_scale_map = None

	ps = PorterStemmer()
	SET_LIST = ['train', 'dev', 'eval']

	def create_feature_columns(self):
		"""Returns the example feature columns."""
		
		# build embedding features
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

		# build example numerical features
		for name in ['query_len', 'idfs']:
			context_feature_columns[name] = tf.feature_column.numeric_column(
											name, shape=(1,), default_value=0.0)

		# build example numerical features
		for name in ['doc_len', 'tfs', 'tfidfs', 'bm25s', 'lmabs', 'lmdirs', 'lmjrs']:
			example_feature_columns[name] = tf.feature_column.numeric_column(
											name, shape=(1,), default_value=0.0)

		return context_feature_columns, example_feature_columns

	def _tokenize(self, line):
		text = line.strip()
		for ch in string.punctuation:
			text = text.replace(ch, '')
		return [self.ps.stem(w.lower()) for w in text.split(' ')]

	def __init__(self, data_json_file_path, list_size):
		self.settings = json.load(open(data_json_file_path))
		self.vocabulary = {}
		self.vocab_info = []
		self.vocab_idxs = {}
		self.list_size = list_size
		vocab_file = self.settings["WORKING_PATH"] + '/vocab_min_count_%d.txt' % self.settings["word_min_count"]
		if os.path.isfile(vocab_file):
			print('Load vocabulary')
			with open(vocab_file) as fin:
				idx = 0
				for line in fin:
					arr = line.strip().split(' ')
					self.vocabulary[arr[0]] = [float(x) for x in arr[1:]]
					self.vocab_idxs[arr[0]] = idx
					self.vocab_info.append(self.vocabulary[arr[0]])
					idx += 1
					if idx != len(self.vocabulary):
						print(idx)
		else:
			print('Build vocabulary')
			if not os.path.exists(self.settings["WORKING_PATH"]):
				os.makedirs(self.settings["WORKING_PATH"])
			
			def _add_words_to_vocab(raw_word_list, in_corpus):
				word_list = [w.strip() for w in raw_word_list]
				word_list = [w.lower() for w in word_list if len(w) > 0]
				for word in word_list:
					if word not in self.vocabulary:
						self.vocabulary[word] = [0,0] #cf, df, cf in both corpus and queries
					if in_corpus:
						self.vocabulary[word][0] += 1 #cf
				if in_corpus:
					for word in set(word_list):
						self.vocabulary[word][1] += 1 #df
			# add collection words		
			with open(self.settings["COLLECTION_PATH"] + '/corpus_text.txt') as text_fin:
				for line in text_fin:
					words = self._tokenize(line)
					_add_words_to_vocab(words, True)

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
						_add_words_to_vocab(self._tokenize(query_text), False)

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
					self.vocab_info.append(self.vocabulary[w])
					fout.write('%s %d %d\n' % (w, self.vocabulary[w][0], self.vocabulary[w][1]))
					idx += 1
		print('Vocabulary loading finished, size %d' % (len(self.vocabulary)))

		# load basic corpus information
		statistic_file = self.settings["WORKING_PATH"] + '/stats_min_count_%d.json' % self.settings["word_min_count"]
		if os.path.isfile(statistic_file):
			print('Load Statistic Information')
			stats = json.load(open(statistic_file))
			self.max_doc_length = stats['max_doc_length']
			self.max_query_length = stats['max_query_length']
			self.corpus_length = stats['corpus_length']
			self.doc_count = stats['doc_count']
			self.avg_doc_len = stats['avg_doc_len']
			self.numerical_feature_scale_map = stats['numerical_feature_scale_map']
		else:
			# load corpus to build stats
			self.load_corpus_data()
			stats = {
				'max_doc_length' : self.max_doc_length,
				'max_query_length' : self.max_query_length,
				'corpus_length' : self.corpus_length,
				'doc_count' : self.doc_count,
				'avg_doc_len' : float(self.corpus_length)/float(self.doc_count)
			}
			# compute dense feature scale
			self.compute_dense_feature_scale('train', self.list_size)
			stats['numerical_feature_scale_map'] = self.numerical_feature_scale_map
			# write to file
			with open(statistic_file, 'w') as fout:
				json.dump(stats, fout, sort_keys = True, indent = 4)



	def load_corpus_data(self):
		# Build vocabulary
		print('Start loading data')

		# Build doc_term stats
		def _build_term_stats(raw_terms):
			doc_terms = {}
			for t in raw_terms:
				if t not in doc_terms:
					doc_terms[t] = 0.0
				doc_terms[t] += 1
			return doc_terms

		# Load passages and build example features
		doc_file = self.settings["WORKING_PATH"] + '/collection_min_count_%d.txt.gz' % self.settings["word_min_count"]
		self.example_pid_feature_map = {}
		self.max_doc_length = 0
		self.context_qid_feature_map = {}
		self.max_query_length = 0
		self.corpus_length = 0
		def _get_word_idxs(words):
			word_idxs = []
			for i in range(len(words)):
				if words[i] in self.vocab_idxs:
					word_idxs.append(self.vocab_idxs[words[i]])
			return word_idxs

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
					self.example_pid_feature_map[pid]['doc_len'] = len(words)
					# add term distribution 
					self.example_pid_feature_map[pid]['term_stats'] = _build_term_stats(words)
					if len(words) > self.max_doc_length:
						self.max_doc_length = len(words)
					if len(self.example_pid_feature_map) % 10000 == 0:
						print('Read %d docs' % len(self.example_pid_feature_map))
					self.corpus_length += self.example_pid_feature_map[pid]['doc_len']

		else: # if the collection hasn't been processed yet, process it and store the file.
			# load raw collection
			print('Create passage features...')

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
						self.example_pid_feature_map[pid]['doc_len'] = len(word_idxs)
						# add term distribution 
						self.example_pid_feature_map[pid]['term_stats'] = _build_term_stats(word_idxs)
						if len(word_idxs) > self.max_doc_length:
							self.max_doc_length = len(word_idxs)
						if len(self.example_pid_feature_map) % 10000 == 0:
							print('Read %d docs' % len(self.example_pid_feature_map))
						self.corpus_length += self.example_pid_feature_map[pid]['doc_len']

			# write processed collection
			with gzip.open(doc_file, 'wt') as fout:
				for pid in pid_list:
					word_idxs = self.example_pid_feature_map[pid]["doc_unigrams"]
					fout.write('%d\t%s\n' % (pid, ' '.join([str(x) for x in word_idxs])))

		self.doc_count = len(self.example_pid_feature_map)
		self.avg_doc_len = float(self.corpus_length) / self.doc_count
		
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
						self.context_qid_feature_map[qid]["query_len"] = len(words)
						idf_list = [self.doc_count/(self.vocab_info[w][1]+0.5) for w in words]
						self.context_qid_feature_map[qid]["idfs"] = sum(idf_list) / float(len(idf_list))
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
						self.context_qid_feature_map[qid]["query_len"] = len(word_idxs)
						idf_list = [self.doc_count/(self.vocab_info[w][1]+0.5) for w in word_idxs]
						self.context_qid_feature_map[qid]["idfs"] = sum(idf_list) / float(len(idf_list))
						if len(word_idxs) > self.max_query_length:
							self.max_query_length = len(word_idxs)
				# write processed collection
				with gzip.open(query_file, 'wt') as fout:
					for qid in qid_list:
						word_idxs = self.context_qid_feature_map[qid]["query_unigrams"]
						fout.write('%d\t%s\n' % (qid, " ".join([str(x) for x in word_idxs])))

		tf.logging.info("Collection size {}".format(str(self.corpus_length)))
		tf.logging.info("Collection doc count {}".format(str(self.doc_count)))
		tf.logging.info("Max doc length {}".format(str(self.max_doc_length)))
		tf.logging.info("Max query length {}".format(str(self.max_query_length)))


		print('Load finish')

	def compute_dense_feature_scale(self, set_name, list_size):

		tf.logging.info("Computing numerical feature scales for {}".format(set_name))
		
		# if raw data are not loaded, load them
		if self.example_pid_feature_map is None:
			self.load_corpus_data()	
		
		# Read data
		qid_to_doc = {} # The list of docs seen so far for a query.
		max_list_length = 0
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
				label = 0
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
		numerical_feature_scale_map = {}
		total_docs = 0
		discarded_docs = 0
		for qid in qid_to_doc:
			feature_map = {}
			# create context features
			for k in context_feature_columns:
				if not k.endswith('unigrams'): # this is a numerical feature
					feature_map[k] = self.context_qid_feature_map[qid][k]
					if k not in numerical_feature_scale_map:
						numerical_feature_scale_map[k] = [feature_map[k], feature_map[k]]
					else:
						if feature_map[k] < numerical_feature_scale_map[k][0]:
							numerical_feature_scale_map[k][0] = feature_map[k]
						if feature_map[k] > numerical_feature_scale_map[k][1]:
							numerical_feature_scale_map[k][1] = feature_map[k]
			
			# compute dense example features	
			for i in range(len(qid_to_doc[qid])):
				if i < list_size:
					pid = qid_to_doc[qid][i][0]
					feature_list, name_list = self.get_example_dense_features(
												self.context_qid_feature_map[qid]['query_unigrams'],
												self.example_pid_feature_map[pid]['term_stats'],
												self.example_pid_feature_map[pid]['doc_len'])
					for k in range(len(name_list)):
						key = name_list[k]
						value = feature_list[k]
						if key not in numerical_feature_scale_map:
							numerical_feature_scale_map[key] = [value, value]
						else:
							if value < numerical_feature_scale_map[key][0]:
								numerical_feature_scale_map[key][0] = value
							if value > numerical_feature_scale_map[key][1]:
								numerical_feature_scale_map[key][1] = value

		# print feature scale information
		for key in numerical_feature_scale_map:
			tf.logging.info("%s feature scale: [%.3f, %.3f]" % (
				key, numerical_feature_scale_map[key][0], numerical_feature_scale_map[key][1]))

		self.numerical_feature_scale_map = numerical_feature_scale_map

	def get_example_dense_features(self, query_terms, doc_terms, doc_len):
		doc_len = max(doc_len, 0.1)
		tfs, tfidfs, bm25s, lmabs, lmdirs, lmjrs = [],[],[],[],[],[]
		for t in query_terms:
			tf = 0.0
			cf, df = self.vocab_info[t]
			idf = self.doc_count/(df+0.5)
			if t in doc_terms:
				tf = doc_terms[t]	
			# TF, IDF, TF-IDF
			tfs.append(tf)
			tfidfs.append(tf * idf)
			# BM25
			numerator = tf * (1 + self.k)
			denominator = tf + self.k * (1 - self.b + self.b * doc_len / self.avg_doc_len)
			bm25 = idf * numerator / denominator
			bm25s.append(bm25)
			# LMABS, LMDIR, LMJR
			background = cf / self.corpus_length
			gamma = self.theta * len(doc_terms) / float(doc_len)
			lmab = max(tf - self.theta, 0.0) / doc_len + gamma * background
			lmab = max(lmab, self.theta * background)
			lmdir = (tf + self.mu * background) / (doc_len + self.mu)
			lmjr = (1-self.jm_lambda)*tf/doc_len + self.jm_lambda * background
			lmabs.append(np.log(lmab))
			lmdirs.append(np.log(lmdir))
			lmjrs.append(np.log(lmjr))
		feature = [
			sum(x) / len(x) for x in [tfs, tfidfs, bm25s, lmabs, lmdirs, lmjrs]
		]
		feature.append(doc_len)
		feature_names = ['tfs', 'tfidfs', 'bm25s', 'lmabs', 'lmdirs', 'lmjrs', 'doc_len']
		return feature, feature_names

	def get_TFReord_parser(self):
		''' Create the parser used to parse data read from  TFRecord'''
		context_feature_columns, example_feature_columns = self.create_feature_columns()
		# build feature map
		feature_map = {}
		feature_map['label'] = tf.FixedLenFeature([self.list_size], tf.float32)
		for k in context_feature_columns:
			if k.endswith('unigrams'):
				feature_map[k] = tf.SparseFeature(index_key=['%s_idx' % k],
												value_key='%s_int_value' % k,
												dtype=tf.int64,
												size=[self.max_query_length])
			else:
				feature_map[k] = tf.FixedLenFeature([1], tf.float32)
		for k in example_feature_columns:
			if k.endswith('unigrams'):
				feature_map[k] = tf.SparseFeature(index_key=['%s_list_idx' % k, '%s_idx' % k],
												value_key='%s_int_value' % k,
												dtype=tf.int64,
												size=[self.list_size, self.max_doc_length])
			else:
				feature_map[k] = tf.FixedLenFeature([self.list_size], tf.float32)
												

		def parser(serialized_example):
			"""Parses a single tf.Example into image and label tensors."""

			features = tf.parse_single_example(serialized_example,
												features=feature_map)
			label = features.pop('label')
			print(features['bm25s'])
			
			return features, label
			
		return parser
		
	def get_file_paths(self, set_name, list_size):
		# return a list of file paths for TFRecord dataset

		# check if corresponding files exists
		file_paths = []
		root_path = self.settings["WORKING_PATH"] + '/list_size_%d/%s/' % (list_size, set_name)
		if not os.path.exists(root_path):
			os.makedirs(root_path)
		data_info_file = root_path + '/info.json'
		if os.path.isfile(data_info_file):
			data_info = json.load(open(data_info_file))
			file_paths = data_info['file_paths']
			max_list_length = data_info['max_list_length']
			#list_size = list_size if list_size > 0 else max_list_length
			if max_list_length > self.max_list_length:
				self.max_list_length = max_list_length	
			# TODO: load feature scales
		else:
			if self.example_pid_feature_map is None:
				self.load_corpus_data()

			tf.logging.info("Creating TFRecord data for {}".format(set_name))
			# if raw data are not loaded, load them
			#if self.max_doc_length < 1:
			#	self.load_corpus_data()
				
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
			total_docs = 0
			discarded_docs = 0
			count_record = 0
			record_file_path = root_path + '%d.tfrecord' % count_record
			fout = tf.python_io.TFRecordWriter(record_file_path)
			file_paths.append(record_file_path)
			for qid in qid_to_doc:
				feature_map = {}
				# create label
				feature_map['label'] = [-1.0 for _ in range(list_size)]
				for i in range(len(qid_to_doc[qid])):
					if i < list_size:
						pid = qid_to_doc[qid][i][0]
						label = qid_to_doc[qid][i][1]
						feature_map['label'][i] = label
					else:
						discarded_docs += 1
					total_docs += 1

				# create context features
				for k in context_feature_columns:
					if k.endswith('unigrams'): # use sparse feature
						idx_key = '%s_idx' % k
						value_key = '%s_int_value' % k
						feature_map[idx_key] = []
						feature_map[value_key] = []
						context_feature_vector = self.context_qid_feature_map[qid][k]
						for i in range(len(context_feature_vector)):
							feature_map[idx_key].append(i)
							feature_map[value_key].append(context_feature_vector[i])
					else: # use dense features
						feature_map[k] = [self.context_qid_feature_map[qid][k]]
				
				# compute dense example features	
				dense_features = {}
				for i in range(len(qid_to_doc[qid])):
					if i < list_size:
						pid = qid_to_doc[qid][i][0]
						feature_list, name_list = self.get_example_dense_features(
													self.context_qid_feature_map[qid]['query_unigrams'],
													self.example_pid_feature_map[pid]['term_stats'],
													self.example_pid_feature_map[pid]['doc_len'])
						for k in range(len(name_list)):
							key = name_list[k]
							value = feature_list[k]
							if key not in dense_features:
								dense_features[key] = [self.padding_number for _ in range(list_size)]
							dense_features[key][k] = value

				# create example features
				for k in example_feature_columns:
					if k.endswith('unigrams'): # use sparse feature
						list_idx_key = '%s_list_idx' % k
						idx_key = '%s_idx' % k
						value_key = '%s_int_value' % k
						feature_map[list_idx_key] = []
						feature_map[idx_key] = []
						feature_map[value_key] = []
						for i in range(len(qid_to_doc[qid])):
							if i < list_size:
								pid = qid_to_doc[qid][i][0]
								label = qid_to_doc[qid][i][1]
								example_feature_vector = self.example_pid_feature_map[pid][k]
								for j in range(len(example_feature_vector)):
									feature_map[list_idx_key].append(i)
									feature_map[idx_key].append(j)
									feature_map[value_key].append(example_feature_vector[j])
					else:
						feature_map[k] = dense_features[k]
				
				
				# convert feature map to example
				for key in feature_map:
					if key.endswith('idx') or key.endswith('int_value'): # key for sparse features
						feature_map[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature_map[key]))
					elif key != 'label': # key for numerical features
						for i in range(len(feature_map[key])):
							if feature_map[key][i] == self.padding_number: # its a padding, just set as 0
								feature_map[key][i] = 0.0
							else:
								feature_map[key][i] = (
									feature_map[key][i] - self.numerical_feature_scale_map[key][0])/(
									self.numerical_feature_scale_map[key][1] - self.numerical_feature_scale_map[key][0] + self.smoothing
									)
						feature_map[key] = tf.train.Feature(float_list=tf.train.FloatList(value=feature_map[key]))
					else: # this is a key for 'label'
						feature_map[key] = tf.train.Feature(float_list=tf.train.FloatList(value=feature_map[key]))
				feature_example = tf.train.Example(features=tf.train.Features(feature=feature_map))

				# write to TFRecord file
				fout.write(feature_example.SerializeToString())
				count_record += 1
				if count_record % 10000 == 0:
					fout.close()
					record_file_path = root_path + '%d.tfrecord' % count_record
					fout = tf.python_io.TFRecordWriter(record_file_path)
					file_paths.append(record_file_path)
			fout.close()

			# write data info
			data_info = {
				'file_paths' : file_paths,
				'max_list_length' : max_list_length,
				'query_number' : len(qid_to_doc),
				'total_document' : total_docs,
				'discarded_document' : discarded_docs,
			}
			with open(data_info_file, 'w') as fout:
				json.dump(data_info, fout, sort_keys = True, indent = 4)
			
			tf.logging.info("Number of queries: {}".format(len(qid_to_doc)))
			tf.logging.info("Number of documents in total: {}".format(total_docs))
			tf.logging.info("Number of documents discarded: {}".format(discarded_docs))

		self.list_size = list_size
		return file_paths





