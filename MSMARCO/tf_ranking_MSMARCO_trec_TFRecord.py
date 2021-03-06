# Copyright 2018 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""TF Ranking sample code for LETOR datasets in LibSVM format.

WARNING: All data sets are loaded into memory in this sample code. It is
for small data sets whose sizes are < 10G.

A note on the LibSVM format:
--------------------------------------------------------------------------
Due to the sparse nature of features utilized in most academic datasets for
learning to rank such as LETOR datasets, data points are represented in the
LibSVM format. In this setting, every line encapsulates features and a (graded)
relevance judgment of a query-document pair. The following illustrates the
general structure:

<relevance int> qid:<query_id int> [<feature_id int>:<feature_value float>]

For example:

1 qid:10 32:0.14 48:0.97	51:0.45
0 qid:10 1:0.15	31:0.75	32:0.24	49:0.6
2 qid:10 1:0.71	2:0.36	 31:0.58	51:0.12
0 qid:20 4:0.79	31:0.01	33:0.05	35:0.27
3 qid:20 1:0.42	28:0.79	35:0.30	42:0.76

In the above example, the dataset contains two queries. Query "10" has 3
documents, two of which relevant with grades 1 and 2. Similarly, query "20"
has 1 relevant document. Note that query-document pairs may have different
sets of zero-valued features and as such their feature vectors may only
partly overlap or not at all.
--------------------------------------------------------------------------

Sample command lines:

OUTPUT_DIR=/tmp/output && \
TRAIN=tensorflow_ranking/examples/data/train.txt && \
VALI=tensorflow_ranking/examples/data/vali.txt && \
TEST=tensorflow_ranking/examples/data/test.txt && \
rm -rf $OUTPUT_DIR && \
bazel build -c opt \
tensorflow_ranking/examples/tf_ranking_libsvm_py_binary && \
./bazel-bin/tensorflow_ranking/examples/tf_ranking_libsvm_py_binary \
--train_path=$TRAIN \
--vali_path=$VALI \
--test_path=$TEST \
--output_dir=$OUTPUT_DIR \
--num_features=136

You can use TensorBoard to display the training results stored in $OUTPUT_DIR.
"""

from absl import flags

import os,sys
import numpy as np
import six
import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow_ranking.python import feature as feature_lib
import data_util_trec_TFRecord as data_util

flags.DEFINE_string("setting_path", None, "The json file path used for experiment settings, including data path.")
flags.DEFINE_string("output_dir", None, "Output directory for models.")

flags.DEFINE_integer("train_batch_size", 256, "The batch size for training.")
flags.DEFINE_integer("train_prefetch_buffer_size", -1, "The batch size for training.")
flags.DEFINE_integer("dev_prefetch_buffer_size", -1, "The batch size for training.")
flags.DEFINE_integer("eval_prefetch_buffer_size", -1, "The batch size for training.")
flags.DEFINE_integer("num_train_steps", 100000, "Number of steps for training.")

flags.DEFINE_float("learning_rate", 0.01, "Learning rate for optimizer.")
flags.DEFINE_float("dropout_rate", 0.0, "The dropout rate before output layer.")
flags.DEFINE_list("hidden_layer_dims", ["256", "128", "64"],
									"Sizes for hidden layers.")

#flags.DEFINE_integer("num_features", 136, "Number of features per document.")
flags.DEFINE_integer("list_size", 1000, "List size used for training. -1 means using all docs.")
flags.DEFINE_integer("group_size", 1, "Group size used in score function.")
flags.DEFINE_integer("rank_cut", 10, "The number of documents output for each query in prediction phase.")

flags.DEFINE_string("loss", "pairwise_logistic_loss",
										"The RankingLossKey for loss function.")
flags.DEFINE_string("activate", "relu", "The activate function in the network.")

flags.DEFINE_boolean("evaluation_only", False, "Only conduct evaluation if True.")
flags.DEFINE_boolean("train_from_scratch", False, 
										"Delete existing files in output_dir and train the model from scratch if True.")

FLAGS = flags.FLAGS

# get the shape of a list
def get_shape(lst, shape=()):
	"""
	returns the shape of nested lists similarly to numpy's shape.

	:param lst: the nested list
	:param shape: the shape up to the current recursion depth
	:return: the shape including the current depth
			(finally this will be the full depth)
	"""

	if not isinstance(lst, list):
		# base case
		return shape

	# peek ahead and assure all lists in the next depth
	# have the same length
	if isinstance(lst[0], list):
		l = len(lst[0])
		if not all(len(item) == l for item in lst):
			print(l)
			msg = 'not all lists have the same length'
			raise ValueError(msg)

	shape += (len(lst), )

	# recurse
	shape = get_shape(lst[0], shape)

	return shape

class IteratorInitializerHook(tf.train.SessionRunHook):
	"""Hook to initialize data iterator after session is created."""

	def __init__(self):
		super(IteratorInitializerHook, self).__init__()
		self.iterator_initializer_fn = None

	def after_create_session(self, session, coord):
		"""Initialize the iterator after the session has been created."""
		del coord
		self.iterator_initializer_fn(session)


def get_train_inputs(file_paths, parser, batch_size, prefetch_buffer_size):
	"""Set up training input in batches."""
	iterator_initializer_hook = IteratorInitializerHook()

	def _train_input_fn():
		"""Defines training input fn."""
		print('\n\nRunning _train_input_fn\n\n')

		filename_placeholder = tf.placeholder(tf.string, shape=[None])
		dataset = tf.data.TFRecordDataset(filename_placeholder)
		dataset = dataset.shuffle(batch_size).repeat()
		
		#dataset = dataset.map(parser, num_parallel_calls=8)
		#dataset = dataset.batch(batch_size)
		dataset = dataset.apply(tf.data.experimental.map_and_batch(
								map_func=parser, batch_size=batch_size))
		if prefetch_buffer_size > 0:
			dataset = dataset.prefetch(buffer_size=int(prefetch_buffer_size))
		iterator = dataset.make_initializable_iterator()
		feed_dict = {filename_placeholder: file_paths}
		run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
		iterator_initializer_hook.iterator_initializer_fn = (
				lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict, options=run_options))
		return iterator.get_next()

	return _train_input_fn, iterator_initializer_hook


def get_eval_inputs(file_paths, parser, batch_size, prefetch_buffer_size):
	"""Set up eval inputs in a single batch."""
	iterator_initializer_hook = IteratorInitializerHook()

	def _eval_input_fn():
		print('\n\nRunning _eval_input_fn\n\n')
		"""Defines eval input fn."""
		filename_placeholder = tf.placeholder(tf.string, shape=[None])
		dataset = tf.data.TFRecordDataset(filename_placeholder)
		
		#dataset = dataset.map(parser, num_parallel_calls=8)
		#dataset = dataset.batch(batch_size)
		dataset = dataset.apply(tf.data.experimental.map_and_batch(
								map_func=parser, batch_size=batch_size))

		if prefetch_buffer_size > 0:
			dataset = dataset.prefetch(buffer_size=int(prefetch_buffer_size))
		iterator = dataset.make_initializable_iterator()
		feed_dict = {filename_placeholder: file_paths}
		run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
		iterator_initializer_hook.iterator_initializer_fn = (
				lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict, options=run_options))
		return iterator.get_next()

	return _eval_input_fn, iterator_initializer_hook


def make_score_fn(data):
	"""Returns a groupwise score fn to build `EstimatorSpec`."""
	
	context_feature_columns, example_feature_columns = data.create_feature_columns()

	def _score_fn(context_features, group_features, mode, unused_params,
								unused_config):
		"""Defines the network to score a group of documents."""
		with tf.name_scope("input_layer"):
			group_input = [
					tf.layers.flatten(group_features[name])
					for name in sorted(example_feature_columns)
			]
			#for x in group_input:
			#	print(x.shape)
			
			context_input = [
					tf.layers.flatten(context_features[name])
					for name in sorted(context_feature_columns)
			]
			#for x in context_input:
			#	print(x.shape)
			
			final_input = context_input + group_input
			input_layer = tf.concat(final_input, 1)
			tf.summary.scalar("input_sparsity", tf.nn.zero_fraction(input_layer))
			tf.summary.scalar("input_max", tf.reduce_max(input_layer))
			tf.summary.scalar("input_min", tf.reduce_min(input_layer))

		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		tf.logging.info("Activate function: %s" % FLAGS.activate)
		cur_layer = tf.layers.batch_normalization(input_layer, training=is_training)
		for i, layer_width in enumerate(int(d) for d in FLAGS.hidden_layer_dims):
			cur_layer = tf.layers.dense(cur_layer, units=layer_width)
			cur_layer = tf.layers.batch_normalization(cur_layer, training=is_training)
			activate = tf.nn.elu
			if FLAGS.activate == 'relu':
				activate = tf.nn.relu
				tf.summary.scalar("fully_connected_{}_sparsity".format(i),
													tf.nn.zero_fraction(cur_layer))
			elif FLAGS.activate == 'tanh':
				activate = tf.nn.tanh
			cur_layer = activate(cur_layer)
		cur_layer = tf.layers.dropout(
				cur_layer, rate=FLAGS.dropout_rate, training=is_training)
		logits = tf.layers.dense(cur_layer, units=FLAGS.group_size)
		return logits

	return _score_fn


def get_eval_metric_fns():
	"""Returns a dict from name to metric functions."""
	metric_fns = {}
	metric_fns.update({
			"metric/mrr@%d" % topn: tfr.metrics.make_ranking_metric_fn(
					tfr.metrics.RankingMetricKey.MRR, topn=topn)
			for topn in [10]
	})
	metric_fns.update({
			"metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name) for name in [
					tfr.metrics.RankingMetricKey.ARP,
					#tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY,
			]
	})
	metric_fns.update({
			"metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
					tfr.metrics.RankingMetricKey.NDCG, topn=topn)
			for topn in [1, 3, 5, 10]
	})
	return metric_fns

def make_transform_fn(input_size, context_feature_columns, example_feature_columns):

	def _transform_fn(features, mode):
		"""Splits the features into context and per-example features."""
		
		'''
		print('Before feature transform_fn')
		for k in features:
			print(k)
			print(features[k].shape)
		'''
		context_features, example_features = feature_lib.encode_listwise_features(
				features,
				input_size=input_size,
				context_feature_columns=context_feature_columns,
				example_feature_columns=example_feature_columns,
				mode=mode)
		'''
		print('After feature transform_fn')
		for k in example_features:
			print(k)
			print(example_features[k].shape)
		for k in context_features:
			print(k)
			print(context_features[k].shape)
		'''
		return context_features, example_features

	return _transform_fn


def train_and_eval():
	"""Train and Evaluate."""
	# load collection
	data = data_util.MsMarcoData(FLAGS.setting_path, FLAGS.list_size)
	context_feature_columns, example_feature_columns = data.create_feature_columns()

	# load train/vali/test
	train_input_fn, train_hook = get_train_inputs(data.get_file_paths("train", FLAGS.list_size), data.get_TFReord_parser(),
												FLAGS.train_batch_size, FLAGS.train_prefetch_buffer_size)

	#vali_input_fn, vali_hook = get_eval_inputs(data.get_file_paths("dev", data.list_size), data.get_TFReord_parser(), 
	vali_input_fn, vali_hook = get_eval_inputs(data.get_file_paths("dev", data.list_size), data.get_TFReord_parser(), 
												FLAGS.train_batch_size, FLAGS.dev_prefetch_buffer_size)


	def _train_op_fn(loss):
		"""Defines train op used in ranking head."""
		return tf.contrib.layers.optimize_loss(
				loss=loss,
				global_step=tf.train.get_global_step(),
				learning_rate=FLAGS.learning_rate,
				optimizer="Adagrad")

	ranking_head = tfr.head.create_ranking_head(
			loss_fn=tfr.losses.make_loss_fn(FLAGS.loss),
			eval_metric_fns=get_eval_metric_fns(),
			train_op_fn=_train_op_fn)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	#config.run_options.report_tensor_allocations_upon_oom = True
	estimator = tf.estimator.Estimator(
			model_fn=tfr.model.make_groupwise_ranking_fn(
					group_score_fn=make_score_fn(data),
					group_size=FLAGS.group_size,
					transform_fn=make_transform_fn(data.list_size, context_feature_columns, example_feature_columns), # add context feature
					#transform_fn=tfr.feature.make_identity_transform_fn(context_feature_columns.keys()), # add context feature
					ranking_head=ranking_head),
			config=tf.estimator.RunConfig(
					FLAGS.output_dir, save_checkpoints_steps=1000, session_config=config))

	train_spec = tf.estimator.TrainSpec(
			input_fn=train_input_fn,
			hooks=[train_hook],
			max_steps=FLAGS.num_train_steps)
	vali_spec = tf.estimator.EvalSpec(
			input_fn=vali_input_fn,
			hooks=[vali_hook],
			#steps=1000,
			steps=None,
			start_delay_secs=30,
			throttle_secs=30)

	# Train and validate
	tf.estimator.train_and_evaluate(estimator, train_spec, vali_spec)

	# Evaluate on the test data.
	test_input_fn, test_hook = get_eval_inputs(data.get_file_paths("eval", data.list_size), data.get_TFReord_parser(), 
												FLAGS.train_batch_size, FLAGS.eval_prefetch_buffer_size)
	estimator.evaluate(input_fn=test_input_fn, hooks=[test_hook])


def eval_only():
	"""Train and Evaluate."""
	# load collection
	data = data_util.MsMarcoData(FLAGS.setting_path, FLAGS.list_size)
	context_feature_columns, example_feature_columns = data.create_feature_columns()

	# load test
	test_input_fn, test_hook = get_eval_inputs(data.get_file_paths("eval", FLAGS.list_size), data.get_TFReord_parser(), 
												FLAGS.train_batch_size, FLAGS.eval_prefetch_buffer_size)
	input_size = FLAGS.list_size if FLAGS.list_size > 0 else data.max_list_length
	tf.logging.info("Actual list size: {}".format(input_size))

	def _train_op_fn(loss):
		"""Defines train op used in ranking head."""
		return tf.contrib.layers.optimize_loss(
				loss=loss,
				global_step=tf.train.get_global_step(),
				learning_rate=FLAGS.learning_rate,
				optimizer="Adagrad")

	ranking_head = tfr.head.create_ranking_head(
			loss_fn=tfr.losses.make_loss_fn(FLAGS.loss),
			eval_metric_fns=get_eval_metric_fns(),
			train_op_fn=_train_op_fn)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	#config.run_options.report_tensor_allocations_upon_oom = True
	estimator = tf.estimator.Estimator(
			model_fn=tfr.model.make_groupwise_ranking_fn(
					group_score_fn=make_score_fn(data),
					group_size=FLAGS.group_size,
					transform_fn=make_transform_fn(input_size, context_feature_columns, example_feature_columns), # add context feature
					#transform_fn=tfr.feature.make_identity_transform_fn(context_feature_columns.keys()), # add context feature
					ranking_head=ranking_head),
			config=tf.estimator.RunConfig(
					FLAGS.output_dir, save_checkpoints_steps=1000, session_config=config))

	# Evaluate on the test data.
	estimator.evaluate(input_fn=test_input_fn, hooks=[test_hook])

	# Make prediction and output trec list
	predict_input_fn, predict_hook = get_eval_inputs(data.get_file_paths("eval", FLAGS.list_size), data.get_TFReord_parser(), 
												FLAGS.train_batch_size, FLAGS.eval_prefetch_buffer_size)
	result_generator = estimator.predict(input_fn=predict_input_fn, hooks=[predict_hook])
	with open(FLAGS.output_dir + '/eval.trec.ranklist', 'w') as fout:
		data.generate_trec_ranklist_with_result_generator('eval', FLAGS.list_size, 
														'Groupwise', result_generator, fout, FLAGS.rank_cut)


def main(_):
	tf.logging.set_verbosity(tf.logging.INFO)
	if not os.path.exists(FLAGS.output_dir):
		os.makedirs(FLAGS.output_dir)
	if FLAGS.train_from_scratch:
		print('Train from scratch...')
		os.system('rm -r %s' % FLAGS.output_dir)
	
	if not FLAGS.evaluation_only:
		train_and_eval()
	else:
		eval_only()


if __name__ == "__main__":
	flags.mark_flag_as_required("setting_path")
	flags.mark_flag_as_required("output_dir")

	tf.app.run()
