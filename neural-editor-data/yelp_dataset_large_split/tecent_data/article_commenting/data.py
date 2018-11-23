# encoding: utf-8
import json
import jieba
import jieba.analyse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_input_helper as data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import os
from data_input_helper import clean_str, replace_words

# 不使用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""


def get_wordvec():
	word2vec = {}
	with open('vector_small_data.txt') as f:
		f.readline()
		for line in f.readlines():
			word2vec[line.strip().split(' ')[0]] = line.strip().split(' ')[1:]
	return word2vec


def get_data1():
	print ('start getting data...')

	test_data = []
	with open('newdata.test.clean.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)
			test_data.append(data)
	print ('finish getting test data.')

	train_data = []
	with open('newdata.train.clean.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)
			train_data.append(data)
	print ('finish getting train data.')

	dev_data = []
	with open('newdata.dev.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)
			dev_data.append(data)
	print('finish getting dev data.')

	return train_data, dev_data, test_data


def get_data2():
	print ('start getting data...')

	test_data = []
	with open('newdata.test.clean.json') as f:
		test_data = json.load(f)
	print ('finish getting test data.')

	train_data = []
	with open('newdata.train.clean.json') as f:
		train_data = json.load(f)
	print ('finish getting train data.')

	dev_data = []
	with open('newdata.dev.clean.json') as f:
		dev_data = json.load(f)
	print('finish getting dev data.')

	return train_data, dev_data, test_data


def preprocess_content():
	'''
	预处理数据集中的新闻内容。
	:return:
	'''
	train_data, dev_data, test_data = get_data1()
	# for k, item in enumerate(train_data):
	# 	if k % 100 == 0:
	# 		print 'clean train data: {}/{}'.format(k, len(train_data))
	# 	tmp = ' '.join(jieba.cut(item['body']))
	# 	result = clean_str(tmp)
	# 	tmp_clean = ' '.join([i for i in tmp.split(' ') if i not in result.split(' ')])
	# 	tmp_clean = replace_words(tmp_clean)
	# 	train_data[k]['body'] = tmp_clean
	#
	# for k, item in enumerate(test_data):
	# 	if k % 100 == 0:
	# 		print 'clean test data: {}/{}'.format(k, len(test_data))
	# 	tmp = ' '.join(jieba.cut(item['body']))
	# 	result = clean_str(tmp)
	# 	tmp_clean = ' '.join([i for i in tmp.split(' ') if i not in result.split(' ')])
	# 	tmp_clean = replace_words(tmp_clean)
	# 	test_data[k]['body'] = tmp_clean

	for k, item in enumerate(dev_data):
		if k % 100 == 0:
			print 'clean dev data: {}/{}'.format(k, len(dev_data))
		tmp = ' '.join(jieba.cut(item['body']))
		result = clean_str(tmp)
		tmp_clean = ' '.join([i for i in tmp.split(' ') if i not in result.split(' ')])
		tmp_clean = replace_words(tmp_clean)
		dev_data[k]['body'] = tmp_clean

	# with open('newdata.train.clean.json', 'w') as f:
	# 	json.dump(train_data, f)
	# with open('newdata.test.clean.json', 'w') as f:
	# 	json.dump(test_data, f)
	with open('newdata.dev.clean.json', 'w') as f:
		json.dump(dev_data, f)


def news_len():
	# 新闻内容关键字：'body'
	train_data, dev_data, test_data = get_data2()
	train_len, dev_len, test_len = [], [],[]
	for i in range(len(train_data)):
		train_len.append(len(train_data[i]['body'].split(' ')))
	for i in range(len(dev_data)):
		dev_len.append(len(dev_data[i]['body'].split(' ')))
	for i in range(len(test_data)):
		test_len.append(len(test_data[i]['body'].split(' ')))

	print 'max len of train content: {}'.format(max(train_len))
	print 'max len of dev content: {}'.format(max(dev_len))
	print 'max len of test content: {}'.format(max(test_len))

	with open('train.len.json', 'w') as f:
		json.dump(train_len, f)

	with open('dev.len.json', 'w') as f:
		json.dump(dev_len, f)

	with open('test.len.json', 'w') as f:
		json.dump(test_len, f)


def count_news_len(file):
	with open(file) as f:
		data = json.load(f)

	num1, num2, num3, num4, num5 = 0,0,0,0,0
	for i, item in enumerate(data):
		if 0 <= item <= 50: num1 += 1
		elif 50 < item <= 100: num2 += 1
		elif 100 < item <= 500: num3 += 1
		elif 500 <= item < 1000: num4 += 1
		else: num5 += 1

	t = float(len(data))
	print '0-50: {}'.format(float(num1) / t)
	print '50-100: {}'.format(float(num2)/t)
	print '100-500: {}'.format(float(num3)/t)
	print '500-1000: {}'.format(float(num4)/t)
	print '1000- : {}'.format(float(num5)/t)


def get_small_data():
	test_data = []
	with open('data.test.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)
			test_data.append(data)
	print ('finish getting test data.')

	train_data = []
	with open('data.train.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)
			train_data.append(data)
	print ('finish getting train data.')

	test_data_small = test_data[:5]
	train_data_small = train_data[:20]

	with open('data.small.train.json', 'w') as f:
		json.dump(train_data_small, f)
	with open('data.small.test.json', 'w') as f:
		json.dump(test_data_small, f)


def doc2vec():
	word2vec = get_wordvec()
	train_data, test_data = get_data()
	train_vec, test_vec = [], []

	for i, item in enumerate(train_data):
		if i % 100 == 0:
			print ('train data: {} / {}'.format(i, len(train_data)))
		word_ls = item['title'].encode('utf8').split(' ')
		word_ls2 = []
		for word in word_ls:
			try:
				word_ls2.append(map(eval, word2vec[word]))
			except:
				continue
		word_arr = np.array(word_ls2)
		word_arr_mean = np.mean(word_arr, axis=0)
		train_vec.append(word_arr_mean.tolist())

	for i, item in enumerate(test_data):
		if i % 100 == 0:
			print ('test data: {} / {}'.format(i, len(test_data)))
		word_ls = item['title'].encode('utf8').split(' ')
		word_ls2 = []
		for word in word_ls:
			try:
				word_ls2.append(map(eval, word2vec[word]))
			except:
				continue
		word_arr = np.array(word_ls2)
		word_arr_mean = np.mean(word_arr, axis=0)
		test_vec.append(word_arr_mean.tolist())

	print ('len of train data: {}'.format(len(train_data)))
	print ('len of train vec: {}'.format(len(train_vec)))
	print ('len of test data: {}'.format(len(test_data)))
	print ('len of test vec: {}'.format(len(test_vec)))

	with open('train_vec.small.json', 'w') as f:
		json.dump(train_vec, f)

	with open('test_vec.small.json', 'w') as f:
		json.dump(test_vec, f)


def calculate_cos(a, b):
	calculate_pair = [a, b]
	cos = cosine_similarity(calculate_pair)[0][1]
	return cos


def sort_by_value(d):
	return sorted(d.items(), key=lambda item: item[1], reverse=True)


def sort_news():
	with open('train_vec.small.json') as f:
		train_vec = json.load(f)
	with open('test_vec.small.json') as f:
		test_vec = json.load(f)

	similar_dic = dict()
	for i, item in enumerate(test_vec):
		dis_dic = dict()
		for j, item2 in enumerate(train_vec):
			dis_dic[j] = calculate_cos(item, item2)

		dis_ls = sort_by_value(dis_dic)[:10]

		similar_ls = []
		for k, item3 in enumerate(dis_ls):
			similar_ls.append(item3[0])

		similar_dic[i] = similar_ls

	return similar_dic


def prepare_cnn_data():
	cnn_data = []

	train_data = []
	with open('data.train.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)
			train_data.append(data)

	random.shuffle(train_data)

	for i in range(10000):
		candidate_comments = train_data[i]['comment']
		candidate = candidate_comments[0]
		for item in candidate_comments:
			if item[1] > candidate[1]:
				candidate = item
		cnn_data.append((1, (train_data[i]['title'], candidate[0])))

	print ("正样例构造完成")

	random.shuffle(train_data)

	for i in range(10000):
		candidate_comments = train_data[i+1]['comment']
		candidate = candidate_comments[0]
		for item in candidate_comments:
			if item[1] > candidate[1]:
				candidate = item
		cnn_data.append((0, (train_data[i]['title'], candidate[0])))

	print ("负样例构造完成")

	with open('cnn_data.txt', 'w') as f:
		for i, item in enumerate(cnn_data):
			f.write(str(item[0]) + '\t' + item[1][0].encode('utf-8') + ' ' + item[1][1].encode('utf-8') + '\n')

	print ("cnn训练数据保存完成")


def load_data(w2v_model,max_document_length = 1290):

    """Loads starter word-vectors and train/dev/test data."""
    # Load the starter word vectors
    print("Loading data...")
    x_text, y_test = data_helpers.load_data_and_labels(FLAGS.valid_data_file)
    y_test = np.argmax(y_test, axis=1)

    if(max_document_length == 0) :
        max_document_length = max([len(x.split(" ")) for x in x_text])

    print ('max_document_length = ' , max_document_length)

    x = data_helpers.get_text_idx(x_text,w2v_model.vocab_hash,max_document_length)


    return x,y_test



def cnn_score(a, b):
	score = 0

	# Data Parameters
	tf.flags.DEFINE_string("valid_data_file", "../data/cutclean_label_corpus10000.txt",
	                       "Data source for the positive data.")
	tf.flags.DEFINE_string("w2v_file", "../data/vectors.bin", "w2v_file path")

	# Eval Parameters
	tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
	tf.flags.DEFINE_string("checkpoint_dir", "./runs/1539589743/checkpoints/", "Checkpoint directory from training run")
	tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

	# Misc Parameters
	tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
	tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

	FLAGS = tf.flags.FLAGS
	FLAGS._parse_flags()
	print("\nParameters:")
	for attr, value in sorted(FLAGS.__flags.items()):
		print("{}={}".format(attr.upper(), value))
	print("")

	checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(
			allow_soft_placement=FLAGS.allow_soft_placement,
			log_device_placement=FLAGS.log_device_placement)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			# Load the saved meta graph and restore variables
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)

	# with sess.as_default():
			# Get the placeholders from the graph by name
			input_x = graph.get_operation_by_name("input_x").outputs[0]

			dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

			# Tensors we want to evaluate
			predictions = graph.get_operation_by_name("output/predictions").outputs[0]

			x_test, y_test = load_data(w2v_model, 1290)
			# Generate batches for one epoch
			batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

			# Collect the predictions here
			all_predictions = []

			for x_test_batch in batches:
				batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
				all_predictions = np.concatenate([all_predictions, batch_predictions])

	return score


def sort_comments(similar_dic):
	train_data, test_data = get_data()

	similar_comments = dict()

	for item in similar_dic.keys():
		comments = []
		for i, item2 in enumerate(similar_dic[item]):
			for j, item3 in enumerate(train_data[item2]['comment']):
				comments.append(item3[0])

		scores = dict()
		for j, item3 in enumerate(comments):
			scores[j] = cnn_score(test_data[item]['title'], item3)

		scores_sort = sort_by_value(scores)

		sort_comments_ls = []
		for k, item4 in enumerate(scores_sort):
			sort_comments_ls.append(comments[item4[0]])

		similar_comments[test_data[item]['title']] = sort_comments_ls[:10]

	with open('test_similar_comments.json', 'w') as f:
		json.dump(similar_comments, f)


if __name__ == '__main__':
	# similar_dic = sort_news()
	# sort_comments(similar_dic)
	# prepare_cnn_data()
	# get_small_data()
	# doc2vec()
	# news_len()
	# preprocess_content()
	print 'train len: '
	count_news_len('train.len.json')
	print '----'
	print 'dev len: '
	count_news_len('dev.len.json')
	print '----'
	print 'test len: '
	count_news_len('test.len.json')
	print '----'
