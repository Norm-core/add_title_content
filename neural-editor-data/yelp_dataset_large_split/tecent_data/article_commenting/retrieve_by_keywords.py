# coding: utf-8
import json
import jieba
import jieba.analyse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math


def get_wordvec():
	word2vec = {}
	with open('newvector.txt') as f:
		f.readline()
		for line in f.readlines():
			word2vec[line.strip().split(' ')[0]] = line.strip().split(' ')[1:]
	return word2vec


def get_data():
	print ('start getting data...')

	test_data = []
	with open('newdata.test.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)
			test_data.append(data)
	print ('finish getting test data.')

	print '测试集数量：{}'.format(len(test_data))

	train_data = []
	with open('newdata.train.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)
			train_data.append(data)
	print ('finish getting train data.')

	print '训练集数量：{}'.format(len(train_data))

	return train_data, test_data


def get_small_data():
	test_data = []
	with open('newdata.test.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)
			test_data.append(data)
	print ('finish getting test data.')

	train_data = []
	with open('newdata.train.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)
			train_data.append(data)
	print ('finish getting train data.')

	test_data_small = test_data[:20]
	train_data_small = train_data[:100]

	with open('newdata.small.train.json', 'w') as f:
		json.dump(train_data_small, f)
	with open('newdata.small.test.json', 'w') as f:
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

	with open('train_vec.json', 'w') as f:
		json.dump(train_vec, f)

	with open('test_vec.json', 'w') as f:
		json.dump(test_vec, f)


def calculate_cos(a, b):
	# calculate_pair = [a, b]
	cos = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]
	return cos


def sort_by_value(d):
	return sorted(d.items(), key=lambda item: item[1], reverse=True)


def sort_news():
	with open('train_vec.json') as f:
		train_vec = json.load(f)
	with open('test_vec.json') as f:
		test_vec = json.load(f)

	similar_dic = dict()
	for i, item in enumerate(test_vec):
		if i %10 == 0:
			print '测试集相似新闻检索：{}/{}'.format(i, len(test_vec))
		dis_dic = dict()
		for j, item2 in enumerate(train_vec):
			if type(item2) != list:
				continue
			dis_dic[j] = calculate_cos(np.array(item), np.array(item2))

		dis_ls = sort_by_value(dis_dic)[:10]

		similar_ls = []
		for k, item3 in enumerate(dis_ls):
			similar_ls.append(item3[0])

		similar_dic[i] = similar_ls

	print '相似新闻检索完毕'

	return similar_dic


def keywords_score(a, b):
	key_a = jieba.analyse.extract_tags(a)
	key_b = jieba.analyse.extract_tags(b)
	# print key_a, key_b
	key1 = key_a + key_b
	key2 = list(set(key_a + key_b))
	score = (float(len(key1) - len(key2)) / float(len(key1))) / 0.5
	return score


def sort_comments(similar_dic):
	train_data, test_data = get_data()

	similar_comments = dict()

	count = 0
	for item in similar_dic.keys():
		if count % 10 == 0:
			print '测试集相似评论检索中：{}/{}'.format(count, len(similar_dic))
		count += 1

		comments = []
		for i, item2 in enumerate(similar_dic[item]):
			for j, item3 in enumerate(train_data[item2]['comment']):
				comments.append(item3[0])

		scores = dict()
		for j, item3 in enumerate(comments):
			scores[j] = keywords_score(test_data[item]['title'] + ' ' + test_data[item]['body'], item3)

		scores_sort = sort_by_value(scores)

		sort_comments_ls = []
		for k, item4 in enumerate(scores_sort):
			sort_comments_ls.append(comments[item4[0]])

		similar_comments[test_data[item]['title']] = sort_comments_ls[:10]

	print '相似评论检索完毕'

	with open('test_similar_comments.json', 'w') as f:
		json.dump(similar_comments, f)


if __name__ == '__main__':
	doc2vec()
	similar_news = sort_news()
	sort_comments(similar_news)
