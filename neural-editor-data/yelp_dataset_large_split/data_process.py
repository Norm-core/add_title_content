# coding: utf8
import sys
import json
import jieba
import jieba.analyse
import random


def read_data(file):
	train_data = []
	with open(file, 'rb') as f:
		for i, line in enumerate(f):
			train_data.append(json.loads(line))

	return train_data


def ceshi_extract_keywords():
	test = '他会自己长大远去。'
	result = jieba.tokenize(test.decode('utf8'))
	word_dic = {}
	for item in result:
		word_dic[item[0]] = item[1]
	keywords = jieba.analyse.extract_tags(test)
	keywords_dic = {}
	for word in keywords:
		keywords_dic[word_dic[word]] = word
	keywords_ls = [keywords_dic[i] for i in sorted(keywords_dic.keys())]
	keywords_new = ' '.join(item for item in keywords_ls)

	print keywords_new


def extract_keywords(data):
	result = jieba.tokenize(data)
	word_dic = {}
	for item in result:
		word_dic[item[0]] = item[1]

	keywords = jieba.analyse.extract_tags(data)
	keywords_dic = {}
	for item in keywords:
		keywords_dic[word_dic[item]] = item
	keywords_ls = [keywords_dic[i] for i in sorted(keywords_dic.keys())]
	keywords_new = ' '.join(item for item in keywords_ls)

	return keywords_new


def construct_new_data(data):
	data_new = []
	for i in range(len(data)):
		if i % 100 == 0:
			print 'construct: {} / {}'.format(i, len(data))
		for item in data[i]['comment']:
			keywords_tmp = extract_keywords(item[0])
			data_new.append(keywords_tmp + '\t' + item[0] + '\t' + data[i]['title'])

	return data_new


def write_file(file, data):
	with open(file, 'w') as f:
		for i, item in enumerate(data):
			if i % 100 == 0:
				print('save: {} / {}'.format(i, len(data)))
			f.write(item.encode('utf8') + '\n')


def construct_seq2seq_data(file, file_src, file_tgt):
	data = read_data(file)

	src_data, tgt_data = [], []
	for i, item in enumerate(data):
		if i % 100 == 0: print 'construct: {} / {}'.format(i, len(data))
		for item2 in item['comment']:
			src_data.append(item['title'])
			tgt_data.append(item2[0])

	write_file(file_src, src_data)
	write_file(file_tgt, tgt_data)


def construct_train_data():
	train_data = read_data('tecent_data/article_commenting/data.train.json')
	train_data_new = construct_new_data(train_data)
	print 'saving train data...'
	write_file('train_title.tsv', train_data_new)

	valid_data = read_data('tecent_data/article_commenting/data.dev.json')
	valid_data_new = construct_new_data(valid_data)
	print 'saving valid data...'
	write_file('valid_title.tsv', valid_data_new)

	test_data = read_data('tecent_data/article_commenting/data.test.json')
	test_data_new = construct_new_data(test_data)
	print 'saving test data...'
	write_file('test_title.tsv', test_data_new)


def sort_by_value(d):
	return sorted(d.items(), key=lambda item: item[1])


def comments_sub():
	comments = []
	with open('tecent_data/comments_tecent.txt') as f:
		for line in f.readlines():
			comments.append(line.strip())

	comments2 = comments[:1000]
	with open('tecent_data/comments_tecent_sub.txt', 'w') as f:
		for item in comments2:
			f.write(item + '\n')


def extract_words():
	comments = []
	with open('../word_vectors/totals_tecent.txt') as f:
		for line in f.readlines():
			comments.append(line.strip())

	vocab = {}
	for i in range(len(comments)):
		if i % 1000 == 0:
			print '{} / {}'.format(i, len(comments))
		comment_ls = comments[i].split(' ')
		for j in range(len(comment_ls)):
			if comment_ls[j] in vocab:
				vocab[comment_ls[j]] += 1
			else:
				vocab[comment_ls[j]] = 0

	vocab2 = sort_by_value(vocab)

	vocab3 = []
	for i in range(len(vocab2)):
		vocab3.append(vocab2[i][0])

	vocab4 = vocab3[-100000:]
	with open('../word_vectors/vocab_100k.json', 'w') as f:
		json.dump(vocab4, f)


def continue_run(num):
	train_data = []
	with open('train_title.tsv') as f:
		for line in f.readlines():
			train_data.append(line.strip())

	train_data2 = train_data[num:]
	with open('train_con.tsv', 'w') as f:
		for item in train_data2:
			f.write(item + '\n')


def create_debug_file():
	train_data = []
	with open('train.tsv') as f:
		for line in f.readlines():
			train_data.append(line.strip())

	train = train_data[:1000]
	valid = train_data[1000:2000]
	test = train_data[2000:3000]

	with open('train_debug.tsv', 'w') as f:
		for item in train:
			f.write(item + '\n')

	with open('valid_debug.tsv', 'w') as f:
		for item in valid:
			f.write(item + '\n')

	with open('test_debug.tsv', 'w') as f:
		for item in test:
			f.write(item + '\n')


def create_debug_title():
	train_data = []
	with open('train.tsv') as f:
		for line in f.readlines():
			train_data.append(line.strip())

	train = train_data[:1000]
	valid = train_data[1000:2000]
	test = train_data[2000:3000]

	with open('train_debug.tsv', 'w') as f:
		for item in train:
			f.write(item + '\n')

	with open('valid_debug.tsv', 'w') as f:
		for item in valid:
			f.write(item + '\n')

	with open('test_debug.tsv', 'w') as f:
		for item in test:
			f.write(item + '\n')

	train_data = []
	with open('tecent_data/article_commenting/data.test.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)
			train_data.append(data)

	titles = []
	for item in train_data:
		titles.append(item['title'].encode('utf8'))

	train_title = titles[:100] * 10
	valid_title = titles[100:200] * 10
	test_title = titles[200:300]*10

	train_total, valid_total, test_total = [], [], []
	for i in range(len(train)):
		train_total.append(train[i] + '\t' + train_title[i])
	for i in range(len(valid)):
		valid_total.append(valid[i] + '\t' + valid_title[i])
	for i in range(len(test)):
		test_total.append(test[i] + '\t' + test_title[i])

	with open('train_debug.tsv', 'w') as f:
		for item in train_total:
			f.write(item + '\n')

	with open('valid_debug.tsv', 'w') as f:
		for item in valid_total:
			f.write(item + '\n')

	with open('test_debug.tsv', 'w') as f:
		for item in test_total:
			f.write(item + '\n')


def construct_train_title(file_T, file_tsv, file_out):
	keywords, comments, titles = [], [], []
	with open(file_T) as f:
		for line in f.readlines():
			titles.append(line.strip().split('\t')[0])

	with open(file_tsv) as f:
		for line in f.readlines():
			try:
				comments.append(line.strip().split('\t')[1])
				keywords.append(line.strip().split('\t')[0])
			except:
				keywords.append('none')
				comments.append(line.strip().split('\t')[0])

	assert len(keywords) == len(comments) == len(titles)

	data = []
	for i in range(len(keywords)):
		if keywords[i] != 'none':
			data.append(keywords[i] + '\t' + comments[i] + '\t' + titles[i])

	with open(file_out, 'w') as f:
		for item in data:
			f.write(item + '\n')


def construct_train_title_content(file_source, file_out):
	'''
	构造添加了新闻标题和内容的数据集，需要去掉新闻内容长度大于1000个词语的新闻。
	:param file_source: 新闻源文件
	:param file_out: 训练文件
	:return: 无
	'''
	keywords, comments, titles, contents = [],[],[],[]
	with open(file_source) as f:
		source_data = json.load(f)
	for i, news in enumerate(source_data):
		if i % 100 == 0:
			print '构造数据集：{}/{}'.format(i, len(source_data))
		if len(news['body'].split(' ')) > 1000:
			continue
		for j, comment in enumerate(news['comment']):
			keyword = extract_keywords(comment[0]).encode('utf8')
			if keyword != '':
				keywords.append(keyword)
			else:
				keywords.append('none')
			comments.append(comment[0].encode('utf8'))
			titles.append(news['title'].encode('utf8'))
			contents.append(news['body'].encode('utf8'))
	assert len(keywords) == len(comments) == len(titles) == len(contents)
	data = []
	for i in range(len(keywords)):
		if keywords[i] != 'none':
			data.append(keywords[i] + '\t' + comments[i] + '\t' + titles[i] + ' ' + contents[i])
	with open(file_out, 'w') as f:
		for item in data:
			f.write(item + '\n')


def shuffle_data(file):
	with open(file) as f:
		data = f.readlines()
	random.shuffle(data)
	with open(file, 'w') as f:
		for item in data:
			f.write(item)


if __name__ == '__main__':
	# construct_train_data()
	# comments_sub()
	# extract_words()
	# continue_run(2951*50)
	# test_extract_keywords()
	# create_debug_file()
	# create_debug_title()
	# construct_train_title('tecent_data/trainT.txt', 'tecent_data/train.tsv', 'tecent_data/train_title.tsv')
	# construct_train_title('tecent_data/devT.txt', 'tecent_data/valid.tsv', 'tecent_data/valid_title.tsv')
	# construct_train_title('tecent_data/testT.txt', 'tecent_data/test.tsv', 'tecent_data/test_title.tsv')
	# construct_train_title_content('tecent_data/article_commenting/newdata.test.clean.json', 'test_title_content.tsv')
	# construct_train_title_content('tecent_data/article_commenting/newdata.dev.clean.json', 'valid_title_content.tsv')
	# construct_train_title_content('tecent_data/article_commenting/newdata.train.clean.json', 'train_title_content.tsv')
	# extract_words()
	# shuffle_data('train_title_content_30w.tsv')
	construct_seq2seq_data('tecent_data/article_commenting/data.train.json', 'src-train.txt', 'tgt-train.txt')
	construct_seq2seq_data('tecent_data/article_commenting/data.dev.json', 'src-val.txt', 'tgt-val.txt')
	construct_seq2seq_data('tecent_data/article_commenting/data.test.json', 'src-test.txt', 'tgt-test.txt')
