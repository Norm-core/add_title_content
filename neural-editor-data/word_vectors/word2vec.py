# coding: utf-8
import gensim
import json


def construct_comments():
	comments = []
	with open('../yelp_dataset_large_split/tecent_data/train.tsv') as f:
		for line in f.readlines():
			try:
				comments.append(line.strip().split('\t')[1])
			except:
				continue
	with open('../yelp_dataset_large_split/tecent_data/valid.tsv') as f:
		for line in f.readlines():
			try:
				comments.append(line.strip().split('\t')[1])
			except:
				continue
	with open('../yelp_dataset_large_split/tecent_data/test.tsv') as f:
		for line in f.readlines():
			try:
				comments.append(line.strip().split('\t')[1])
			except:
				continue
	print('total number of comments: {}'.format(len(comments)))
	with open('../yelp_dataset_large_split/tecent_data/comments_tecent.txt', 'w') as f:
		for item in comments:
			f.write(item + '\n')


def construct_comments_titles():
	comments = []
	with open('../yelp_dataset_large_split/tecent_data/comments_tecent.txt') as f:
		for line in f.readlines():
			comments.append(line.strip())

	titles = []

	test_data = []
	with open('data.test.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)
			test_data.append(data)
	print 'finish getting test data.'

	train_data = []
	with open('data.train.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)
			train_data.append(data)
	print 'finish getting train data.'

	for i, item in enumerate(train_data):
		titles.append(item['title'].encode('utf8'))
	for i, item in enumerate(test_data):
		titles.append(item['title'].encode('utf8'))

	totals = comments + titles
	with open('totals_tecent.txt', 'w') as f:
		for item in totals:
			f.write(item + '\n')

def word2vec():
	# comments3.txt: 每一行为一条评论，由所有评论组成的文本
    with open('totals_tecent.txt') as f:
        tmp = f.readlines()
    data = []
    for i in range(len(tmp)):
        data.append(tmp[i].strip())

    train_data = []
    for i in range(len(data)):
        tmp = data[i].strip().split(' ')
        train_data.append(tmp)

    model = gensim.models.Word2Vec(train_data, min_count=1)
    model.wv.save_word2vec_format('vector_totals_tecent.txt', binary=False)


def select_vec():
	with open('vocab_100k.json') as f:
		vocab = json.load(f)

	vec = {}
	with open('vector_totals_tecent.txt') as f:
		f.readline()
		for line in f.readlines():
			vec[line.strip().split(' ')[0]] = line.strip().split(' ')[1:]

	vec2 = []
	for item in vocab:
		if item != '':
			vec2.append([item.encode('utf-8')] + vec[item.encode('utf-8')])

	with open('vector_totals_tecent_100k.txt', 'w') as f:
		f.write('100000 100\n')
		for item in vec2:
			f.write(' '.join(item) + '\n')


if __name__ == '__main__':
	# construct_comments()
	# construct_comments_titles()
	# word2vec()
	select_vec()
