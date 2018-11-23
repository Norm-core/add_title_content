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

def get_small_data():
	with open('newdata.small.train.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)

	small_data = []
	for i in range(len(data)):
		small_data.append(data[i]['title'].encode('utf8'))
		small_data.append(data[i]['body'].encode('utf8'))
		for j in range(len(data[i]['comment'])):
			small_data.append(data[i]['comment'][j][0].encode('utf8'))

	with open('newdata.small.test.json') as f:
		for i, eachline in enumerate(f):
			data = json.loads(eachline)

	for i in range(len(data)):
		small_data.append(data[i]['title'].encode('utf8'))
		small_data.append(data[i]['body'].encode('utf8'))
		for j in range(len(data[i]['comment'])):
			small_data.append(data[i]['comment'][j][0].encode('utf8'))

	with open('newsmall_data.txt', 'w') as f:
		for k in range(len(small_data)):
			f.write(small_data[k] + '\n')


def get_new_data():
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

	total_data = []
	for i in range(len(test_data)):
		total_data.append(test_data[i]['title'].encode('utf8'))
		total_data.append(test_data[i]['body'].encode('utf8'))
		for j in range(len(test_data[i]['comment'])):
			total_data.append(test_data[i]['comment'][j][0].encode('utf8'))

	for i in range(len(train_data)):
		total_data.append(train_data[i]['title'].encode('utf8'))
		total_data.append(train_data[i]['body'].encode('utf8'))
		for j in range(len(train_data[i]['comment'])):
			total_data.append(train_data[i]['comment'][j][0].encode('utf8'))

	with open('newdata.txt', 'w') as f:
		for k in range(len(total_data)):
			f.write(total_data[k] + '\n')


def word2vec():
	# comments3.txt: 每一行为一条评论，由所有评论组成的文本
    with open('newdata.txt') as f:
        tmp = f.readlines()
    data = []
    for i in range(len(tmp)):
        data.append(tmp[i].strip())

    train_data = []
    for i in range(len(data)):
        tmp = data[i].strip().split(' ')
        train_data.append(tmp)

    model = gensim.models.Word2Vec(train_data, min_count=1)
    model.wv.save_word2vec_format('newvector.txt', binary=False)


def select_vec():
	with open('../yelp_dataset_large_split/tecent_data/vocab_30k.json') as f:
		vocab = json.load(f)

	vec = {}
	with open('vector_tecent.txt') as f:
		f.readline()
		for line in f.readlines():
			vec[line.strip().split(' ')[0]] = line.strip().split(' ')[1:]

	vec2 = []
	for item in vocab:
		if item != '':
			vec2.append([item.encode('utf-8')] + vec[item.encode('utf-8')])

	with open('vector_tecent_30k.txt', 'w') as f:
		f.write('30000 100\n')
		for item in vec2:
			f.write(' '.join(item) + '\n')


if __name__ == '__main__':
	# get_new_data()
	word2vec()
