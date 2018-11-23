# coding: utf8
import json
import jieba
import jieba.analyse


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


def create():
	with open('test_similar_comments.json') as f:
		similar_comments = json.load(f)

	test_data = []
	retrieve_comments = []
	for key in similar_comments.keys():
		test_data.append(key)
		for i in range(len(similar_comments[key])):
			retrieve_comments.append(similar_comments[key][i])

	valid = []
	for i in range(len(retrieve_comments)):
		print '构造测试集：{}/{}'.format(i, len(retrieve_comments))
		valid.append(extract_keywords(retrieve_comments[i]) + '\t' + retrieve_comments[i])

	with open('valid_test1.tsv', 'w') as f:
		for item in valid:
			f.write(item.encode('utf8') + '\n')

	with open('valid_test1_title.tsv', 'w') as f:
		for item in test_data:
			f.write(item.encode('utf8') + '\n')


def create_test_add_title():
	titles = []
	with open('valid_title.tsv') as f:
		for line in f.readlines():
			for num in range(10):
				titles.append(line.strip())

	test_data = []
	with open('valid_test1.tsv') as f:
		for line in f.readlines():
			test_data.append(line.strip())

	assert len(titles) == len(test_data)

	test_add_title = []
	for i in range(len(test_data)):
		test_add_title.append(test_data[i] + '\t' + titles[i])

	with open('valid_test2_add_title_35k.tsv', 'w') as f:
		for item in test_add_title:
			f.write(item + '\n')


def create_test_add_title_content():
	titles = []
	with open('valid_title.tsv') as f:
		for line in f.readlines():
			for num in range(10):
				titles.append(line.strip())

	contents = []
	with open('newdata.test.clean.json') as f:
		test_data = json.load(f)

	title_content_dic = {}
	for i, item in enumerate(test_data):
		title_content_dic[test_data[i]['title']] = test_data[i]['body']

	for i, item in enumerate(titles):
		contents.append(title_content_dic[item.decode('utf8')])

	test_add_title = []
	with open('valid_test2_add_title.tsv') as f:
		for line in f.readlines():
			test_add_title.append(line.strip())

	test_add_title_content = []
	for i, item in enumerate(test_add_title):
		test_add_title_content.append(item + ' ' + contents[i].encode('utf8'))

	assert len(test_add_title_content) == len(contents)

	with open('valid_test3_add_title_content.tsv', 'w') as f:
		for i, item in enumerate(test_add_title_content):
			f.write(item + '\n')


if __name__ == '__main__':
	# create()
	# create_test_add_title()
	create_test_add_title_content()
