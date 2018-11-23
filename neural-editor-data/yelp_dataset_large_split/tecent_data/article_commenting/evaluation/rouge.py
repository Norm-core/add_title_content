# python3
from sumeval.metrics.rouge import RougeCalculator
import json
import numpy as np


def rouge_per_test_data(comments1, comments2):
	rouge = RougeCalculator(stopwords=True)

	rouge_l_ls = []

	for i in range(len(comments1)):
		rouge_l = rouge.rouge_l(
			summary=comments1[i],
			references=comments2)
		rouge_l_ls.append(rouge_l)

	rouge_l_final = np.mean(np.array(rouge_l_ls))

	return rouge_l_final


def rouge():
	rouge_ls = []

	with open('../test_similar_comments.json') as f:
		similar_comments = json.load(f)

	test_data = []
	with open('../newdata.test.json') as f:
		for line in f:
			data = json.loads(line)
			test_data.append(data)

	for i in range(len(test_data)):
		if i % 50 == 0:
			print('进度：{}/{}'.format(i, len(test_data)))
		news_title = test_data[i]['title']
		gold_comments = [c[0] for c in test_data[i]['comment']]
		retrieve_comments = similar_comments[news_title]
		rouge_tmp = rouge_per_test_data(retrieve_comments, gold_comments)
		rouge_ls.append(rouge_tmp)

	rouge_total = np.mean(np.array(rouge_ls))

	return rouge_total


if __name__ == '__main__':
	rouge_l = rouge()
	print('Rouge_L: {}'.format(rouge_l))
