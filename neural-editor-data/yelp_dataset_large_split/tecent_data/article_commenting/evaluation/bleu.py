# python3
from sumeval.metrics.bleu import BLEUCalculator
import json
import numpy as np


def bleu_per_test_data(comments1, comments2):
	bleu = BLEUCalculator()

	bleu_1_ls = []

	for i in range(len(comments1)):
		tmp = []
		for j in range(len(comments2)):
			bleu_1 = bleu.bleu(
				summary=comments1[i],
				references=comments2[j])
			tmp.append(bleu_1)

		bleu_1_ls.append(np.mean(np.array(tmp)))

		bleu_1_final = np.mean(np.array(bleu_1_ls))

	return bleu_1_final


def Bleu():
	bleu_ls = []

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
		bleu_tmp = bleu_per_test_data(retrieve_comments, gold_comments)
		bleu_ls.append(bleu_tmp)

	bleu_total = np.mean(np.array(bleu_ls))

	return bleu_total


if __name__ == '__main__':
	bleu_1 = Bleu()
	print('Blue_1: {}'.format(bleu_1))
