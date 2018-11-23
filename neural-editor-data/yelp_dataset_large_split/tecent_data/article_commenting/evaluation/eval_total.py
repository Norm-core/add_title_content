# coding: utf8
# python2
import nlgeval
from nlgeval import NLGEval
import numpy as np
import json

n = NLGEval(no_skipthoughts=True, no_glove=True)
n_w = NLGEval(no_skipthoughts=True, no_glove=True, no_weight=False)

def single_meteor(h, ref, weight):
	commonTerms = sum(1 for w in h if w in ref)
	nChunks = 0
	prevTrans = False
	if (commonTerms != 0):
		for i in range(0, len(ref)):
			if ref[i] not in h and prevTrans == False:
				prevTrans = True
				nChunks += 1
			if ref[i] in h:
				prevTrans = False
			if i == len(ref) - 1 and ref[i] in h:
				nChunks += 1
		alpha = 0.77
		# print "nChunks-->"+str(nChunks)
		precision = float(commonTerms) / float(len(h))
		recall = float(commonTerms) / float(len(ref))
		F = (precision * recall) / ((alpha * precision) + ((1 - alpha) * recall))
		Penalty = (0.5 * nChunks) / (commonTerms)
		MeteorScore = (1 - Penalty) * weight * F
		return MeteorScore
	else:
		return 0


def cal_meteor(comments1, comments2, weights):
	meteor_ls = []
	for i in range(len(comments1)):
		meteor = 0
		for j in range(len(comments2)):
			meteor_tmp = single_meteor(comments1[i].split(' '), comments2[j].split(' '), weights[j])
			if meteor_tmp > meteor:
				meteor = meteor_tmp
			meteor_ls.append(meteor)
	return np.mean(np.array(meteor_ls))


def eval_per_test_data(n, comments1, comments2, weights):
	'''
	计算每个测试集新闻包含的评论和检索出来的评论的测试指标
	:param n:
	:param comments1:
	:param comments2:
	:return:
	'''
	scores = n.compute_metrics(ref_list=[[c]*len(comments1) for c in comments2], hyp_list=comments1,
	                           weight_list=weights)

	METEOR = scores['METEOR']
	Rouge_L = scores['ROUGE_L']
	CIDEr = scores['CIDEr']
	BLEU_1 = scores['Bleu_1']

	return METEOR, Rouge_L, CIDEr, BLEU_1


def eval_test_data():
	with open('../test_title_att_generate_comments_25k.json') as f:
		similar_comments = json.load(f)

	test_data = []
	with open('../newdata.test.json') as f:
		for line in f:
			data = json.loads(line)
			test_data.append(data)

	meteor_ls, rouge_l_ls, cider_ls, bleu_1_ls = [], [], [], []
	meteor_w_ls, rouge_l_w_ls, cider_w_ls, bleu_1_w_ls = [], [], [], []

	for i in range(len(test_data)):
		print('测试进度：{}/{}'.format(i, len(test_data)))

		news_title = test_data[i]['title']
		gold_comments = [c[0] for c in test_data[i]['comment']]
		gold_comments_weight = [c[2] for c in test_data[i]['comment']]
		retrieve_comments = [item.replace('<unk> ', '').replace(' </s>', '') for item in similar_comments[news_title]]
		# retrieve_comments = [item for item in similar_comments[news_title]]
		meteor, rouge_l, cider, bleu_1 = eval_per_test_data(n, retrieve_comments, gold_comments, gold_comments_weight)
		_, rouge_l_w, cider_w, bleu_1_w = eval_per_test_data(n_w, retrieve_comments, gold_comments, gold_comments_weight)
		meteor_w = cal_meteor(retrieve_comments, gold_comments, gold_comments_weight)

		meteor_ls.append(meteor)
		rouge_l_ls.append(rouge_l)
		cider_ls.append(cider)
		bleu_1_ls.append(bleu_1)

		meteor_w_ls.append(meteor_w)
		rouge_l_w_ls.append(rouge_l_w)
		cider_w_ls.append(cider_w)
		bleu_1_w_ls.append(bleu_1_w)

	meteor_final = np.mean(np.array(meteor_ls))
	rouge_l_final = np.mean(np.array(rouge_l_ls))
	cider_final = np.mean(np.array(cider_ls))
	bleu_1_final = np.mean(np.array(bleu_1_ls))

	meteor_w_final = np.mean(np.array(meteor_w_ls))
	rouge_l_w_final = np.mean(np.array(rouge_l_w_ls))
	cider_w_final = np.mean(np.array(cider_w_ls))
	bleu_1_w_final = np.mean(np.array(bleu_1_w_ls))

	return meteor_final, rouge_l_final, cider_final, bleu_1_final,\
			meteor_w_final, rouge_l_w_final, cider_w_final, bleu_1_w_final


if __name__ == '__main__':
	meteor, rouge_l, cider, bleu_1,\
		meteor_w, rouge_l_w, cider_w, bleu_1_w= eval_test_data()
	print 'meteor: {}'.format(meteor)
	print 'meteor_w: {}'.format(meteor_w)

	print 'rouge_l: {}'.format(rouge_l)
	print 'rouge_l_w: {}'.format(rouge_l_w)

	print 'bleu_1: {}'.format(bleu_1)
	print 'bleu_1_w: {}'.format(bleu_1_w)
