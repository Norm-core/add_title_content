# coding:utf-8
import json
import jieba
import jieba.analyse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


class evalQuality:
	def __init__(self, news_file, comments_file):
		'''
		导入新闻标题内容和生成评论
		:param news_file:
		:param comments_file:
		'''
		self.news = []
		with open(news_file) as f:
			for line in f:
				data = json.loads(line)
				self.news.append(data)

		self.title_contents = dict()
		for i, item in enumerate(self.news):
			title = item['title']
			content = item['body']
			self.title_contents[title] = content

		with open(comments_file) as f:
			self.generate_comments = json.load(f)


	def extract_keywords(self, data):
		result = jieba.tokenize(data)
		word_dic = {}
		for item in result:
			word_dic[item[0]] = item[1]

		keywords = jieba.analyse.extract_tags(data)
		keywords_dic = {}
		for item in keywords:
			keywords_dic[word_dic[item]] = item
		keywords_ls = [keywords_dic[i] for i in sorted(keywords_dic.keys())]

		return keywords_ls


	def comments_in_news(self):
		'''
		计算每篇新闻生成的评论的关键词在新闻中的比例，同时返回关键词数量
		:return:
		'''
		bili = []
		average_keywords = []
		for title in self.title_contents:
			comments_keywords = []
			tmp_keywords = []
			for comment in self.generate_comments[title]:
				comments_keywords.append(self.extract_keywords(comment))
				tmp_keywords.append(len(self.extract_keywords(comment)))

			# 计算每篇新闻的评论的平均关键词数量
			average_keywords.append(np.mean(np.array(tmp_keywords)))

			# 计算对应于每篇新闻的所有生成评论的关键词匹配比例
			tmp = []
			for item in comments_keywords:
				num_in = 0
				for item2 in item:
					if item2 in self.title_contents[title]:
						num_in += 1
				try:
					tmp.append(float(num_in) / float(len(item)))
				except ZeroDivisionError:
					tmp.append(0.0)

			bili.append(np.mean(np.array(tmp)))

		return bili, np.mean(np.array(bili)), np.std(np.array(bili)),\
		       average_keywords, np.mean(np.array(average_keywords)), np.std(np.array(average_keywords))


class evalGoldQuality:
	def __init__(self, news_file):
		'''
		导入测试集新闻内容和对应的评论
		:param news_file:
		'''
		self.news = []
		with open(news_file) as f:
			for line in f:
				data = json.loads(line)
				self.news.append(data)

		self.title_contents = dict()
		self.gold_comments = dict()
		for i, item in enumerate(self.news):
			title = item['title']
			content = item['body']
			self.title_contents[title] = content
			comments = []
			for j, item2 in enumerate(item['comment']):
				comments.append(item2[0])
			self.gold_comments[title] = comments


	def extract_keywords(self, data):
		result = jieba.tokenize(data)
		word_dic = {}
		for item in result:
			word_dic[item[0]] = item[1]

		keywords = jieba.analyse.extract_tags(data)
		keywords_dic = {}
		for item in keywords:
			keywords_dic[word_dic[item]] = item
		keywords_ls = [keywords_dic[i] for i in sorted(keywords_dic.keys())]

		return keywords_ls


	def comments_in_news(self):
		'''
		计算每篇新闻生成的评论的关键词在新闻中的比例，同时返回关键词数量
		:return:
		'''
		bili = []
		average_keywords = []
		for title in self.title_contents:
			comments_keywords = []
			tmp_keywords = []
			for comment in self.gold_comments[title]:
				comments_keywords.append(self.extract_keywords(comment))
				tmp_keywords.append(len(self.extract_keywords(comment)))

			# 计算每篇新闻的评论的平均关键词数量
			average_keywords.append(np.mean(np.array(tmp_keywords)))

			# 计算对应于每篇新闻的所有生成评论的关键词匹配比例
			tmp = []
			for item in comments_keywords:
				num_in = 0
				for item2 in item:
					if item2 in self.title_contents[title]:
						num_in += 1
				try:
					tmp.append(float(num_in) / float(len(item)))
				except ZeroDivisionError:
					tmp.append(0.0)

			bili.append(np.mean(np.array(tmp)))

		return bili, np.mean(np.array(bili)), np.std(np.array(bili)),\
		       average_keywords, np.mean(np.array(average_keywords)), np.std(np.array(average_keywords))


	def draw_bar(self, ls, file_save, title):
		plt.bar(range(len(ls)), ls, color='b')
		plt.title(title)
		plt.savefig(file_save)


def draw_bar(ls, file_save):
	plt.figure(figsize=[20,10])
	plt.subplot(231)
	plt.bar(range(len(ls[0])), ls[0], color='b')
	plt.title('seq2seq')
	plt.subplot(232)
	plt.bar(range(len(ls[1])), ls[1], color='b')
	plt.title('att')
	plt.subplot(233)
	plt.bar(range(len(ls[2])), ls[2], color='b')
	plt.title('att-tc')
	plt.subplot(234)
	plt.bar(range(len(ls[3])), ls[3], color='b')
	plt.title('edit_vector')
	plt.subplot(235)
	plt.bar(range(len(ls[4])), ls[4], color='b')
	plt.title('edit_vector+title')
	plt.subplot(236)
	plt.bar(range(len(ls[5])), ls[5], color='b')
	plt.title('edit_vector+title+content')
	plt.savefig(file_save)


def compute_correlation(df):
	print('pearson:')
	print(df.corr())
	print('-----')
	print('spearman:')
	print(df.corr('spearman'))


if __name__ == '__main__':
	# quality = evalQuality('../newdata.test.json', '../test_content_att_generate_comments_25k.json')
	# ratio_ls, ratio, ratio_var,\
	# keywords_num_ls, keywords_num, keywords_num_var= quality.comments_in_news()
	# print('ratio: {}'.format(ratio))
	# print('keywords_num: {}'.format(keywords_num))
	# quality.draw_bar(ratio_ls, '../plot/test_content_att_25k.jpg')
	# quality.draw_bar(keywords_num_ls, '../plot/test_content_att_25k_keywords.jpg')
	# print('ratio std:{}'.format(ratio_var))
	# print('keywords_num_std: {}'.format(keywords_num_var))

	quality1 = evalQuality('../newdata.test.json', '../test_title_seq2seq_generate_comments_25k.json')
	match_ls1, _, _, keywords_ls1, _, _ = quality1.comments_in_news()
	quality2 = evalQuality('../newdata.test.json', '../test_title_att_generate_comments_25k.json')
	match_ls2, _, _, keywords_ls2, _, _ = quality2.comments_in_news()
	quality3 = evalQuality('../newdata.test.json', '../test_content_att_generate_comments_25k.json')
	match_ls3, _, _, keywords_ls3, _, _ = quality3.comments_in_news()
	quality4 = evalQuality('../newdata.test.json', '../test1_generate_comments.json')
	match_ls4, _, _, keywords_ls4, _, _ = quality4.comments_in_news()
	quality5 = evalQuality('../newdata.test.json', '../test_add_title_generate_comments_25k.json')
	match_ls5, _, _, keywords_ls5, _, _ = quality5.comments_in_news()
	quality6 = evalQuality('../newdata.test.json', '../test_add_title_content_generate_comments_4k.json')
	match_ls6, _, _, keywords_ls6, _, _ = quality6.comments_in_news()

	# match_ls = [match_ls1, match_ls2, match_ls3, match_ls4, match_ls5, match_ls6]
	# keywords_ls = [keywords_ls1, keywords_ls2, keywords_ls3, keywords_ls4, keywords_ls5, keywords_ls6]
	# draw_bar(match_ls, '../plot/match.jpg')
	# draw_bar(keywords_ls, '../plot/keywords.jpg')

	quality = evalGoldQuality('../newdata.test.json')
	ratio_ls, ratio, ratio_var, keywords_num_ls, keywords_num, keywords_num_var= quality.comments_in_news()
	# print 'average match ratio: {}'.format(ratio)
	# print 'match ratio var: {}'.format(ratio_var)
	# print 'average keywords: {}'.format(keywords_num)
	# print 'keywords var: {}'.format(keywords_num_var)
	# quality.draw_bar(ratio_ls, '../plot/gold_match.jpg', 'gold match ratio')
	# quality.draw_bar(keywords_num_ls, '../plot/gold_keywords', 'gold keywords num')

	df_ratio = pd.DataFrame({'gold': ratio_ls,
	                         'seq2seq': match_ls1, 'att': match_ls2, 'att-tc': match_ls3,
	                         'edit-vector': match_ls4, 'edit-vector-t': match_ls5, 'edit-vector-tc': match_ls6})
	df_keywords = pd.DataFrame({'gold': keywords_num_ls,
	                           'seq2seq': keywords_ls1, 'att': keywords_ls2, 'att-tc': keywords_ls3,
	                            'edit-vector': keywords_ls4, 'edit-vector-t': keywords_ls5, 'edit-vector-tc': keywords_ls6})

	print('match correlation:')
	compute_correlation(df_ratio)
	print
	print('keywords correlation:')
	compute_correlation(df_keywords)
