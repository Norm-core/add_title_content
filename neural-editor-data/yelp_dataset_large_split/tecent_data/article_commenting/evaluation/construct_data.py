# coding: utf8
import json


titles = []
with open('../valid_title.tsv') as f:
	for line in f.readlines():
		titles.append(line.strip())

comments = []
with open('../result_title_att_25k.txt') as f:
	for line in f.readlines():
		comments.append(line.strip())

comments_test1 = {}
for i in range(len(titles)):
	# comments_test1[titles[i]] = comments[i:i+10]
	comments_test1[titles[i]] = [comments[i]] * 10

with open('../test_title_att_generate_comments_25k.json', 'w') as f:
	json.dump(comments_test1, f)
