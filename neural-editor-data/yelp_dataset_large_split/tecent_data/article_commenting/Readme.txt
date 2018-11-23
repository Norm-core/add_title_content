This directory contains the dataset for the ACL 2018 short paper ``Automatic Article Commenting: the Task and Dataset".

1. Note
The dataset is crawled from http://kuaibao.qq.com, it is used for research only but is prohibited for any commercial purpose. If you find it is useful in your research, please cite our paper shown at the end of this doc. 

2. Dataset Usage 

a.	About format
The files are with json format in each line, and particularly each line in both data.train.json and data.dev.json indicates an article including title, comment, url and channel etc as follows:
=========
"channel": "product","comment":[["I am looking forward to iPhone 8 .",9], ["Exciting .", 4]], "title":"Apple's iPhone 8 event is happening in Sept.", "url":" http://kuaibao.qq.com"
=========
Regarding to this article, its title is "Apple's iPhone 8 event is happening in Sept.", channel is pet, url shows the website of the article, and it contains 2 comments with 9 and 4 upvotes, respectively.  

Besides the upvotes for comment in both data.train.json and data.dev.json, data.test.json includes manual score which evaluates the quality of this comment from human annotators as follows:
=========
"channel": "product","comment":[["I am looking forward to iPhone 8 .",9, 0.7], ["Exciting .", 4, 0.3]], "title":"Apple's iPhone 8 event is happening in Sept."
=========
where 0.7 and 0.3 are the scores indicating the quality of these two comments. 

b.	How to get the content for these articles?
Note that due to the copyright permission, the dataset does not contain the content of any articles, which can be downloaded by users using our toolkit "preprocess.py" in this directory.

3. Reference

Lianhui Qin, Lemao Liu, Victoria Bi, Yan Wang, Xiaojiang Liu, Zhiting Hu, Hai Zhao and Shuming Shi. 2018. Automatic Article Commenting: the Task and Dataset. In Proceedings of ACL. 

