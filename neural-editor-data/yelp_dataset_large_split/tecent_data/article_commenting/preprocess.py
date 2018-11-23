#!/usr/bin/python
# -*- encoding:utf-8 -*-
#读取文件中的内容，并进行分词
import json
import time
import numpy as np
import re
import os
import pandas as pd
import cPickle as pickle
import urllib2
import sys

""" 
To fetch the body for each url and then save the news including body into a file
python preprocess.py
"""

reload(sys)
sys.setdefaultencoding('utf-8')

def print_json(jline, fout):
    fout.write(json.dumps(jline, sort_keys=True, 
                           separators=(',', ': ')))
    fout.write('\n')


def extract_txt(url):
    global Max_Num
    Max_Num = 100
    for i in range(Max_Num):
        try:
            response = urllib2.urlopen(url, timeout=100).read()
            break
        except:
            if i < Max_Num - 1:
                continue
            else:
                print('URLError: <urlopen error timed out> All times is failed ')
    # response = urllib2.urlopen(url).read()
    #m = re.search('<p class=\"text\">(.+?)<\/p>', text)
    found = re.findall('<p class=\"text\">(.+?)<\/p>', response)
    found = '\n'.join(found)
    text = re.sub('<[^<]+?>', '', found)
    return text


def wrapper(in_name, out_name):
    f = open(in_name,'rb')
    f_write = open(out_name, 'wb')
    skipped_num = 0
    for i, eachline in enumerate(f):
        try:
            data = json.loads(eachline)
            url = data['url']
            data['body'] = extract_txt(url)
            print_json(data, f_write)
            if data['body'] == "":
                #print >>sys.stderr, url
                #print >>sys.stderr, 'fetch url unsuccessfully and skip the %d_th line in %s'%(i+1, in_name)
                skipped_num += 1
        except:
            continue
    f.close()
    f_write.close()
    print >>sys.stderr, 'empty body for %d lines in %s'%(skipped_num, in_name)

# filelist = ['data.dev.json','data.test.json','data.train.json']
# outfilelist = ['newdata.dev.json','newdata.test.json','newdata.train.json']
filelist = ['data.train.json']
outfilelist = ['newdata.train.json']
for x, y in zip(filelist,outfilelist):
    wrapper(x, y)
