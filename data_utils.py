#!/usr/bin/env python
# encoding: utf-8

import os
import string
from tqdm import tqdm
from zhon import hanzi
from pyltp import Segmentor
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer

ltp_segmentor = Segmentor()
ltp_segmentor.load('../ltp_data/cws.model')
def segment(corpus):
    ret = []
    for sent in corpus:
        tokens = ltp_segmentor.segment(sent)
        ret.append(' '.join(tokens))
    return ret

def del_punctuation(text):
    text = text.translate(None, string.punctuation)
    text = text.translate(None, hanzi.punctuation.encode('utf-8'))
    return text

def load_data(data_dir, fname):
    fpath = os.path.join(data_dir, fname)
    print 'Loading data from %s'%fpath
    tree = ET.ElementTree(fpath)
    root = tree.getroot()
    dic={}
    for child in root[0]:
        dic[child.tag]=[]
    for item in tqdm(root):
        for child in item:
            dic[child.tag].append(del_punctuation(child.text))
    return dic

tf_vectorizer = CountVectorizer(ngram_range=(1,2))
def gram_tf_feature(data):
    return tf_vectorizer.fit_transform(data)

def get_feature(data):
    return gram_tf_feature(data)


