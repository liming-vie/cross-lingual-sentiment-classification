#!/usr/bin/env python
# encoding: utf-8

import os
from tqdm import tqdm
from pyltp import Segmentor
import xml.etree.ElementTree as ET

ltp_segmentor = Segmentor()
ltp_segmentor.load('../ltp_data/cws.model')
def segment(corpus):
    ret = []
    for sent in corpus:
        tokens = ltp_segmentor.segment(sent)
        ret.append(' '.join(tokens))
    return ret

def load_data(data_dir, fname):
    fpath = os.path.join(data_dir, fname)
    print 'Loading data from %s'%fpath
    tree = ET.ElementTree(file=fpath)
    root = tree.getroot()
    dic={}
    for child in root[0]:
        dic[child.tag]=[]
    for item in tqdm(root):
        for child in item:
            text = child.text.encode('utf-8') if child.text else ""
            dic[child.tag].append(text)
    return dic

def get_dict_corpus(data_dict, k1, k2):
    t1 = data_dict[k1]
    t2 = data_dict[k2]
    return ['%s %s'%(a,b) for a, b in zip(t1, t2)]

def prepare_data(train_dir, train_en, train_cn, unlabel_cn,
        test_dir, test_file, test_label):
    ten = load_data(train_dir, train_en)
    tcn = load_data(train_dir, train_cn)
    ucn = load_data(train_dir, unlabel_cn)
    test = load_data(test_dir, test_file)
    test_label = load_data('', test_label)

    train_en = get_dict_corpus(ten, 'summary', 'text') + \
            get_dict_corpus(tcn, 'tr_summary', 'tr_text')
    train_cn = get_dict_corpus(ten, 'tr_summary', 'tr_text') + \
            get_dict_corpus(tcn, 'summary', 'text')

    unlabel_en = get_dict_corpus(ucn, 'tr_summary', 'tr_text')
    unlabel_cn = get_dict_corpus(ucn, 'summary', 'text')

    test_cn = get_dict_corpus(test, 'summary', 'text')
    test_en = get_dict_corpus(test, 'tr_summary', 'tr_text')

    mp = {'P':1, 'N':0}
    train_label = ten['polarity'] + tcn['polarity']
    train_label = map(lambda x:mp[x], train_label)
    test_label = test_label['polarity']
    test_label = map(lambda x:mp[x], test_label)

    train_cn = segment(train_cn)
    unlabel_cn = segment(unlabel_cn)
    test_cn = segment(test_cn)

    test_id = test['review_id']

    return train_en, train_cn, train_label, unlabel_en, unlabel_cn, test_cn, test_en, test_label, test_id
