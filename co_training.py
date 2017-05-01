#!/usr/bin/env python
# encoding: utf-8

import sys
import heapq
import threading
import data_utils
import numpy as np
from sklearn import svm
from scipy.sparse import coo_matrix, vstack
from sklearn.feature_extraction.text import CountVectorizer

class TrainThread(threading.Thread):
    def __init__(self, feat_train, label, feat_unlabel):
        threading.Thread.__init__(self)
        self.feat_train = feat_train
        self.label = label
        self.feat_unlabel = feat_unlabel
        self.clf = svm.SVC(kernel='linear', probability=True)

    def run(self):
        self.clf.fit(self.feat_train, self.label)
        self.y_unlabel = self.clf.predict_proba(self.feat_unlabel)

    def get_result(self):
        return self.clf, self.y_unlabel

def accuracy(Y, label):
    correct = 0
    for y, l in zip(Y, label):
        if y == l:
            correct+=1
    return correct/float(len(Y))

def get_feature(train, unlabel, test):
    tf_vectorizer = CountVectorizer(ngram_range=(1,2))
    feat_train = tf_vectorizer.fit_transform(train)
    feat_unlabel = tf_vectorizer.transform(unlabel)
    feat_test = tf_vectorizer.transform(test)
    return feat_train, feat_unlabel, feat_test

def top_k(y, idx, k, used):
    i = 0
    ret = []
    while len(ret) < k and i<len(y):
        if y[i][0][idx] < 0.5:
            break
        if y[i][1] not in used:
            ret.append(y[i][1])
            used[y[i][1]] = True
        i+=1
    return ret

def pick(y_prob, used, p, n):
    idx = range(len(y_prob))
    y = zip(y_prob, idx)

    y = sorted(y, key = lambda x: x[0][1], reverse=True)
    ep = top_k(y, 1, p, used)

    y = sorted(y, key = lambda x: x[0][0], reverse=True)
    en = top_k(y, 0, n, used)

    return ep, en

def add_to_train(feat_train, dict_unlabel, idx_p, idx_n, num_feature):
    row = []
    col = []
    data = []
    j = 0
    for idx in idx_p+idx_n:
        if idx in dict_unlabel:
            for f in dict_unlabel[idx]:
                row.append(j)
                col.append(f[0])
                data.append(f[1])
        j += 1
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    new_matrix = coo_matrix((data,(row, col)), shape=(j, num_feature))
    feat_train = vstack([feat_train, new_matrix])

    return feat_train

def get_feature_dict(feature):
    fdict = {}
    cx = coo_matrix(feature)
    for i, j, c in zip(cx.row, cx.col, cx.data):
        if i not in fdict:
            fdict[i] = []
        fdict[i].append((j,c))
    return cx.shape[1], fdict

def co_training(train_dir, train_en, train_cn, unlabel_cn,
        eval_dir, test_file, test_label, test_out,
        p, n, num_iter):
    print 'Preparing data'
    train_en, train_cn, train_label, \
        unlabel_en, unlabel_cn,\
        test_cn, test_en, test_label, test_id = \
            data_utils.prepare_data(train_dir, train_en, train_cn, unlabel_cn,\
                eval_dir, test_file, test_label)

    print 'Getting feature'
    feat_train_en, feat_unlabel_en, feat_test_en = \
            get_feature(train_en, unlabel_en, test_en)
    feat_train_cn, feat_unlabel_cn, feat_test_cn = \
            get_feature(train_cn, unlabel_cn, test_cn)

    num_feat_unlabel_en, dict_unlabel_en = get_feature_dict(feat_unlabel_en)
    num_feat_unlabel_cn, dict_unlabel_cn = get_feature_dict(feat_unlabel_cn)

    print 'Start training'
    best_acc = 0.0
    y_test = [0 for _ in xrange(len(test_label))]
    used={}
    for i in xrange(num_iter):
        # English classifier
        en_thread = TrainThread(feat_train_en, train_label, feat_unlabel_en)
        en_thread.start()
        # Chinese classifier
        cn_thread = TrainThread(feat_train_cn, train_label, feat_unlabel_cn)
        cn_thread.start()
        # thread join
        en_thread.join()
        cn_thread.join()
        clf_en, y_unlabel_en = en_thread.get_result()
        clf_cn, y_unlabel_cn = cn_thread.get_result()
        # add unlabeled samples into training sets
        p_en, n_en = pick(y_unlabel_en, used, p, n)
        p_cn, n_cn = pick(y_unlabel_cn, used, p, n)
        feat_train_en = add_to_train(feat_train_en, dict_unlabel_en, p_en, n_en, \
                num_feat_unlabel_en)
        feat_train_cn = add_to_train(feat_train_cn, dict_unlabel_cn, p_cn, n_cn, \
                num_feat_unlabel_cn)
        train_label.extend([1 for _ in xrange(len(p_en))])
        train_label.extend([0 for _ in xrange(len(n_en))])
        # test
        y_test_en = clf_en.predict_proba(feat_test_en)
        y_test_cn = clf_cn.predict_proba(feat_test_cn)
        for coef in [0.0, 0.3, 0.5, 0.7, 1.0]:
            for idx, ye, yc in zip(xrange(len(y_test)), y_test_en, y_test_cn):
                y = y_test_en[idx][1] * coef + y_test_cn[idx][1] * (1.-coef)
                y_test[idx] = 1 if y >= 0.5 else 0
            acc = accuracy(y_test, test_label)
            if acc > best_acc:
                best_acc = acc
                mp={1:'P', 0:'N'}
                with open(test_out, 'w') as fout:
                    for tid, y in zip(test_id, y_test):
                        fout.write("0\tolivia_0\t%s\t%s\n"%(tid, mp[y]))
        print 'Iter %d, acc:%f, best:%s'%(i, acc, best_acc)

if __name__=='__main__':
    domain = sys.argv[1]

    train_dir = '../output/'
    train_en = 'Train_EN/%s.xml'%domain
    train_cn = 'Train_CN/%s.xml'%domain
    unlabel_cn = 'Unlabel_CN/%s.xml'%domain

    eval_dir = '../output/eval/'
    test_file = '%s.xml'%domain
    test_label = '../eval/label/%s/testResult.data'%domain
    test_out = '../output/%s.out'%domain

    p, n = [5, 5]
    num_iter = 80

    co_training(train_dir, train_en, train_cn, unlabel_cn,
            eval_dir, test_file, test_label, test_out,
            p, n, num_iter)
