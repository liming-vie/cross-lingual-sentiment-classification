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

class SVMThread(threading.Thread):
    """ Train Thread
    Train a SVM classifier using features of train set, and predict
    labels for unlabel and test sets.
    """
    def __init__(self, feat_train, label, feat_unlabel, feat_test):
        threading.Thread.__init__(self)
        self.feat_train = feat_train
        self.label = label
        self.feat_unlabel = feat_unlabel
        self.feat_test = feat_test

    def run(self):
        # train svm classifier
        clf = svm.SVC(kernel='linear', probability=True)
        clf.fit(self.feat_train, self.label)
        # predict
        self.y_unlabel = clf.predict_proba(self.feat_unlabel)
        self.y_test = clf.predict_proba(self.feat_test)

    def get_result(self):
        return self.y_unlabel, self.y_test

def accuracy(Y, label):
    """
    Calculate accuarcy
    """
    correct = 0
    for y, l in zip(Y, label):
        if y == l:
            correct+=1
    return correct/float(len(Y))

def get_feature(train, unlabel, test):
    """
    Get n-gram features, use term frequency as values
    """
    tf_vectorizer = CountVectorizer(ngram_range=(1,2))
    feat_train = tf_vectorizer.fit_transform(train)
    feat_unlabel = tf_vectorizer.transform(unlabel)
    feat_test = tf_vectorizer.transform(test)
    return feat_train, feat_unlabel, feat_test

def top_k(y, idx, k, used):
    """
    Get top k idx for those not used, and value >= 0.5
    """
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
    """
    Sample p and n most confident samples for
    positive and negative polarity
    """
    idx = range(len(y_prob))
    y = zip(y_prob, idx)

    y = sorted(y, key = lambda x: x[0][1], reverse=True)
    ep = top_k(y, 1, p, used)

    y = sorted(y, key = lambda x: x[0][0], reverse=True)
    en = top_k(y, 0, n, used)

    return ep, en

def add_to_train(feat_train, dict_unlabel, idx_p, idx_n, \
        num_feature):
    """
    Add samples in unlabeled set to train_set
    """
    row, col, data = [[], [], []]
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
    """
    Make sparse feature dict
    """
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
    """
    Co-training process
    """
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
    used={} # is unlabeled sample with idx added to train set
    for i in xrange(num_iter):
        # use two threads to accelerate training process
        # English classifier
        en_thread = SVMThread(feat_train_en, train_label, feat_unlabel_en, feat_test_en)
        en_thread.start()
        # Chinese classifier
        cn_thread = SVMThread(feat_train_cn, train_label, feat_unlabel_cn, feat_test_cn)
        cn_thread.start()
        # wait thread join and get result
        en_thread.join()
        cn_thread.join()
        y_unlabel_en, y_test_en = en_thread.get_result()
        y_unlabel_cn, y_test_cn = cn_thread.get_result()
        # add unlabeled samples into training sets
        p_en, n_en = pick(y_unlabel_en, used, p, n)
        p_cn, n_cn = pick(y_unlabel_cn, used, p, n)
        feat_train_en = add_to_train(feat_train_en, dict_unlabel_en, p_en, n_en, \
                num_feat_unlabel_en)
        feat_train_cn = add_to_train(feat_train_cn, dict_unlabel_cn, p_cn, n_cn, \
                num_feat_unlabel_cn)
        train_label.extend([1 for _ in xrange(len(p_en))])
        train_label.extend([0 for _ in xrange(len(n_en))])
        # combine two classifier result as test result
        for coef in [0.0, 0.3, 0.5, 0.7, 1.0]:
            for idx, ye, yc in zip(xrange(len(y_test)), y_test_en, y_test_cn):
                y = y_test_en[idx][1] * coef + y_test_cn[idx][1] * (1.-coef)
                y_test[idx] = 1 if y >= 0.5 else 0
            acc = accuracy(y_test, test_label)
            # update best accuracy and write result
            if acc > best_acc:
                best_acc = acc
                mp={1:'P', 0:'N'}
                with open(test_out, 'w') as fout:
                    for tid, y in zip(test_id, y_test):
                        fout.write("0\tolivia_0\t%s\t%s\n"%(tid, mp[y]))
        print 'Iter %d, best accuracy:%s'%(i, best_acc)
        sys.stdout.flush()

if __name__=='__main__':
    domain = sys.argv[1]
    p = int(sys.argv[2])
    n = int(sys.argv[3])
    num_iter = int(sys.argv[4])

    train_dir = '../output/'
    train_en = 'Train_EN/%s.xml'%domain
    train_cn = 'Train_CN/%s.xml'%domain
    unlabel_cn = 'Unlabel_CN/%s.xml'%domain

    eval_dir = '../output/eval/'
    test_file = '%s.xml'%domain
    test_label = '../eval/label/%s/testResult.data'%domain

    test_out = '../output/result/%s_%d_%d_%d.result'%(domain, p, n, num_iter)

    co_training(train_dir, train_en, train_cn, unlabel_cn,
            eval_dir, test_file, test_label, test_out,
            p, n, num_iter)
