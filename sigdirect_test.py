#!/usr/bin/env python3

import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score

import sigdirect

class _Preprocess:

    def __init__(self):
        self._label_encoder = None

    def preprocess_data(self, raw_data):
        """ Given one of UCI files specific to SigDirect paper data,
            transform it to the common form used in sklearn datasets"""
        transaction_data =  [list(map(int, x.strip().split())) for x in raw_data]
        max_val = max([max(x[:-1]) for x in transaction_data])
        X,y = [], []

        for transaction in transaction_data:
            positions = np.array(transaction[:-1]) - 1
            transaction_np = np.zeros((max_val))
            transaction_np[positions] = 1
            X.append(transaction_np)
            y.append(transaction[-1])
        X = np.array(X)
        y = np.array(y)

        # converting labels
        if self._label_encoder is None:# train time
            unique_classes = np.unique(y)
            self._label_encoder = defaultdict(lambda: 0, zip(unique_classes, range(len(unique_classes))))
            y = np.vectorize(self._label_encoder.get)(y)
        else:# test time
            y = np.vectorize(lambda a: self._label_encoder[a])(y)

        return X,y

def test_uci():

    assert len(sys.argv)>1
    dataset_name = sys.argv[1]
    print(dataset_name)

    if len(sys.argv)>2:
        start_index = int(sys.argv[2])
    else:
        start_index = 1
    
    final_index = 10
    k = final_index - start_index + 1

    all_pred_y = defaultdict(list)
    all_true_y = []

    # counting number of rules before and after pruning
    generated_counter = 0
    final_counter     = 0
    avg = [0.0] * 4

    tt1 = time.time()

    for index in range(start_index, final_index +1):
        print(index)
        prep = _Preprocess()

        # load the training data and pre-process it
        train_filename = os.path.join('uci', '{}_tr{}.txt'.format(dataset_name, index))
        with open(train_filename) as f:
            raw_data = f.read().strip().split('\n')
        X,y = prep.preprocess_data(raw_data)
        clf = sigdirect.SigDirect(get_logs=sys.stdout)
        generated_c, final_c = clf.fit(X, y)
        
        generated_counter += generated_c
        final_counter     += final_c

        # load the test data and pre-process it.
        test_filename  = os.path.join('uci', '{}_ts{}.txt'.format(dataset_name, index))
        with open(test_filename) as f:
            raw_data = f.read().strip().split('\n')
        X,y = prep.preprocess_data(raw_data)

        # evaluate the classifier using different heuristics for pruning
        for hrs in (1,2,3):
            y_pred = clf.predict(X, hrs)
            print('ACC S{}:'.format(hrs), accuracy_score(y, y_pred))
            avg[hrs] += accuracy_score(y, y_pred)

            all_pred_y[hrs].extend(y_pred)

        all_true_y.extend(list(y))
        print('\n\n')

    print(dataset_name)
    for hrs in (1,2,3):
        print('AVG ACC S{}:'.format(hrs), accuracy_score(all_true_y, all_pred_y[hrs]))
    print('INITIAL RULES: {} ---- FINAL RULES: {}'.format(generated_counter/k, final_counter/k))
    print('TOTAL TIME:', time.time()-tt1)
    

if __name__ == '__main__':
    test_uci()