# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import sys
import json


if __name__ == '__main__':
    gold_path = sys.argv[1]
    pred_path = sys.argv[2]

    pred_labels = []
    with open(pred_path, encoding='utf-8') as pf:
        for line in pf:
            item = json.loads(line)
            pl = item['pred_label']
            pred_labels.append(pl)
    gold_labels = []
    with open(gold_path, encoding='utf-8') as gf:
        for line in gf:
            item = json.loads(line)
            gl = item['gold_label_id']
            gold_labels.append(gl)
    n_correct = 0
    total = 0
    for pl, gl in zip(pred_labels, gold_labels):
        total += 1
        if pl == gl:
            n_correct += 1
    print('ACC: {}'.format(n_correct/total))


