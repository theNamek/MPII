# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import os
# from data_utils.vocab import Vocab
# import pickle
from .my_logger import set_logging_process

#
# def try_load_vocab(vocab_file, vocab_dump_file, vocab_tokens=None, max_vocab_size=26000):
#     if not os.path.exists(vocab_dump_file):
#         vocab = Vocab(vocab_file, vocab_tokens=vocab_tokens, max_vocab_size=max_vocab_size)
#         with open(vocab_dump_file, 'wb') as vf:
#             pickle.dump(vocab, vf, protocol=2)
#         return vocab
#
#     with open(vocab_dump_file, 'rb') as vf:
#         vocab = pickle.load(vf)
#     return vocab


def make_env(opt, no_logger=False):
    workdir = opt.workdir

    if not os.path.exists(workdir):
        os.makedirs(workdir)
    model_dir = os.path.join(workdir, 'saved')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not no_logger:
        log_path = os.path.join(workdir, 'log_train.txt')
        set_logging_process(log_path)
    pass

