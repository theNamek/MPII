# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from fairseq.models.bart import BARTModel
from fairseq.data.data_utils import collate_tokens
import torch
from onmt.utils.info_logging import InfoLogger
import json
from torch.optim import Adam
from torch.nn import NLLLoss
from onmt.utils.env_utils import make_env
import sys
import logging
import os
import cls_bart_opts
import argparse

train_info_logger = InfoLogger('loss', 'acc')
valid_info_logger = InfoLogger('loss', 'acc')
info_template = '\nStep: {:7}, LR: {}\n{}\n'
step = 0
best_eval_acc = 0.0


def get_batch_data(step, datas, batch_size, bart):
    n_data = len(datas)
    s_idx = step * batch_size
    cur_batch_ids = [(s_idx + i) % n_data for i in range(batch_size)]
    items = [datas[i] for i in cur_batch_ids]
    contexts = []
    for item in items:
        context = item['context']
        q, cs = context.split(' the choices are ')
        css = cs.split(' , ')
        left_cs = css[-1].split(' or ')
        new_ctx = (q, css[0], css[1], css[2], left_cs[0], left_cs[1])
        contexts.append(new_ctx)

    batch = collate_tokens(
        [bart.encode(*ctx_item) for ctx_item in contexts], pad_idx=1
    )
    target = [item['label'] for item in items]
    target = torch.LongTensor(target).cuda()

    return batch, target


def load_data(data_path):
    datas = []
    with open(data_path, encoding='utf-8') as sf:
        for line in sf:
            item = json.loads(line)
            datas.append(item)
        pass
    return datas


def _cal_acc(gt, logits):
    pred = torch.argmax(logits, dim=1)
    correct = torch.eq(gt, pred)
    acc = torch.mean(correct.float())
    return acc.item()


def train():
    global step
    bart.train()

    for idx in range(n_steps_per_epoch):
        step += 1

        batch_data, target = get_batch_data(idx, train_datas, batch_size, bart)
        logprobs = bart.predict('cqa_cls', batch_data)
        loss = loss_fn(logprobs, target)
        loss_val = loss.item()
        acc = _cal_acc(target, logprobs)
        train_info_logger.update(loss_val, acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if 0 == step % 100:
            # logging.info(info_template.format(
            #     step, 0.001, train_info_logger.get_metric()
            # ))
            logging.info(info_template.format(
                step, lr, train_info_logger.get_metric()
            ))
            train_info_logger.clear()
    pass


def valid():
    global best_eval_acc
    valid_info_logger.clear()
    bart.eval()

    logging.info('-' * 80)
    for idx in range(n_steps_per_valid):
        batch_data, target = get_batch_data(idx, valid_datas, batch_size, bart)
        logprobs = bart.predict('cqa_cls', batch_data)
        loss = loss_fn(logprobs, target)
        loss_val = loss.item()
        acc = _cal_acc(target, logprobs)
        valid_info_logger.update(loss_val, acc)
    logging.info('Eval results: {}'.format(valid_info_logger.get_metric()))
    acc = valid_info_logger.acc / n_steps_per_valid
    flag = False
    if acc > best_eval_acc:
        best_eval_acc = acc
        logging.info('Finding Best')
        flag = True
    logging.info('-' * 80)
    return flag


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='bart cls::train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cls_bart_opts.add_md_help_argument(parser)
    cls_bart_opts.cls_bart(parser)
    opt = parser.parse_args()

    bart_dir = opt.bart_dir
    batch_size = opt.batch_size
    workdir = opt.workdir
    data_dir = opt.data_dir
    lr = opt.lr
    train_from = opt.train_from
    make_env(opt)

    bart = BARTModel.from_pretrained(bart_dir, checkpoint_file='model.pt')
    bart.register_classification_head('cqa_cls', num_classes=5)
    if train_from is not None:
        logging.info('+' * 80)
        logging.info('Loading model from: {}'.format(train_from))
        state_dict = torch.load(train_from)
        bart.load_state_dict(state_dict)
        logging.info('+' * 80)
    bart = bart.cuda()
    optimizer = Adam(params=bart.parameters(), lr=lr)
    loss_fn = NLLLoss()

    train_data_path = os.path.join(data_dir, 'train.jsonl')
    train_datas = load_data(train_data_path)
    n_steps_per_epoch = len(train_datas) // batch_size
    valid_data_path = os.path.join(data_dir, 'dev.jsonl')
    valid_datas = load_data(valid_data_path)
    n_steps_per_valid = len(valid_datas) // batch_size

    n_epochs = 20
    for epid in range(n_epochs):
        logging.info('Epoch: {}'.format(epid))
        train()
        torch.save(bart.state_dict(), os.path.join(opt.workdir, 'md_stateDict_last.pt'))
        find_best = valid()
        if find_best:
            logging.info('Saving model ... ')
            torch.save(bart.state_dict(), os.path.join(opt.workdir, 'md_stateDict_best.pt'))
    pass



