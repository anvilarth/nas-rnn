import torch
import torch.nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F

import sys 
sys.path.append('./nas-bench-nlp-release')

from splitcross import SplitCrossEntropyLoss

from scores import compute_te_nas, compute_ze_nas, compute_norm_score
import glob

import numpy as np
import networkx as nx
import math
import json
import time
import tqdm

import data
import os
from utils import batchify
from argparse import Namespace
from my_model import AWDRNNModel
import datetime
from utils import get_batch, repackage_hidden

logs = sorted(glob.glob('./logs_folder/*.json'))
for log_name in tqdm.tqdm(logs[:200]):
    log = json.load(open(log_name, 'r'))
    args = Namespace(**log)
    if log['status'] != "OK":
        continue

    args.data = './data/ptb'

    corpus = data.Corpus(args.data)
    cuda = 'cuda:0'

    train_eval_data = batchify(corpus.train, args.eval_batch_size, args, cuda)
    ntokens = len(corpus.dictionary)

    model = AWDRNNModel(args.model, 
                                ntokens, 
                                args.emsize, 
                                args.nhid, 
                                args.nlayers, 
                                args.dropout, 
                                args.dropouth, 
                                args.dropouti, 
                                args.dropoute, 
                                args.wdrop, 
                                args.tied,
                                args.recepie,
                                verbose=False)

    model.to(cuda);
    gpu = 0
    criterion = SplitCrossEntropyLoss(args.emsize, splits=[], verbose=False)
    # prefetch = [get_batch(train_eval_data, 0, args, evaluation=True) for i in range(0, 100, args.bptt)]
    prefetch = [get_batch(train_eval_data, 0, args, evaluation=True) for i in range(0, train_eval_data.size(0) - 1, args.bptt)][:10]
    res_tenas, trace_metric = compute_te_nas(model, prefetch, criterion, args.eval_batch_size)
    res_zenas = compute_ze_nas(gpu, model, batch_size=args.eval_batch_size, repeat=1, mixup_gamma=0.1, batch_len=50, fp16=False)['avg_nas_score']
    res_gradnorm = compute_norm_score(gpu, model, criterion, batch_size=args.eval_batch_size, batch_len=50)

    results = {'zenas': res_zenas, 'gradnorm': res_gradnorm, 'tenas': res_tenas, 'trace': trace_metric}
    with open('logs/' + log_name.split('/')[-1], 'w') as f:
        json.dump(results, f)

