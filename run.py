#!/usr/bin/env python
# coding: utf-8

import models
import utils

import argparse
import torch
import logging
import torch.nn as nn
from torch import optim
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn


# In[6]:


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help="location of the data corpus")
parser.add_argument('--batchsz', type=int, default=64, help="batch size")

parser.add_argument('--seed',type=str, default=2, help='set random seed')
parser.add_argument('--unrolled', action = 'store_true', default=False, help='use one-step unrolled validation loss')

args = parser.parse_args()
print(args.data, args.batchsz)


# In[7]:


def main():
    np.random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled= True
    torch.manual_seed(args.seed)
    
    total, uset = os.opoen('nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
                        ).read().split('\n')[args.gpu].split(',')
    total = int(total)
    used = int(used)
    
    print('GPU:',total,' used:',used)
    
    args.unrolled = True
    
    logging.info("GPU device = %d"%args.gpu)
    logging.info("args = %s"%args)
    
    criterion = nn.CrossEntropyLoss().to(device)
    model = Network(args.init_ch, 10, args.layers, creterion).to(device)


# In[ ]:




