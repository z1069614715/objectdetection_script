import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
import math
import os
import random
import time
import sys
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from utils.torch_utils import select_device
from models.common import DetectMultiBackend

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='trained weights path')
    parser.add_argument('--batch', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--imgs', nargs='+', type=int, default=[640, 640], help='[height, width] image sizes')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--warmup', default=200, type=int, help='warmup time')
    parser.add_argument('--testtime', default=1000, type=int, help='test time')
    parser.add_argument('--half', action='store_true', default=False, help='fp16 mode.')
    opt = parser.parse_args()
    
    device = select_device(opt.device, batch_size=opt.batch)
    
    # Model
    weights = opt.weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        model = DetectMultiBackend(weights, device=device)
        print(f'Loaded {weights}')  # report
    else:
        assert weights.endswith('.pt'), "compress need weights."
    
    example_inputs = torch.randn((opt.batch, 3, *opt.imgs)).to(device)
    
    if opt.half:
        model = model.half()
        example_inputs = example_inputs.half()
    
    print('begin warmup...')
    for i in tqdm(range(opt.warmup), desc='warmup....'):
        model(example_inputs)
    
    print('begin test latency...')
    time_arr = []
    
    for i in tqdm(range(opt.testtime), desc='test latency....'):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        model(example_inputs)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        time_arr.append(end_time - start_time)
    
    std_time = np.std(time_arr)
    infer_time_per_image = np.sum(time_arr) / (opt.testtime * opt.batch)
    
    print(f'model weights:{opt.weights} size:{get_weight_size(opt.weights)}M (bs:{opt.batch})Latency:{infer_time_per_image:.5f}s +- {std_time:.5f}s fps:{1 / infer_time_per_image:.1f}')