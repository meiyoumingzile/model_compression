from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import platform
from functools import cmp_to_key

import torchvision
from torch.utils.data import DataLoader

from torchvision.transforms import transforms
from nasbench import api
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import networkx as nx
import os
from torch_geometric.nn import GATConv
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cudaID', type=int, default=0)
args = parser.parse_args()
print(platform.system(),args.cudaID)


NASBENCH_TFRECORD = '/home/data/hw/Z_bing333/project/RobNets/nasbench-master/nasbench_full.tfrecord'
INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT]
def randMat():
    dotList = []
    for i in range(0, 7):
        for j in range(i + 1, 7):
            dotList.append((i, j))
    matrix=np.zeros((7,7),dtype=np.int32)
    n=len(dotList)
    i=0
    cnt=0
    while(i<6):
        j=random.randint(i+1,6)
        matrix[i][j] = 1
        i=j
        cnt+=1
    li = random.sample(dotList, 30)
    for a in li:
        if matrix[a[0]][a[1]]==0:
            matrix[a[0]][a[1]]=1
            cnt+=1
            if cnt==9:
                break
    # if nx.is_weakly_connected(tonxGraph(matrix)):
    #     return matrix
    return matrix
def getQuerryAns(mat,nasbench):
    model_spec = api.ModelSpec(matrix=mat,ops=ops)
    # print('Querying an Inception-like model.')
    data = nasbench.query(model_spec)
    # print(data)#结果的邻接矩阵可能与输入不相同，那是因为节点顺序变了，网络结构不变
    # ans=nasbench.get_budget_counters()
    return data

nasbench = api.NASBench(NASBENCH_TFRECORD)
mat=randMat()

for i in range(0, 7):
    for j in range(i + 1, 7):
        if mat[i][j]==1:
            pass
data=getQuerryAns(mat,nasbench)

print(mat,data)