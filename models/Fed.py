#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
#from torch import nn
import scipy.stats
import numpy as np


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

# portion 对应每一个client的数据量
def FedAvg_pro(w, portion):
    portion = portion / np.sum(portion)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        #print(k)
        w_avg[k] = w_avg[k].to(torch.float32)
        w[0][k] = w[0][k].to(torch.float32)
        for i in range(1, len(w)):
            w[i][k] = w[i][k].to(torch.float32) #batchnorm2d层得到的tensor类型为int64，不能直接计算
            #print((torch.mul(w[i][k], weight[i])).dtype)
            #print(w_avg[k].dtype)
            w_avg[k] += torch.mul(w[i][k], portion[i])
        w_avg[k] = w_avg[k] - torch.mul(w[0][k], (1-portion[0]))
    return w_avg



def FedJS(w, js):
    # smaller JS_divergence, bigger weight
    weight = 1 - js/np.sum(js)
    weight = weight/np.sum(weight)
    # dic type is changable, direct assignment will change the original dict
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        #print(k)
        w_avg[k] = w_avg[k].to(torch.float32)
        w[0][k] = w[0][k].to(torch.float32)
        for i in range(1, len(w)):
            w[i][k] = w[i][k].to(torch.float32) #batchnorm2d层得到的tensor类型为int64，不能直接计算
            #print((torch.mul(w[i][k], weight[i])).dtype)
            #print(w_avg[k].dtype)
            w_avg[k] += torch.mul(w[i][k], weight[i])
        w_avg[k] = w_avg[k] - torch.mul(w[0][k], (1-weight[0]))
    return w_avg



def get_JSweight(dataset, dict_users):
    
    labels = dataset.targets.numpy()
    # 总的labels的概率
    p = []
    k_labels = set(np.sort(labels))
    for i in k_labels:
        p.append(np.sum(labels==i)/len(labels))
    p = np.array(p)
    
    q = [] # q store all users JS divergence with p
    for i in dict_users.keys():
        labeltmp = labels[dict_users[i]]
        k_labeltmp = set(np.sort(labeltmp))
        q_each = np.zeros(10)
        for j in k_labeltmp:
            q_each[j] = np.sum(labeltmp==j)/len(labeltmp)
        #q.append(q_each)
        q.append(JS_divergence(p, q_each))
    
    #weights = [1-i/sum(q) for i in q]
    
    return q
    


def JS_divergence(p, q):
    M = (p+q)/2
    return 0.5*scipy.stats.entropy(p, M) + 0.5*scipy.stats.entropy(q, M)

'''
import numpy as np
import scipy.stats

x = np.array([1, 1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 3, 4, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1])
y = np.array([2, 2, 2, 2, 3, 4, 4, 2, 2, 1, 2, 4, 3, 2, 2, 1, 3, 4, 4, 2, 4, 3])
#x=np.array([0.65,0.25,0.07,0.03])
#y=np.array([0.6,0.25,0.1,0.05])
k_x = set(x)
p = []
for i in k_x:
    p.append(np.sum(x==i)/len(x))
    #p.append(x.count(i) / len(x))
p = np.array(p)

k_y = set(y)
q = []
for i in k_y:
    #q.append(y.count(i) / len(y))
    q.append(np.sum(y==i)/len(y))
q = np.array(q)

def KL_divergence(p, q):
    return scipy.stats.entropy(p, q)

print(KL_divergence(q,p))


def JS_divergence(p, q):
    M = (p+q)/2
    return 0.5*scipy.stats.entropy(p, M) + 0.5*scipy.stats.entropy(q, M)
print(JS_divergence(q,p))
'''


