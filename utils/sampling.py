#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import pickle as pkl
import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards = 2*num_users
    num_imgs = 60000//num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    #idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    '''
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    '''
    idxs = labels.argsort()

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

# Uneven noniid
def mnist_irnoniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    if num_users == 10:
        frac = 0.2
    else:
        frac = 0.1

    num_items = int(len(dataset)/num_users)
    all_idxs = [i for i in range(len(dataset))]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    labels = np.array(dataset.targets)

    for i in range(int(frac*num_users)):
        dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))


    #labels_rest = labels[all_idxs] #不可以先取出，不然idx有变化
    num_shards = int(2*(1-frac)*num_users)
    num_imgs = len(all_idxs)//num_shards
    idx_shard = [i for i in range(num_shards)]
    idxs_all = labels.argsort()
    #idxs = idxs[all_idxs] # 这里all_idxs是作为下标引的，因此有误
    idxalltmp = list(idxs_all)
    '''
    idx_tmp = [c for c in idxalltmp if c in all_idxs] #这里利用的思想是根据第一个列表的顺序保留共同存在于两个列表的数据，是可行的，然而效率非常低（接近5min）
    idxs = np.array(idx_tmp)
    '''
    '''
    idx_tmp = filter(lambda x: x in all_idxs, idxalltmp)# 这个是上面思想一致，可行但效率也很低
    idxs = np.array(list(idx_tmp))
    '''

    idx_tmp = sorted(all_idxs, key = idxalltmp.index)
    idxs = np.array(idx_tmp)


    for i in range(int(frac*num_users), num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """
    y_train = dataset.targets
    if len(set(y_train)) == 10:
        k_portion = 2
        dict_users_file = '../save/cifar10_noniid_extreme.pkl'
    elif len(set(y_train)) == 100:
        k_portion = 20
        dict_users_file = '../save/cifar100_noniid_extreme.pkl'
    print(k_portion)
    num_train = len(dataset)
    num_shards = k_portion*num_users
    num_imgs = num_train//num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    #idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)

    idxs = labels.argsort()

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, k_portion, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    
    # dict_users_file = '../save/cifar100_noniid_extreme.pkl'
    with open(dict_users_file, 'wb') as f:
        pkl.dump(dict_users, f, pkl.HIGHEST_PROTOCOL)

    return dict_users


def cifar_noniid_dirichlet(dataset, num_users, beta=0.5):
    """
    Sample non-I.I.D client data from CIFAR dataset (dirichlet distribution)
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    y_train = np.array(dataset.targets)
    # print(y_train)
    num_train = len(dataset)

    min_size = 0
    min_require_size = 10
    # K = len(set(y_train)) # 标签类别个数
    
    if len(set(y_train)) == 10:
        K = 10
        dict_users_file = '../save/cifar10_noniid_dirichlet_beta{}.pkl'.format(beta)
    elif len(set(y_train)) == 100:
        K = 100
        dict_users_file = '../save/cifar100_noniid_dirichlet_beta{}.pkl'.format(beta)
    print(K)

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            proportions = np.array([p * (len(idx_j) < num_train / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            # if K == 2 and n_parties <= 10:
            #     if np.min(proportions) < 200:
            #         min_size = 0
            #         break
    
    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
    
    # dict_users_file = '../save/cifar100_noniid_dirichlet_beta{}.pkl'.format(beta)
    with open(dict_users_file, 'wb') as f:
        pkl.dump(dict_users, f, pkl.HIGHEST_PROTOCOL)

    return dict_users


if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
    dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=transform_train)
    num = 10
    d = cifar_noniid_dirichlet(dataset_train, num)
    length = [len(d[i]) for i in range(10)]
    
    from collections import Counter
    print(length)
    a = np.array(dataset_train.targets)
    print(Counter(a[d[0]]))
   
