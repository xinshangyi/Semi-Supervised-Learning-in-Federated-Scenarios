#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import pickle as pkl

from dataset.randaugment import RandAugmentMC
from models.Fed import FedAvg, FedJS, get_JSweight, FedAvg_pro
from models.Nets import MLP, CNNCifar, CNNMnist
from models.SemisuperUpdate import SemisuperUpdate
from models.test import test_img
from models.Update import Update
from save.savefile import savexcl
from utils.options import args_parser
from utils.sampling import (cifar_iid, cifar_noniid, mnist_iid, mnist_irnoniid,
                            mnist_noniid)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_irnoniid(dataset_train, args.num_users)
    elif args.dataset in ('cifar10', 'cifar100'):
        # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),  # 上下左右各填充四个，即变为40*40
                                  padding_mode='reflect'),
            # RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])

        trans_cifar_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])
        if args.dataset == 'cifar10':
            dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
            # print(dataset_train[0])
            dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_val)
            args.classes = 10
        else: 
            dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar_train)
            # print(dataset_train[0])
            dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar_val)
            args.num_classes = 100

        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            if args.noniid_type == 1:
                # dict_users = cifar_noniid(dataset_train, args.num_users)
                dict_users_file = './save/{}_noniid_extreme.pkl'.format(args.dataset)
                with open(dict_users_file, 'rb') as f:
                    dict_users = pkl.load(f)
                print('extreme noniid')
            else:
                # load
                dict_users_file = './save/{}_noniid_dirichlet_beta{}.pkl'.format(args.dataset, args.beta)
                with open(dict_users_file, 'rb') as f:
                    dict_users = pkl.load(f)
                print('dirichlet noniid')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_test[0][0].shape

    print(args.num_classes)

    # build model
    if args.model == 'cnn' and args.dataset in ('cifar10', 'cifar100'):
        net_glob = CNNCifar(args.num_classes).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args.num_classes).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train, loss_test = [], []
    acc_train, acc_test = [], []
    commu = [] #2-dimensional
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    num_train = len(dataset_train) # 50000

    local_w_dict = {i: net_glob for i in range(args.num_users)}
    local_test_dict = {i: [] for i in range(args.num_users)}

    # JS = get_JSweight(dataset_train, dict_users)
    # for finetune
    all_idxs = [i for i in range(len(dataset_train))]
    global_idxs = np.random.choice(all_idxs, int(0.05 * len(dataset_train)), replace=False)
    cloud = Update(args=args, dataset=dataset_train, idxs=global_idxs, flag=1)
    # w_cloud, _ = cloud.train(net=copy.deepcopy(net_glob).to(args.device))
    # net_glob.load_state_dict(w_cloud)

    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        num_portion_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # JS_tmp = np.array(JS)[idxs_users]
        for idx in idxs_users:
            num_portion_locals.append(len(dict_users[idx]))  # 每一个client的数据量
            # local = Update(args=args, dataset=dataset_train, idxs=dict_users[idx], flag=0)
            # w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            # print('Round {:3d}. User_id: {}. Local_loss: {:.4f}'.format(iter + 1, idx, loss))
            
            # w, loss, loss_sv, loss_dis = local.gd_train(net=copy.deepcopy(net_glob).to(args.device))
            # print('Round {:3d}. User_id: {}. Local_loss: {:.4f}. loss_sv: {:.4f}. loss_dis: {:.4f}'.format(iter + 1, idx, loss, loss_sv, loss_dis))

            
            # w, loss = local.Uncertain_train(net=copy.deepcopy(net_glob).to(args.device),
            #                                 refernet=copy.deepcopy(local_w_dict[idx]).to(args.device))
            

            # # semi: LSS
            local = SemisuperUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss, loss_x, loss_u = local.train(net=copy.deepcopy(net_glob).to(args.device))
            print('Round {:3d}. User_id: {:3d}. Local_loss: {:.4f}. Local_lossx: {:.4f}. Local_lossu: {:.4f}'
                  .format(iter + 1, idx, loss, loss_x, loss_u))

            # # for local model performance observation
            refernet = copy.deepcopy(net_glob)
            refernet.load_state_dict(w)
            local_w_dict[idx] = refernet
            
            # # fc_norm
            # refernet = copy.deepcopy(net_glob)
            # refernet.load_state_dict(w)
            
            # for layer in refernet.modules():
            #     if isinstance(layer, nn.Linear):
            #         fc_weight = layer.weight.data
            #         neuron_num = layer.out_features
            #         if neuron_num == args.num_classes:
            #             norm_tensor = torch.norm(fc_weight, p=2, dim=1)
            #             layer.weight.data = fc_weight / norm_tensor.reshape((neuron_num, 1))        
            # w_locals.append(copy.deepcopy(refernet.state_dict()))
            
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        if args.global_aggregation:
            num_portion_locals.append(num_train - np.sum(num_portion_locals))
            w_locals.append(copy.deepcopy(net_glob.state_dict()))
        
        print(num_portion_locals)

        w_glob = FedAvg_pro(w_locals, num_portion_locals)

        # if args.global_aggregation:
        #     for _ in range(args.num_users - m):
        #         w_locals.append(copy.deepcopy(net_glob.state_dict()))
        
        # print(len(w_locals))

        # # update global weights
        # w_glob = FedAvg(w_locals)
        # # w_glob = FedJS(w_locals, JS_tmp)


        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)


        '''
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        '''

        # acc_bf, loss_bf = test_img(net_glob, dataset_test, args)
        # print('Round {:3d}. before_finetune_loss: {:.4f}. before_finetune_acc: {:.2f}.'
        #       .format(iter + 1, loss_bf, acc_bf.item()))

        # # fc_norm
        # for layer in net_glob.modules():
        #     if isinstance(layer, nn.Linear):
        #         fc_weight = layer.weight.data
        #         neuron_num = layer.out_features
        #         if neuron_num == args.num_classes:
        #             norm_tensor = torch.norm(fc_weight, p=2, dim=1)
        #             layer.weight.data = fc_weight / norm_tensor.reshape((neuron_num, 1))

        # finetune
        # w_cloud, _ = cloud.train(net=copy.deepcopy(net_glob).to(args.device))
        # fc_finetune
        # w_cloud, _ = cloud.fc_train_global(net=copy.deepcopy(net_glob).to(args.device))
        # net_glob.load_state_dict(w_cloud)

        # after every communication, test the global model
        if args.dataset == 'cifar10':
            dataset_train2 = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_val)
        else:
            dataset_train2 = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar_val)
        acc_traintmp, loss_traintmp = test_img(net_glob, dataset_train2, args)
        acc_testtmp, loss_testtmp = test_img(net_glob, dataset_test, args)
        acc_traintmp = acc_traintmp.item()
        acc_testtmp = acc_testtmp.item()
        commu_each = [loss_traintmp, loss_testtmp, acc_traintmp, acc_testtmp]
        # commu_each = [loss_traintmp, loss_testtmp, acc_traintmp, acc_testtmp, acc_bf.item()]
        print('Round {:3d}, Training loss {:.3f}'.format(iter+1, loss_traintmp))
        print('Round {:3d}, Testing loss {:.3f}'.format(iter+1, loss_testtmp))
        print("Training accuracy: {:.2f}".format(acc_traintmp))
        print("Testing accuracy: {:.2f}".format(acc_testtmp))
        loss_train.append(loss_traintmp)
        loss_test.append(loss_testtmp)
        acc_train.append(acc_traintmp)
        acc_test.append(acc_testtmp)
        commu.append(commu_each)

        # after every communication, every client test the global model
        for idx in range(args.num_users):
            acc_testtmp, loss_testtmp = test_img(local_w_dict[idx], dataset_test, args)
            local_test_dict[idx].append(acc_testtmp.item())
        result = np.array([local_test_dict[i] for i in range(args.num_users)])
        savepath = './save/semi_iid{}_GA{}.npy'.format(args.iid, args.global_aggregation)
        np.save(savepath, result)


    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # #plt.savefig('./log/fedAvg_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    # plt.savefig('./log/loss_{}_{}_{}_C{}_iid.png'.format(args.dataset, args.model, args.epochs, args.frac))
    #
    # # plot acc curve
    # plt.figure()
    # plt.plot(range(len(acc_test)), acc_test)
    # plt.ylabel('test_accuracy')
    # # plt.savefig('./log/fedAvg_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    # plt.savefig('./log/acc_{}_{}_{}_C{}_iid.png'.format(args.dataset, args.model, args.epochs, args.frac))

    # savefile
    if args.noniid_type == 1:
        savexcl(commu, './save/semi_{}_{}_{}_C{}_iid{}_GA{}.xlsx'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.global_aggregation))
    else:
        savexcl(commu, './save/semi_{}_{}_{}_C{}_beta{}_GA{}.xlsx'.format(args.dataset, args.model, args.epochs, args.frac, args.beta, args.global_aggregation))
    # savexcl(commu, './save4/finetune_{}_{}_{}_C{}_iid{}_fcnorm.xlsx'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    print('Best test accuracy: {:.2f}'.format(np.max(acc_test)))
