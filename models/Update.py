#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from utils.misc import AverageMeter
import numpy as np
import torch.nn.functional as F
from torchvision import datasets
import random
from sklearn import metrics
import copy


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        # self.image = np.array(self.data)[idxs]
        self.targets = np.array(dataset.targets)[self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Update(object):
    # flag为1代表是GlobalUpdate(公用数据集初训以及后续finetune), flag为0代表是LocalUpdate
    def __init__(self, args, dataset=None, idxs=None, flag=1):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        # self.selected_clients = []
        if flag:
            self.bs = 64
            self.epochs = 5
        else:
            self.bs = self.args.local_bs
            self.epochs = self.args.local_ep
        base_dataset = DatasetSplit(dataset, idxs)
        self.ldr_train = DataLoader(base_dataset, batch_size=self.bs, shuffle=True)

        # # 对照实验
        # labeled_idxs = []
        # for i in range(10):
        #     idx = np.where(base_dataset.targets == i)[0]  # 返回一个array: 在数据集中标签为i的数据位置
        #     # print(len(idx))
        #     np.random.shuffle(idx)
        #     labeled_idxs.extend(idx[:50])
        # ablation_dataset = DatasetSplit(dataset, labeled_idxs)
        # # print(len(ablation_dataset))
        # self.ldr_train = DataLoader(ablation_dataset, batch_size=self.bs, shuffle=True)



    def train(self, net):

        losses = AverageMeter()
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

        for iter in range(self.epochs):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                losses.update(loss.item())

                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))

        return net.state_dict(), losses.avg

    def gd_train(self, net):

        base_model = copy.deepcopy(net).to(self.args.device)
        losses = AverageMeter()
        losses_dis = AverageMeter()
        losses_sv = AverageMeter()

        net.train()
        base_model.eval()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

        for iter in range(self.epochs):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()

                logits = net(images)
                sv_loss = self.loss_func(logits, labels)

                dis_logits = base_model(images)
                dis_logits_pro = dis_logits.true_divide(self.args.tempera)
                logits_pro = logits.true_divide(self.args.tempera)
                logsoftmax = nn.LogSoftmax(dim=1)
                lsm_logits = logsoftmax(logits_pro)

                dis_target = F.softmax(dis_logits_pro, dim=1)
                dis_loss = torch.sum(-lsm_logits * dis_target, dim=1)
                dis_loss = self.args.tempera**2 * dis_loss

                loss = dis_loss.mean() + sv_loss
                loss.backward()

                losses.update(loss.item())
                losses_dis.update(dis_loss.mean().item())
                losses_sv.update(sv_loss.item())

                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tLoss_dis: {:.6f} \tLoss_sv: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item(), loss_dis.mean().item(), loss_sv.item()))

        return net.state_dict(), losses.avg, losses_sv.avg, losses_dis.avg

    def fc_train_global(self, net):

        losses = AverageMeter()
        for para in net.features.parameters():
            para.requires_grad = False

        net.train()
        # train and update
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

        for iter in range(self.epochs):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                losses.update(loss.item())

                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))

        return net.state_dict(), losses.avg

    def Uncertain_train(self, net, refernet):

        losses = AverageMeter()
        net.train()
        refernet.eval()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

        for iter in range(self.epochs):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                log_probs_refer = refernet(images)

                logp_logits = F.log_softmax(log_probs, dim=-1)
                p_logits = F.softmax(log_probs_refer, dim=-1)
                var = F.kl_div(logp_logits, p_logits)

                loss_init = self.loss_func(log_probs, labels)
                loss = loss_init * torch.exp(-var) + var

                loss.backward()

                losses.update(loss.item())

                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))

        return net.state_dict(), losses.avg
