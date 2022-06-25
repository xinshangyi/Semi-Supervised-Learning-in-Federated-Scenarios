#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from utils.misc import AverageMeter
import numpy as np
from PIL import Image
import random
from sklearn import metrics
from .Update import DatasetSplit
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F

from dataset.randaugment import RandAugmentMC

from tqdm import tqdm

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


class SemisuperUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        # self.loss_func = nn.CrossEntropyLoss()
        # self.selected_clients = []
        base_dataset = DatasetSplit(dataset, idxs)
        num_labeled = int(args.labeled_ratio * len(base_dataset))
        labeled_dataset, unlabeled_dataset = get_cifar('./data/cifar', base_dataset, num_labeled, args.k_img, args.k_img * args.mu)
        self.labeledtrain = DataLoader(labeled_dataset, batch_size=args.local_bs, shuffle=True)
        self.unlabeledtrain = DataLoader(unlabeled_dataset, batch_size=args.local_bs * args.mu, shuffle=True)

        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)

    def train(self, net):

        iteration = self.args.k_img // self.args.local_bs

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

        for local_epoch in range(self.args.local_ep):

            losses = AverageMeter()
            losses_x = AverageMeter()
            losses_u = AverageMeter()

            p_bar = tqdm(range(iteration))

            train_loader = zip(self.labeledtrain, self.unlabeledtrain)
            net.train()

            for batch_idx, (data_x, data_u) in enumerate(train_loader):
                inputs_x, targets_x = data_x
                (inputs_u_w, inputs_u_s), _ = data_u
                batch_size = inputs_x.shape[0]
                inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(self.args.device)
                targets_x = targets_x.to(self.args.device)
                net.zero_grad()
                logits = net(inputs)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits  # del语句作用在变量上，而不是数据对象上

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)  # dim=-1, torch.size的最里层方向（细粒度）
                mask = max_probs.ge(self.args.threshold).float()  # ge >=是torch.tensor对象有的属性

                Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

                loss = Lx + self.args.lambda_u * Lu
                loss.backward()

                losses.update(loss.item())
                losses_x.update(Lx.item())
                losses_u.update(Lu.item())

                optimizer.step()

                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}.".format(
                        epoch=local_epoch + 1,
                        epochs=self.args.local_ep,
                        batch=batch_idx + 1,
                        iter=iteration,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg, ))
                # mask=mask_prob))
                p_bar.update()
            p_bar.close()


        return net.state_dict(), losses.avg, losses_x.avg, losses_u.avg


def get_cifar(root, base_dataset, num_labeled, num_expand_x, num_expand_u):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),  # 上下左右各填充四个，即变为40*40
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    # print(dir(base_dataset))

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        base_dataset.targets, num_labeled, num_expand_x, num_expand_u, num_classes=10)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFix(mean=cifar10_mean, std=cifar10_std))

    return train_labeled_dataset, train_unlabeled_dataset

class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# 对于无监督数据的两种变化增广（弱和强）
class TransformFix(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def x_u_split(labels,
              num_labeled,
              num_expand_x,
              num_expand_u,
              num_classes):
    label_per_class = num_labeled // num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0] #返回一个array: 在数据集中标签为i的数据位置
        # print(len(idx))
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx)

    # print(len(labeled_idx))

    # 让labeled_idx长度为args.k_img, unlabeled_idx长度为mu*args.k_img

    exapand_labeled = num_expand_x // len(labeled_idx)  #16
    exapand_unlabeled = num_expand_u // len(unlabeled_idx) #9
    labeled_idx = np.hstack(
        [labeled_idx for _ in range(exapand_labeled)]) #循环16次，长度为4000*16
    unlabeled_idx = np.hstack(
        [unlabeled_idx for _ in range(exapand_unlabeled)])

    if len(labeled_idx) < num_expand_x:
        diff = num_expand_x - len(labeled_idx)
        labeled_idx = np.hstack(
            (labeled_idx, np.random.choice(labeled_idx, diff)))
    else:
        assert len(labeled_idx) == num_expand_x

    if len(unlabeled_idx) < num_expand_u:
        diff = num_expand_u - len(unlabeled_idx)
        unlabeled_idx = np.hstack(
            (unlabeled_idx, np.random.choice(unlabeled_idx, diff)))
    else:
        assert len(unlabeled_idx) == num_expand_u

    return labeled_idx, unlabeled_idx
