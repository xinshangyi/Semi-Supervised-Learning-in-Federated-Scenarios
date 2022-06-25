import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import pickle as pkl

from utils.sampling import mnist_iid, mnist_noniid, mnist_irnoniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import Update
# from models.SemisuperUpdate import SemisuperUpdate
from models.UnsuperUpdate import UnsuperUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, get_JSweight, FedJS, FedAvg_pro
from models.test import test_img
from save.savefile import savexcl


from dataset.randaugment import RandAugmentMC

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

    all_idxs = [i for i in range(len(dataset_train))]
    global_idxs = np.random.choice(all_idxs, int(args.publicfrac * len(dataset_train)), replace=False)

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
    commu = []  # 2-dimensional
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # local_w_dict = {i: net_glob for i in range(args.num_users)}
    # local_test_dict = {i: [] for i in range(args.num_users)}

    # JS = get_JSweight(dataset_train, dict_users)

    cloud = Update(args=args, dataset=dataset_train, idxs=global_idxs, flag=1)
    w_cloud, _ = cloud.train(net=copy.deepcopy(net_glob).to(args.device))
    net_glob.load_state_dict(w_cloud)

    # for baseline
    acc_init, loss_init = test_img(net_glob, dataset_test, args)
    print('initial_loss: {:.4f}. initial_acc: {:.2f}.'
          .format(loss_init, acc_init.item()))


    for iter in range(args.epochs):

        # w_cloud, _ = cloud.train(net=copy.deepcopy(net_glob).to(args.device))
        # net_glob.load_state_dict(w_cloud)
        # acc_bf, loss_bf = test_img(net_glob, dataset_test, args)
        # print('Round {:3d}. before_finetune_loss: {:.4f}. before_finetune_acc: {:.2f}.'
        #       .format(iter + 1, loss_bf, acc_bf.item()))

        w_locals, loss_locals = [], []
        num_portion_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # rest_users = list(set(range(args.num_users)) - set(idxs_users))
        # JS_tmp = np.array(JS)[idxs_users]
        for idx in idxs_users:
            num_portion_locals.append(len(dict_users[idx]))  # 每一个client的数据量
            
            local = UnsuperUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss, loss_x, loss_u = local.train(net=copy.deepcopy(net_glob).to(args.device))
            print('Round {:3d}. User_id: {:3d}. Local_loss: {:.4f}. Local_lossx: {:.4f}. Local_lossu: {:.4f}'
                  .format(iter + 1, idx, loss, loss_x, loss_u))

            # # for local model performance observation
            # refernet = copy.deepcopy(net_glob)
            # refernet.load_state_dict(w)
            # local_w_dict[idx] = refernet

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        if args.global_aggregation:
            num_portion_locals.append(num_train - np.sum(num_portion_locals))
            w_locals.append(copy.deepcopy(net_glob.state_dict()))
        
        print(num_portion_locals)

        
        # # update global weights
        # w_glob = FedAvg(w_locals)
        w_glob = FedAvg_pro(w_locals, num_portion_locals)
        # w_glob = FedJS(w_locals, JS_tmp)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        '''
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        '''
        acc_bf, loss_bf = test_img(net_glob, dataset_test, args)
        print('Round {:3d}. before_finetune_loss: {:.4f}. before_finetune_acc: {:.2f}.'
              .format(iter + 1, loss_bf, acc_bf.item()))

        # finetune
        w_cloud, _ = cloud.train(net=copy.deepcopy(net_glob).to(args.device))
        net_glob.load_state_dict(w_cloud)

        # after every communication, test the global model
        dataset_train2 = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_val)
        acc_traintmp, loss_traintmp = test_img(net_glob, dataset_train2, args)
        acc_testtmp, loss_testtmp = test_img(net_glob, dataset_test, args)
        acc_traintmp = acc_traintmp.item()
        acc_testtmp = acc_testtmp.item()
        acc_bftmp = acc_bf.item()
        commu_each = [loss_traintmp, loss_testtmp, acc_traintmp, acc_testtmp, acc_bftmp]
        print('Round {:3d}, Training loss {:.3f}'.format(iter + 1, loss_traintmp))
        print('Round {:3d}, Testing loss {:.3f}'.format(iter + 1, loss_testtmp))
        print("Training accuracy: {:.2f}".format(acc_traintmp))
        print("Testing accuracy: {:.2f}".format(acc_testtmp))
        print("Testing accuracy before finetune: {:.2f}".format(acc_bftmp))
        loss_train.append(loss_traintmp)
        loss_test.append(loss_testtmp)
        acc_train.append(acc_traintmp)
        acc_test.append(acc_testtmp)
        commu.append(commu_each)

        # # after every communication, every client test the global model
        # for idx in range(args.num_users):
        #     acc_testtmp, loss_testtmp = test_img(local_w_dict[idx], dataset_test, args)
        #     local_test_dict[idx].append(acc_testtmp.item())
        # result = np.array([local_test_dict[i] for i in range(args.num_users)])
        # savepath = './save/disjoint_result.npy'
        # np.save(savepath, result)


    # savefile
    if args.noniid_type == 1:
        savexcl(commu, './save/disjoint_{}_{}_C{}_publicC{}_iid{}_{}{}_GA{}.xlsx'.format(args.dataset, args.epochs, args.frac, args.publicfrac, args.iid, args.loss_type, args.tempera, args.global_aggregation))
    else:
        savexcl(commu, './save/disjoint_{}_{}_C{}_publicC{}_beta{}_{}{}_GA{}.xlsx'.format(args.dataset, args.epochs, args.frac, args.publicfrac, args.beta, args.loss_type, args.tempera, args.global_aggregation))
    print('Best test accuracy: {:.2f}'.format(np.max(acc_test)))
