import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from utils.misc import AverageMeter
import torch.nn.functional as F
import copy

from dataset.randaugment import RandAugmentMC
from .SemisuperUpdate import CIFAR10SSL, TransformFix

from tqdm import tqdm

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


class UnsuperUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        # self.loss_func = nn.CrossEntropyLoss()
        # self.selected_clients = []
        unlabeled_dataset = CIFAR10SSL(
            './data/cifar', idxs, train=True,
        transform=TransformFix(mean=cifar10_mean, std=cifar10_std))
        self.k_img = len(unlabeled_dataset)

        self.ldr_train = DataLoader(unlabeled_dataset, batch_size=args.local_bs, shuffle=True)


    def train(self, net):

        iteration = self.k_img // self.args.local_bs
        base_model = copy.deepcopy(net).to(self.args.device)

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

        for local_epoch in range(self.args.local_ep):

            losses = AverageMeter()
            losses_x = AverageMeter()
            losses_u = AverageMeter()

            p_bar = tqdm(range(iteration))

            net.train()
            base_model.eval()

            for batch_idx, data_u in enumerate(self.ldr_train):
                (inputs_u_w, inputs_u_s), _ = data_u
                batch_size = inputs_u_w.shape[0]
                inputs = torch.cat((inputs_u_w, inputs_u_s)).to(self.args.device)
                net.zero_grad()
                logits = net(inputs)
                logits_u_w, logits_u_s = logits.chunk(2)
                del logits  # del语句作用在变量上，而不是数据对象上

                dis_logits = base_model(inputs_u_w.to(self.args.device))
                dis_logits_pro = dis_logits.true_divide(self.args.tempera)
                logits_u_w_pro = logits_u_w.true_divide(self.args.tempera)
                logsoftmax = nn.LogSoftmax(dim=1)
                lsm_logits_u_w = logsoftmax(logits_u_w_pro)

                dis_target = F.softmax(dis_logits_pro, dim=1)
                Lx = torch.sum(-lsm_logits_u_w * dis_target, dim=1)
                Lx = self.args.tempera**2 * Lx

                pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)  # dim=-1, torch.size的最里层方向（细粒度）
                mask = max_probs.ge(self.args.threshold).float()  # ge >=是torch.tensor对象有的属性

                Lu = F.cross_entropy(logits_u_s, targets_u, reduction='none')

                # 标准GDST
                if self.args.loss_type == 'GDST':
                    loss = Lx.mean() + self.args.lambda_u * ((Lu * mask).mean())
                # 标准ST
                elif self.args.loss_type == 'ST':
                    loss = (Lu * mask).mean()
                # KD_only
                elif self.args.loss_type == 'KD':
                    loss = Lx.mean()
                else:
                    exit('Error: unrecognized local loss type')
                loss.backward()

                losses.update(loss.item())
                # losses_x.update((Lx * mask).mean().item())
                losses_x.update(Lx.mean().item())
                losses_u.update((Lu * mask).mean().item())

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
