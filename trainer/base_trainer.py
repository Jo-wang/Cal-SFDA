import torch
import os.path as osp

import wandb
import torch.nn as nn
import statistics
from dataset import dataset
from tqdm import tqdm
import numpy as np
from utils.calibration import *
from utils.meters import AverageMeter as meter
from utils.logger import TensorboardLogger as logger

import pandas as pd
import collections
import functools
import operator

from utils.sce import SCELoss



class BaseTrainer(object):

    def __init__(self, models, optimizers, loaders, up_s, up_t, config,
                 writer):
        self.model = models
        self.optim = optimizers
        self.loader = loaders
        self.config = config
        self.output = {}
        self.up_src = up_s
        self.up_tgt = up_t
        self.writer = writer
        self.meter = meter(self.config.num_classes)
        self.ece = ECELoss()
        
    def set_para_adam(self, multigpu):
        if multigpu:
            for v in self.model.module.value_net.parameters():
                self.param_group += [{
                    'params': v,
                    'lr': self.config.learning_rate,
                    'weight_decay': self.config.weight_decay
                }]

            for v in self.model.module.encoder.parameters():
                self.param_group += [{
                    'params': v,
                    'lr': self.config.learning_rate,
                    'weight_decay': self.config.weight_decay
                }]

            for v in self.model.module.decoder.parameters():
                self.param_group += [{
                    'params': v,
                    'lr': self.config.learning_rate,
                    'weight_decay': self.config.weight_decay
                }]
        else: 
            for v in self.model.value_net.parameters():
                self.param_group += [{
                    'params': v,
                    'lr': self.config.learning_rate,
                    'weight_decay': self.config.weight_decay
                }]

            for v in self.model.encoder.parameters():
                self.param_group += [{
                    'params': v,
                    'lr': self.config.learning_rate,
                    'weight_decay': self.config.weight_decay
                }]

            for v in self.model.decoder.parameters():
                self.param_group += [{
                    'params': v,
                    'lr': self.config.learning_rate,
                    'weight_decay': self.config.weight_decay
                }]

    def set_para_group(self, multigpu, val_net=True):
        if multigpu:
            if val_net:
                for v in self.model.module.value_net.parameters():
                    self.param_group += [{
                        'params': v,
                        'lr': self.config.learning_rate,
                        'momentum': self.config.momentum,
                        'weight_decay': self.config.weight_decay,
                        'nesterov': True
                    }]
            if self.config.encoder_train:
                for v in self.model.module.encoder.parameters():
                    self.param_group += [{
                        'params': v,
                        'lr': self.config.learning_rate,
                        'momentum': self.config.momentum,
                        'weight_decay': self.config.weight_decay,
                        'nesterov': True
                    }]
            if self.config.decoder_train:
                for v in self.model.module.decoder.parameters():
                    self.param_group += [{
                        'params': v,
                        'lr': self.config.learning_rate,
                        'momentum': self.config.momentum,
                        'weight_decay': self.config.weight_decay,
                        'nesterov': True
                    }]
        else: 
            if val_net:
                for v in self.model.value_net.parameters():
                    self.param_group += [{
                        'params': v,
                        'lr': self.config.learning_rate,
                        'momentum': self.config.momentum,
                        'weight_decay': self.config.weight_decay,
                        'nesterov': True
                    }]
            if self.config.encoder_train:
                for v in self.model.encoder.parameters():
                    self.param_group += [{
                        'params': v,
                        'lr': self.config.learning_rate,
                        'momentum': self.config.momentum,
                        'weight_decay': self.config.weight_decay,
                        'nesterov': True
                    }]
            if self.config.decoder_train:
                for v in self.model.decoder.parameters():
                    self.param_group += [{
                        'params': v,
                        'lr': self.config.learning_rate,
                        'momentum': self.config.momentum,
                        'weight_decay': self.config.weight_decay,
                        'nesterov': True
                    }]

    def forward(self):
        pass

    def backward(self):
        pass

    def iter(self):
        pass

    def train(self):
        pass

    def save_model(self, iter):
        tmp_name = '_'.join((self.config.source, str(iter))) + '.pth'
        torch.save(self.model.state_dict(),
                   osp.join(self.config['snapshot'], tmp_name))

    def print_loss(self, iter, max_global_step, epoch):
        iter_infor = ('epoch {}, iter = {:6d}/{:6d}, exp = {}'.format(
            epoch, iter, max_global_step, self.config.note))
        to_print = [
            '{}:{:.4f}'.format(key, self.losses[key].item())
            for key in self.losses.keys()
        ]
        loss_infor = '  '.join(to_print)
        if self.config.screen:
            print(iter_infor + '  ' + loss_infor)
        if self.config.neptune:
            for key in self.losses.keys():
                neptune.send_metric(key, self.losses[key].item())
        if self.config.tensorboard and self.writer is not None:
            for key in self.losses.keys():
                self.writer.log_scalar('train/' + key, self.losses[key], iter)

    def ece(self, logits, label, train):
        ece = ECELoss()
        return ece(logits, label, train)

    def sce(self, pred, label, weight=None):

        sce = SCELoss(num_classes=self.config.num_classes)
        return sce(pred, label, weight=weight)

    def mse(self, output, label):

        mse = nn.MSELoss()
        return mse(output, label)

    def bce_loss(self, pred, real):
        pred = torch.clip(pred, 1e-7, 1 - 1e-7)
        return torch.mean(pred - real * pred + torch.log(1 + torch.exp(-pred)))

    def kl_div(self, p, q, reduction='mean'):

        kl = nn.KLDivLoss(reduction=reduction)
        loss = kl(p, q)
        return loss

    def validate(self, iter, data):
        self.model = self.model.eval()

        if data == 'gta5':
            testloader = dataset.init_test_dataset(
                self.config,
                data,
                set='val',
                list_path=self.config.gta5.data_list_val)
            interp = nn.Upsample(size=(1052, 1914),
                                 mode='bilinear',
                                 align_corners=True)
        elif data == 'synthia':
            testloader = dataset.init_test_dataset(
                self.config,
                data,
                set='val',
                list_path=self.config.synthia.data_list_val)
            interp = nn.Upsample(size=(760, 1280),
                                 mode='bilinear',
                                 align_corners=True)
        else:
            testloader = dataset.init_test_dataset(self.config,
                                                   data,
                                                   set='val')
            interp = nn.Upsample(size=(1024, 2048),
                                 mode='bilinear',
                                 align_corners=True)
        union = torch.zeros(self.config.num_classes, 1,
                            dtype=torch.float).cuda().float()
        inter = torch.zeros(self.config.num_classes, 1,
                            dtype=torch.float).cuda().float()
        preds = torch.zeros(self.config.num_classes, 1,
                            dtype=torch.float).cuda().float()
        all_ece = []

        class_ece_sum = []
        with torch.no_grad():
            for index, batch in tqdm(enumerate(testloader)):
                image, label, _, _, name = batch
                output = self.model(image.cuda())
                label = label.cuda()
                output = interp(output).squeeze()

                C, H, W = output.shape
                Mask = (label.squeeze()) < C

                pred_e = torch.linspace(0, C - 1, steps=C).view(C, 1, 1)
                pred_e = pred_e.repeat(1, H, W).cuda()

                pred = output.argmax(dim=0).float()
                pred_mask = torch.eq(pred_e, pred).byte()
                pred_mask = pred_mask * Mask.byte()

                label_e = torch.linspace(0, C - 1, steps=C).view(C, 1, 1)
                label_e = label_e.repeat(1, H, W).cuda()
                label = label.view(1, H, W)
                label_mask = torch.eq(label_e, label.float()).byte()
                label_mask = label_mask * Mask.byte()
                label_ece = label.squeeze().long()

                tmp_inter = label_mask + pred_mask.byte()

                cu_inter = (tmp_inter == 2).view(C,
                                                 -1).sum(dim=1,
                                                         keepdim=True).float()
                cu_union = (tmp_inter > 0).view(C,
                                                -1).sum(dim=1,
                                                        keepdim=True).float()
                cu_preds = pred_mask.view(C, -1).sum(dim=1,
                                                     keepdim=True).float()
                cu_ece = self.ece(output, label_ece, train=False).item()
                class_ece = self.class_level_ece(output,
                                                 label_ece,
                                                 train=False)
                class_ece_sum.append(class_ece)
                all_ece.append(cu_ece)

                union += cu_union
                inter += cu_inter
                preds += cu_preds

            iou = inter / union
            acc = inter / preds
            avg_ece = self.sum_dict(class_ece_sum)
            if C == 16:
                iou = iou.squeeze()
                class13_iou = torch.cat((iou[:3], iou[6:]))
                class13_iou = torch.where(torch.isnan(class13_iou), torch.full_like(class13_iou, 0), class13_iou)
                class13_miou = class13_iou.mean().item()
                wandb.log({data+'_val/13-class-mIoU': class13_miou},
                                       step=iter)
                print('13-Class mIoU:{:.2%}'.format(class13_miou))
            iou = torch.where(torch.isnan(iou), torch.full_like(iou, 0), iou)
            acc = torch.where(torch.isnan(acc), torch.full_like(acc, 0), acc)
            mIoU = iou.mean().item()
            mAcc = acc.mean().item()
            iou = iou.cpu().numpy()
            max_ece = max(all_ece)
            min_ece = min(all_ece)
            mean_ece = statistics.mean(all_ece)
            iou_list = iou.squeeze()
            iou_list = iou_list.tolist()

            try:
                wandb.log({data + '_val/iter-mIoU': mIoU}, step=iter)
                wandb.log(
                    {
                        data + '_val/iter-mean_ece': mean_ece,
                        data + '_val/iter-max_ece': max_ece,
                        data + '_val/iter-min_ece': min_ece
                    },
                    step=iter)
                    
                dict_iou = {}
                dict_ece = {}
                for cla in range(len(iou_list)):
                    key = data + '_val/iter-iou4class-{}'.format(cla)
                    dict_iou[key] = iou[cla]
                wandb.log(dict_iou, iter)
                for clas, ece in avg_ece.items():
                    key = data + '_val/iter-ece4class-{}'.format(clas)
                    dict_ece[key] = ece
                wandb.log(dict_ece, iter)
                    
            except Exception:
                print(
                    "Please check the epoch and cu_step input when vaildate! (None and not None)"
                )
            print('mIoU: {:.2%} mAcc : {:.2%} '.format(mIoU, mAcc))
            print('max-ece: {:.5%} min-ece : {:.5%} mean-ece : {:.5%} '.format(
                max_ece, min_ece, mean_ece))
        return mIoU

    def class_level_ece(self, output, label, train):
        if train:
            output = output.permute(1, 0, 2, 3)
        n_class = output.shape[0]
        ece_list = {}
        for i in range(n_class):
           
            pred_temp = output[:, label == i]
            label_s = label[label == i]
            class_ece = self.ece(pred_temp, label_s, train=False)
            ece_list[i] = class_ece
        return ece_list

    def class_level_ece_train(self, output, label, train):
     
        if train:
            output = output.permute(1, 0, 2, 3)
        n_class = output.shape[0]
        ece_list = []
        ece_dict = {}
        for i in range(n_class):
            pred_temp = output[:, label == i]
            label_s = label[label == i]
           
            class_ece = self.ece(pred_temp, label_s, train=False)
            ece_list.append(class_ece)
            ece_dict[i] = class_ece
        return [torch.mean(torch.stack(ece_list)), ece_dict]

    def sum_dict(self, dict_list):
        index = []
        ece = []
        for i in dict_list:
            for k, v in i.items():
                index.append(k)
                ece.append(v)
        count = pd.value_counts(index)
        count = count.to_dict()
        sum_ = dict(
            functools.reduce(operator.add, map(collections.Counter,
                                               dict_list)))
        avg = {x: float(sum_[x]) / count[x] for x in count}

        return avg

    def stat_label(self, label):
      
        label_count = pd.value_counts(label)
        label_count = label_count.to_dict()
        return label_count

    
    def prior_tensor(self, src_prior:dict, b:int, h:int, w:int) -> torch.Tensor:
     
        for i in range(self.config.num_classes):
            dim = src_prior[i]
            ones = torch.ones(b,1,h,w)
            cla = dim*ones
            if i == 0:
                all = cla
            else:
                all = torch.cat((all, cla), dim=1)
        return all

    def freeze(self):
      
        if not self.config.encoder_train:
            if self.config.multigpu:
                for _, para in self.model.module.encoder.named_parameters():
                    para.requires_grad = False
            else:
                for _, para in self.model.encoder.named_parameters():
                    para.requires_grad = False

        if not self.config.decoder_train:
            if self.config.multigpu:
                for _, para in self.model.module.decoder.named_parameters():
                    para.requires_grad = False
            else:
                for _, para in self.model.decoder.named_parameters():
                    para.requires_grad = False
        