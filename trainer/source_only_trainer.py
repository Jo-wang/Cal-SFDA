import torch
import wandb
from utils.optimize import *
from .base_trainer import BaseTrainer
import os.path as osp
from dataset import dataset
import  torch.optim as optim
from tqdm import tqdm
import math
import copy
import torch.nn.functional as F
from utils.calibration import *
from utils.meters import AverageMeter as meter
from utils.cal_viz import *
import pandas as pd
import collections
import functools
import operator

class Trainer(BaseTrainer):
    def __init__(self, model, config):
        self.model = model

        self.config = config
        self.ece_loss = ECELoss()
        self.meter = meter(self.config.num_classes)

    
    def iter(self, batch, alpha, iter): 
        img, seg_label, _, _, _ = batch
        seg_label = seg_label.long().cuda()
        img = img.cuda()
       
        seg_pred = self.model(img, reverse=False)   
        cross_entropy = F.cross_entropy(seg_pred, seg_label, ignore_index=255)
      
        if alpha > 0:
            class_ece_list = torch.ones(seg_pred.shape[0]).to(seg_pred.device)
            for i in range(seg_pred.shape[0]):
                class_ece, _ = self.class_level_ece_train(seg_pred[i].unsqueeze(0), seg_label[i].unsqueeze(0), train=True)
                class_ece_list[i] = class_ece
            class_avg_ece = torch.mean(class_ece_list)
            loss_seg_all = cross_entropy + alpha * class_avg_ece
            wandb.log({
            'train/ce': cross_entropy,
            'train/ece': class_avg_ece,
            'train/ce+ece_loss': loss_seg_all}, step=iter)
        else:
            loss_seg_all = cross_entropy 
            wandb.log({
                'train/ce': loss_seg_all}, step=iter)
            
        loss_seg_all.backward()
  
        
    def train(self):
        self.param_group = []

        self.set_para_group(self.config.multigpu, val_net=False)

        self.optim = optim.SGD(self.param_group)

        self.loader, _ = dataset.init_source_dataset(self.config)
        
        for epoch in range(self.config.epochs):
            for i_iter, batch in tqdm(enumerate(self.loader)):
                cu_step = i_iter + epoch * len(self.loader)
                
                self.optim.zero_grad()
                self.model = self.model.train()
                adjust_learning_rate(self.optim, cu_step, len(self.loader), self.config)
                if cu_step < self.config.ece_train:
                    alpha = 0
                else:
                    alpha = self.config.alpha
                self.iter(batch, alpha, cu_step)

                self.optim.step()
                   
            miou = self.validate(iter=cu_step, data=self.config.source)
          
                
    def resume(self):
        self.tea = copy.deepcopy(self.model)
        self.round_start = self.config.round_start 
        print('Resume from Round {}'.format(self.round_start))
        if self.config.lr_decay == 'sqrt':
            self.config.learning_rate = self.config.learning_rate/((math.sqrt(2))**self.round_start)

    def save_model(self, step, src, tar):
        dirs = self.config.source + str(src) + '_iter'
        tmp_name = '_'.join((dirs, str(step))) + '.pth'
        torch.save(self.model.state_dict(),
                       osp.join(self.config['snapshot'], tmp_name))

    def class_level_ece_train(self, output, label, train):
        if train:
            output = output.permute(1, 0, 2, 3)
        n_class = output.shape[0]
        ece_ = torch.ones([n_class]).to(output.device)
        for i in range(n_class):
            pred_temp = output[:, label == i]
            label_s = label[label == i]
            class_ece = self.ece(pred_temp, label_s, train=False)
            ece_[i] = class_ece

        return [torch.mean(ece_), None]

    def sum_dict(self, dict_list):
      
        index = []
        ece = []
        for i in dict_list:
            for k, v in i.items():
                index.append(k)
                ece.append(v)
        count = pd.value_counts(index)
        count = count.to_dict()
        sum_ = dict(functools.reduce(operator.add,
                                     map(collections.Counter, dict_list)))
        avg = {x: float(sum_[x])/count[x] for x in count}

        return avg

    

