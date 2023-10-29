import torch
import wandb
from utils.optimize import *
from .base_trainer import BaseTrainer
import os.path as osp
from dataset import dataset
import torch.optim as optim
from tqdm import tqdm
import math
import copy
from utils.calibration import *
from utils.optimize import adjust_learning_rate
from utils.poly_loss import poly1_CE
import statistics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Trainer(BaseTrainer):

    def __init__(self, model, config):
        self.model = model
        self.config = config

        
    def bce_loss(self, output, target):
        output_neg = 1 - output
        target_neg = 1 - target
        result = torch.mean(target * torch.log(output + 1e-6))
        result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
        return -torch.mean(result)

    def iter(self, batch, alpha, iter):
        img, seg_label, _, _, _ = batch
        seg_label = seg_label.long().cuda()
        img = img.cuda()
        return_list = self.model(img, reverse=False)
        seg_pred = return_list[0]  
        ece_pred = return_list[1]
        if self.config.encoder_train and self.config.decoder_train:
            loss_seg = F.cross_entropy(seg_pred, seg_label, ignore_index=255)
        
        class_ece_list = torch.ones(seg_pred.shape[0]).to(seg_pred.device)
        for i in range(seg_pred.shape[0]):
            class_ece, _ = self.class_level_ece_train(seg_pred[i].unsqueeze(0), seg_label[i].unsqueeze(0), train=True)
            class_ece_list[i] = class_ece
        class_avg_ece = torch.mean(class_ece_list)

        mse_list = torch.ones(seg_pred.shape[0]).to(seg_pred.device)
        for i in range(class_ece_list.shape[0]):
            mse_loss = self.mse(class_ece_list[i], ece_pred[i])
            mse_list[i] = mse_loss
        mse_loss = torch.mean(mse_list)
        
        class_total_loss = mse_loss 

        wandb.log({
                   'train/mse_loss': mse_loss,
                   'train/ece_loss': class_avg_ece
                    }, step=iter)
        class_total_loss.backward()


    def train(self):
        
        self.param_group = []

        self.set_para_group(self.config.multigpu)

        self.optimizer = optim.SGD(self.param_group)

        self.loader, _ = dataset.init_source_dataset(
            self.config)  
        
        for epoch in range(self.config.epochs):
            for i_iter, batch in tqdm(enumerate(self.loader)):
                cu_step = i_iter + epoch * len(self.loader)
                self.optimizer.zero_grad()
                if self.config.encoder_train and self.config.decoder_train:
                    self.model.train()
                elif not self.config.encoder_train and not self.config.decoder_train:
                    self.freeze()
                    self.model.encoder.eval()
                    self.model.decoder.eval()
                    self.model.value_net.train()
                else:
                    import sys
                    print("check config for encoder and decoder train")
                    sys.exit(0)

                adjust_learning_rate(self.optimizer, cu_step, len(self.loader),
                                     self.config)

                self.iter(batch, self.config.alpha, cu_step)

                if self.config.multigpu:
                    if self.config.encoder_train:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.module.encoder.parameters(), self.config.clip)
                    if self.config.decoder_train:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.module.decoder.parameters(), self.config.clip)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.module.value_net.parameters(), self.config.clip)
                else:
                    if self.config.encoder_train:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.encoder.parameters(), self.config.clip)
                    if self.config.decoder_train:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.decoder.parameters(), self.config.clip)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.value_net.parameters(), self.config.clip)

                self.optimizer.step()

            with torch.no_grad():

                val_src = self.validate(iter=cu_step,
                                           data=self.config.source)
                
                self.save_model(step=cu_step, src=val_src)

            self.config.learning_rate = self.config.learning_rate / (math.sqrt(2))
            
        print('Training finished !')
        wandb.finish()

    def resume(self):
        self.tea = copy.deepcopy(self.model)
        self.round_start = self.config.round_start  
        print('Resume from Round {}'.format(self.round_start))
        if self.config.lr_decay == 'sqrt':
            self.config.learning_rate = self.config.learning_rate / (
                (math.sqrt(2))**self.round_start)

    def save_model(self, step, src):
        dirs = self.config.source + str(src)
        tmp_name = '_'.join((dirs, str(step))) + '.pth'
        torch.save(self.model.state_dict(),
                       osp.join(self.config['snapshot'], tmp_name))

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

        union = torch.zeros(self.config.num_classes, 1,
                            dtype=torch.float).cuda().float()
        inter = torch.zeros(self.config.num_classes, 1,
                            dtype=torch.float).cuda().float()
        preds = torch.zeros(self.config.num_classes, 1,
                            dtype=torch.float).cuda().float()
        all_ece = []
        all_mse = []
        class_ece_sum = []
        with torch.no_grad():
            for _, batch in tqdm(enumerate(testloader)):
                image, label, _, _, name = batch 

                output = self.model(image.cuda())
                seg_pred = output[0]
                ece_pred = output[1]
                label = label.cuda()
                output = interp(seg_pred).squeeze()
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
                np_ece = np.array(cu_ece)
                mse = self.mse(
                    ece_pred,
                    torch.Tensor(np_ece).to(
                        ece_pred.device)) 
                class_ece = self.class_level_ece(
                    output, label_ece,
                    train=False)  
                class_ece_sum.append(class_ece)
                all_ece.append(cu_ece)
                all_mse.append(mse.item())

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
            max_mse = max(all_mse)
            min_mse = min(all_mse)
            mean_mse = statistics.mean(all_mse)
            iou_list = iou.squeeze()
            iou_list = iou_list.tolist()

            try:
        
                wandb.log({data + '_val/iter-mIoU': mIoU}, step=iter)
                wandb.log({data + '_val/iter-mean_ece': mean_ece},
                              step=iter)
                wandb.log({data + '_val/iter-max_ece': max_ece}, step=iter)
                wandb.log({data + '_val/iter-min_ece': min_ece}, step=iter)
                wandb.log({data + '_val/iter-mean_mse': mean_mse},
                              step=iter)
                wandb.log({data + '_val/iter-max_mse': max_mse}, step=iter)
                wandb.log({data + '_val/iter-min_mse': min_mse}, step=iter)
                for cla in range(len(iou_list)):
                    wandb.log(
                        {
                            data + '_val/iter-iou4class-{}'.format(cla):
                            iou[cla]
                        },
                        step=iter)
                for clas, ece in avg_ece.items():
                    wandb.log(
                        {
                            data + '_val/iter-ece4class-{}'.format(clas):
                            ece
                        },
                        step=iter)
            except Exception:
                print(
                    "Please check the epoch and cu_step input when vaildate! (None and not None)"
                )
            print('mIoU: {:.2%} mAcc : {:.2%} '.format(mIoU, mAcc))
            print('max-ece: {:.5%} min-ece : {:.5%} mean-ece : {:.5%} '.format(
                max_ece, min_ece, mean_ece))
        return mIoU

    def group_data_by_class(self, data, label):

        data_dict = {}
        for i in range(self.config.num_classes):
            data_dict[i] = data[:, i, :, :]
        return data_dict
