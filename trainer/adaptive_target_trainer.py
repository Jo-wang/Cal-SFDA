import torch
from utils.optimize import adjust_learning_rate
from .base_trainer import BaseTrainer
import os
import os.path as osp
from dataset import dataset
import math
from PIL import Image
from utils.meters import AverageMeter, GroupAverageMeter
import torch.nn.functional as F
import random
from utils.func import Acc, thres_cb_plabel, gene_plabel_prop, mask_fusion
import  torch.optim as optim
from utils.pool import Pool
from utils.flatwhite import *
from trainer.base_trainer import *
from operator import itemgetter

import wandb

criterion_nll = nn.NLLLoss()

class Trainer(BaseTrainer):

    def __init__(self, model, config):
        self.model = model
        self.model.train()
        self.config = config

    def dict2tensor(self, dic):
        list = []
        for i in range(self.config.num_classes):
            att = dic[i]
            if att == 0:
                att = 0.01
            weig = 1 / att
            list.append(weig)

        return torch.cuda.FloatTensor(list)

    def cross_entropy(self, pred, label, weight):
        return F.cross_entropy(pred, label, weight=weight, ignore_index=255)

    def entropy_loss(self, p):
        p = F.softmax(p, dim=1)
        log_p = F.log_softmax(p, dim=1)
        loss = -torch.sum(p * log_p, dim=1)
        return loss

    def iter(self, batch, iter):
        img_s, label_s, gt, _, _ = batch

        pred_s, ece, feature = self.model(img_s.cuda(), reverse=False)

        with torch.no_grad():
            for i in range(feature.shape[0]):
                in_fea = feature[i].reshape(-1, feature.shape[2]*feature.shape[3])
                 
                self.nuc += torch.norm(in_fea, p='nuc')
                self.num += 1
        
        self.estece += ece.sum().item()
        
        label_s = label_s.long().cuda()
        thres_weight = F.softmax(self.dict2tensor(self.cb_thres))
        loss_seg = self.sce(pred_s, label_s, weight=thres_weight)
        
        loss_e = self.entropy_loss(pred_s)
        max_ent = torch.max(loss_e)

        loss_e = loss_e / max_ent

        loss_e = loss_e**2.0 + 1e-8
        loss_e = loss_e**2.0
        loss_e = loss_e.mean()
        
        pred_s1 = pred_s.permute(0, 2, 3, 1).contiguous().view(
            -1, self.config.num_classes) 
        pred_s_softmax = F.softmax(pred_s1, -1)
        label_s = label_s.view(-1)
        width = 3
        k = self.config.num_classes // 2 + random.randint(-width, width)
        _, labels_neg = torch.topk(pred_s_softmax, k, dim=1, sorted=True)
        s_neg = torch.log(torch.clamp(1. - pred_s_softmax, min=1e-5, max=1.))
        labels_neg = labels_neg[:, -1].squeeze().detach()
        loss_neg = criterion_nll(s_neg, labels_neg)
        
        loss_entropy = self.config.eta * loss_e + loss_seg + loss_neg 
        
        self.running_ent += loss_e.item()
        wandb.log(
            {
                'train/ce': loss_seg,
                'train/neg': loss_neg,
                'train/ent': loss_e,
                'train/total_loss': loss_entropy
            },
            step=iter)
        loss_entropy.backward()


    def eval(self, iter):
        mIoU = self.validate(iter=iter, data=self.config.target)
        return self.config.rl_restore_from, mIoU

    def train(self):
        self.round_start = self.config.round_start
        
        if self.config.resume:
            self.gb_step = 59439
        else:  
            self.gb_step = 0
        for r in range(self.config.round_start, self.config.round):
            self.source_all = get_list(self.config.synthia.data_list)
            self.target_all = get_list(self.config.cityscapes.data_list)

            if r != 0:
                self.config.cb_prop += (0.05 * r)
            self.cb_thres = self.gene_thres(self.config.cb_prop, None, self.config.num_classes)
            self.plabel_path = osp.join(self.config.plabel, self.config.note, str(r))
            self.save_pred(r, self.cb_thres)
            
            if self.config.optim == 'sgd':
                self.param_group = []
                if self.config.freeze_value_net:
                    self.set_para_group(self.config.multigpu, val_net=False)
                else:
                    self.set_para_group(self.config.multigpu, val_net=True)
                self.optimizer = optim.SGD(self.param_group)

            elif self.config.optim == 'adam':
                self.param_group = []
                self.set_para_adam(self.config.multigpu)
                self.optimizer = optim.Adam(self.param_group)

            self.loader = dataset.init_target_dataset(
                self.config,
                plabel_path=self.plabel_path,
                target_selected=self.target_all)
            
            for epoch in range(self.config.epochs):
                self.estece = 0.0
                self.running_ent = 0
                self.num = 0.0
                self.nuc = 0.0
                
                for i_iter, batch in tqdm(enumerate(self.loader)):
                    cu_step = epoch * len(self.loader) + i_iter
                    if epoch == i_iter == 0 and not self.config.resume:
                        self.gb_step = 0
                    else:
                        self.gb_step += 1

                    self.optimizer.zero_grad()

                    if epoch == 0 and not self.config.resume:
                        self.model.train()
                        self.setup_tent(val_net=True)
                    else:
                        self.set_require_grad(val_net=True)
                        self.model.train()
                        self.freeze_encoder()
                        self.freeze_value_net()

                    adjust_learning_rate(self.optimizer, self.gb_step, len(self.loader), self.config)

                    self.iter(batch, self.gb_step)
                    self.optimizer.step()
           
                
                with torch.no_grad():
                    nuc = self.nuc / self.num
                    wandb.log({'train/nuc':nuc}, step=self.gb_step)
                    ece = self.estece / len(self.loader) 
                    wandb.log({'train/mean_estece': ece}, step=self.gb_step)
                    mean_ent = self.running_ent / len(self.loader)
                    wandb.log({'train/mean_ent': mean_ent}, step=self.gb_step)

                self.save_model(r=r, iter=self.gb_step)

            self.config.learning_rate = self.config.learning_rate / (
                math.sqrt(2))
        wandb.finish()

    def resume(self):
        iter_num = self.config.init_weight[-5]
        iter_num = int(iter_num)
        self.round_start = int(math.ceil((iter_num + 1) / self.config.epochs))
        print("Resume from Round {}".format(self.round_start))
        if self.config.lr_decay == "sqrt":
            self.config.learning_rate = self.config.learning_rate / (
                (math.sqrt(2))**self.round_start)

    def cluster_subdomain(self, ent_list, thres):
        entropy_list = sorted(ent_list.items(),
                              key=itemgetter(1),
                              reverse=False)  
        copy_list = entropy_list.copy()
        entropy_rank = [item[0] for item in entropy_list]

        easy_split = entropy_rank[:int(len(entropy_rank) * thres)]
        hard_split = entropy_rank[int(len(entropy_rank) * thres):]

        with open(self.config.easy_split, 'w+') as f:
            for item in easy_split:
                f.write('%s\n' % item)

        with open(self.config.hard_split, 'w+') as f:
            for item in hard_split:
                f.write('%s\n' % item)

        return copy_list

    def gene_thres(self, prop, prior, num_cls):
        print('[Calculate Threshold using config.cb_prop]')

        probs = {}
        freq = {}
        loader = dataset.init_test_dataset(self.config,
                                           self.config.target,
                                           set="train",
                                           selected=self.target_all,
                                           batchsize=1)

        for index, batch in tqdm(enumerate(loader)):
            img, _, _, _, _ = batch
            with torch.no_grad():
                output = self.model(img.cuda(), reverse=False)
                pred = output[0]
                pred = F.softmax(pred, dim=1)
                pred_ece = output[1]

            ece_weight = 1 - pred_ece.mean()
            pred_probs = pred.max(dim=1)[0]
            pred_probs = pred_probs.squeeze()

            pred_probs = ece_weight * pred_probs

            pred_label = torch.argmax(pred, dim=1).squeeze()
            for i in range(num_cls):
                cls_mask = pred_label == i
                cnt = cls_mask.sum()
                if cnt == 0: 
                    continue
                cls_probs = torch.masked_select(pred_probs, cls_mask)
                cls_probs = cls_probs.detach().cpu().numpy().tolist()
                cls_probs.sort()
                if i not in probs:
                    probs[i] = cls_probs[::5]
                else:
                    probs[i].extend(cls_probs[::5])
        thres = {}
        for i in range(num_cls):
            if i in probs.keys():
                pass
            else:
                thres[i] = 0.5
        for k in probs.keys():
            cls_prob = probs[k]
            cls_total = len(cls_prob)
            freq[k] = cls_total
            cls_prob = np.array(cls_prob)
            cls_prob = np.sort(cls_prob)
            index = int(cls_total * prop)
            cls_thres = cls_prob[-index]

            thres[k] = cls_thres
        print(thres)
        return thres

    def save_pred(self, round, thres):

        print("[Generate pseudo labels]")

        if round == 0:
            plabel_path = osp.join(self.config.plabel, str(round))

            loader = dataset.init_test_dataset(self.config,
                                               self.config.target,
                                               set="train",
                                               selected=self.target_all)
        else:
            plabel_path = osp.join(self.config.plabel, self.config.note,
                                   str(round - 1))
            loader = dataset.init_test_dataset(self.config,
                                               self.config.target,
                                               set="train",
                                               selected=self.target_all,
                                               plabel_path=plabel_path)

        interp = nn.Upsample(size=(1024, 2048),
                             mode="bilinear",
                             align_corners=True)

        mkdir(self.plabel_path)
        self.config.target_data_dir = self.plabel_path
        self.pool = Pool(
        )  
        accs = AverageMeter() 
        props = AverageMeter()  
        cls_acc = GroupAverageMeter() 

        self.mean_memo = {i: [] for i in range(self.config.num_classes)}
        with torch.no_grad():
            for index, batch in tqdm(enumerate(loader)):
                if round != 0:
                    image, plab, label, img2, name = batch
                    label = label.cuda()
                    plab = plab.cuda()
                else:
                    image, label, gt, img2, name = batch
                    label = label.cuda()

                img_name = name[0].split("/")[-1]
                dir_name = name[0].split("/")[0]
                img_name = img_name.replace("leftImg8bit", "gtFine_labelIds")
                temp_dir = osp.join(self.plabel_path, dir_name)
                if not os.path.exists(temp_dir):
                    os.mkdir(temp_dir)
                output = self.model(image.cuda(), reverse=False)
                output = interp(output[0])

                if round == 0:
                    plab = None

                mask, plabel = thres_cb_plabel(plab,
                                               output,
                                               thres,
                                               ece=None,
                                               num_cls=self.config.num_classes)

                if round >= 0:
                    local_prop = self.config.local_prop
                    if round > 0:
                        local_prop += 0.05 * round
                mask2, plabel2 = gene_plabel_prop(plab,
                                                  output,
                                                  local_prop,
                                                  ece=None)
                mask, plabel = mask_fusion(output, mask, mask2)

                acc, prop, cls_dict = Acc(plabel,
                                          label,
                                          num_cls=self.config.num_classes)
                cnt = (plabel != 255).sum().item()
                accs.update(acc, cnt)
                props.update(prop, 1)
                cls_acc.update(cls_dict)
                plabel = plabel.view(1024, 2048)
                plabel = plabel.cpu().numpy()

                plabel = np.asarray(plabel, dtype=np.uint8)
                plabelz = Image.fromarray(plabel)
                plabelz.save("%s/%s.png" % (temp_dir, img_name.split(".")[0]))
           
            print("finished")
            wandb.log({"acc/acc_cb": accs.avg.item()}, step=self.gb_step)
            wandb.log({"acc/prop_cb": props.avg.item()}, step=self.gb_step)
            for i in cls_acc.avg.keys():
                wandb.log({"class-acc-" + str(i): cls_acc.avg[i]},
                          step=self.gb_step)
        print('The Accuracy :{:.2%} and proportion :{:.2%} of Pseudo Labels'.
              format(accs.avg.item(), props.avg.item()))


    def cal_class(self, plabel, label):
        plabel = plabel.view(-1, 1).squeeze().float()
        label = label.view(-1, 1).squeeze().float()

        mask = (plabel != 255) * (label != 255)

        label = torch.masked_select(label, mask)
        plabel = torch.masked_select(plabel, mask)
        vp, fp = torch.unique(plabel, return_counts=True)
        vg, fg = torch.unique(label, return_counts=True)


    def sum_dic(self, l, f):
        total_i = 0
        sum = {}
        for i in range(self.config.num_classes):
            for j in range(len(l)):
                if l[j] == i:
                    total_i += f[j]
            sum[i] = total_i
            total_i = 0
        return sum


    def save_model(self, r, iter):
      
        name =  str(r) + "gb-iter" + str(iter)

        tmp_name = '_'.join((self.config.target, name)) + '.pth'
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
        all_mse = []
        class_ece_sum = []
        with torch.no_grad():
            nuc = 0.0
            num = 0.0
            running_ent = 0.0
            for _, batch in tqdm(enumerate(testloader)):
                image, label, _, _, _ = batch  

                output = self.model(image.cuda())
                seg_pred = output[0]
                ece_pred = output[1]

                label = label.cuda()
                output = interp(seg_pred).squeeze()

                loss_e = self.entropy_loss(output)
                max_ent = torch.max(loss_e)
               
                loss_e = loss_e / max_ent
               
                loss_e = loss_e**2.0 + 1e-8
                loss_e = loss_e**2.0
                loss_e = loss_e.mean()
                running_ent += loss_e.item()
       

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

            mean_ent = running_ent / len(testloader)
            wandb.log({data + '_val/iter-ent': mean_ent}, step=iter)

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


    def setup_tent(self, val_net=False):

        if self.config.multigpu:
            for name, para in self.model.module.encoder.named_parameters():
                if not 'bn' in name:
                    para.requires_grad = False
                if 'bn' in name:
                    para.requires_grad = True
                    para.track_running_stats = False
                    para.running_mean = None
                    para.running_var = None
            for name, para in self.model.module.decoder.named_parameters():
                if not 'bn' in name:
                    para.requires_grad = False
                elif 'bn' in name:
                    para.requires_grad = True
                    para.track_running_stats = False
                    para.running_mean = None
                    para.running_var = None
            if val_net:
                for name, para in self.model.module.value_net.named_parameters():
                    if not 'bn' in name:
                        para.requires_grad = False
                    elif 'bn' in name:
                        para.requires_grad = True
                        para.track_running_stats = False
                        para.running_mean = None
                        para.running_var = None

        else:
            for name, para in self.model.encoder.named_parameters():
                if not 'bn' in name:
                    para.requires_grad = False
                elif 'bn' in name:
                    para.requires_grad = True
                    para.track_running_stats = False
                    para.running_mean = None
                    para.running_var = None
            for name, para in self.model.decoder.named_parameters():
                if not 'bn' in name:
                    para.requires_grad = False
                elif 'bn' in name:
                    para.requires_grad = True
                    para.track_running_stats = False
                    para.running_mean = None
                    para.running_var = None
            if val_net:
                for name, para in self.model.value_net.named_parameters():
                    if not 'bn' in name:
                        para.requires_grad = False
                    elif 'bn' in name:
                        para.requires_grad = True
                        para.track_running_stats = False
                        para.running_mean = None
                        para.running_var = None

    def freeze_encoder(self):
       
        if self.config.freeze_encoder == True:
            if self.config.multigpu:
                for _, para in self.model.module.encoder.named_parameters():
                    para.requires_grad = False
            else:
                for _, para in self.model.encoder.named_parameters():
                    para.requires_grad = False
      

    def unfreeze_encoder(self):
        if self.config.freeze_encoder == False:
            if self.config.multigpu:
                for name, para in self.model.module.encoder.named_parameters():
                    para.requires_grad = True
            else:
                for name, para in self.model.encoder.named_parameters():
                    para.requires_grad = True

    def freeze_value_net(self):
        if self.config.multigpu:
            for _, para in self.model.module.value_net.named_parameters():
                para.requires_grad = False
        else:
            for _, para in self.model.value_net.named_parameters():
                para.requires_grad = False
                
    def set_require_grad(self, val_net=True):
        if not self.config.multigpu:
            for name, para in self.model.encoder.named_parameters():
                para.requires_grad = True
                if 'bn' in name:
                    para.track_running_stats = True
                
            for name, para in self.model.decoder.named_parameters():
                para.requires_grad = True
                if 'bn' in name:
                    para.track_running_stats = True
            if val_net:
                for name, para in self.model.value_net.named_parameters():
                    para.requires_grad = True
                    if 'bn' in name:
                        para.track_running_stats = True
        else:
            for name, para in self.model.module.encoder.named_parameters():
                para.requires_grad = True
                if 'bn' in name:
                    para.track_running_stats = True
               
            for name, para in self.model.module.decoder.named_parameters():
                para.requires_grad = True
                if 'bn' in name:
                    para.track_running_stats = True
            if val_net:
                for name, para in self.model.module.value_net.named_parameters():
                    para.requires_grad = True
                    if 'bn' in name:
                        para.track_running_stats = True
