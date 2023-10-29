import torch
import torch.nn.functional as F
import numpy as np
import denseCRF
import sys
def Acc(plabel, label, num_cls=19):
    plabel = plabel.view(-1, 1).squeeze().cuda().float()
    label = label.view(-1, 1).squeeze().cuda().float()
    total = label.shape[0]
    mask = (plabel!=255)*(label!=255)

    label = torch.masked_select(label, mask)
    plabel = torch.masked_select(plabel, mask)
    correct = label == plabel
    acc_dict= {} 
    for i in range(num_cls):
    
        cls_mask = label == i
        cls_total = cls_mask.sum().float().item()
        if cls_total == 0:
            continue
        cls_correct = (cls_mask * correct).sum().float().item()
        cls_acc = cls_correct / cls_total
        acc_dict[i] = (cls_acc, cls_total)
    correct_cnt = correct.sum().float()
    selected = mask.sum().float()
    if selected==0:
        acc = 0.0
    else:
        acc = correct_cnt / selected
    prop = selected / total
    
    return acc, prop, acc_dict

 
def thres_cb_plabel(plab, P, thres_dic, ece=None, num_cls=19, label=None):
    
    n,c,h,w = P.shape
    P = F.softmax(P, dim=1)
    n,c,h,w = P.shape
    if label is not None:
        pred_label = label
    else:
        pred_label = P.argmax(dim=1).squeeze()
    vec = [torch.Tensor([thres_dic[k]]) for k in range(num_cls)]
    vec = torch.stack(vec).cuda()
    vec = vec.view(1, c, 1, 1)
    vec = vec.expand_as(P)
    if ece is not None:
        P = 2 * ((1 - ece) * P) / ((1 - ece) + P)

    mask = torch.gt(P, vec)  
    mask = mask.sum(dim=1).byte()
    ignore = 255 * torch.ones_like(pred_label, dtype=torch.long)

    plabel = torch.where(mask, pred_label, ignore)
    mask = mask.squeeze()
    plabel = plabel.squeeze()
   
    
    return mask, plabel


def gene_plabel_prop(plab, P, prop, ece=None):  
   
    assert P.dim()==4
    _, C, H, W = P.shape
    total_cnt = float(H * W)

    P = F.softmax(P, dim=1)
    
    pred_label = P.argmax(dim=1).squeeze()
    
    pred_prob  = (P.max(dim=1)[0]).squeeze()
    plabel = torch.ones_like(pred_label, dtype=torch.long)
    plabel = plabel * 255
    thres_index = int(prop * total_cnt) - 1
    value, _ = torch.sort(pred_prob.view(1, -1), descending=True)
    thres = value[0, thres_index]
    select_mask = (pred_prob >= thres).cuda()
    plabel_f = torch.where(select_mask, pred_label, plabel)
    
    return select_mask, plabel_f


def mask_fusion(P, mask1, mask2, label=None):
    assert mask1.shape == mask2.shape
    mask = mask1 + mask2
    _, C, H, W = P.shape
    if label is None:
        P = F.softmax(P, dim=1)
        pred_label = P.argmax(dim=1).squeeze()
    else:
        pred_label = label.squeeze()

    plabel = torch.ones_like(pred_label, dtype=torch.long)
    plabel = plabel * 255
    plabel_f= torch.where(mask>0, pred_label, plabel)
    return mask, plabel_f


def stat_plabel(c, g, p, label, plabel, ignore_index=255, num_classes=19):
    h, w = label.shape
    h1, w1 = plabel.shape
    assert h == h1
    assert w == w1
    for i in range(num_classes):
        gt = torch.where(label==i, 200, i)
        gt_b = torch.where(gt==200, True, False)
        gt_i = gt_b.sum().item()
        mask = torch.where(plabel==i, 200, i+100)
        mask_b = torch.where(mask==200, True, False)
        mask_i = mask_b.sum().item()
        match = gt == mask
        num = match.sum().item()
        if i not in c.keys():
            c[i] = num
        else:
            c[i] += num
        if i not in g.keys():
            g[i] = gt_i
        else:
            g[i] += gt_i
        if i not in p.keys():
            p[i] = mask_i
        else:
            p[i] += mask_i
    return c, g, p


    