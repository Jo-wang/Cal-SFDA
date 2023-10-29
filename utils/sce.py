import torch
from torch import nn
import torch.nn.functional as F


class SCELoss(nn.Module):

    def __init__(self, num_classes=19, a=0.1, b=1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a  
        self.b = b

    def forward(self, pred, labels, weight=None):
        # CE
        ce = F.cross_entropy(pred, labels, weight=weight, ignore_index=255)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0) 
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        loss = self.a * ce + self.b * rce[labels!=255].mean()
        return loss


def one_hot(label, N):
    label = torch.where(label==255, N, label)
    num_c = N
    N = N + 1
    size = list(label.shape)
    label = label.view(-1)
    ones = torch.sparse.torch.eye(N).to(label.device).index_select(0, label)
    size.append(N)
    all = ones.view(*size)
    all_ = all[:,:,:,0:num_c].permute(0,3,1,2)
    return all_