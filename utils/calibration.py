import torch
from torch import nn
from torch.nn import functional as F
from scipy import optimize

class ECELoss(nn.Module):
    def __init__(self, n_bins=10, LOGIT=True):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1) 
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.LOGIT = LOGIT

    def forward(self, logits, labels, train):
        if self.LOGIT and train:
            softmaxes = F.softmax(logits, dim=1)
        elif self.LOGIT and not train:
            softmaxes = F.softmax(logits, dim=0)
        else:
            softmaxes = logits
        if train:
            confidences, predictions = torch.max(softmaxes, 1)  
        else:
            _, predictions = torch.max(softmaxes, 0)
            t = 0.00001   
            confidences = t*torch.logsumexp(softmaxes/t, dim=0)
    
        correctness = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = correctness[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean().float()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece
    

