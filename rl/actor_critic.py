import torch.nn as nn
import torch.nn.functional as F

class Value(nn.Module): 
    def __init__(self, bn_1=2048, output_size=1):  
        super(Value, self).__init__()
        self.output_size = output_size
        self.bn1 = nn.BatchNorm2d(bn_1, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear2 = nn.Linear(2048, output_size)
        self.act = nn.Sigmoid()
                    
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
    def forward(self, x): 
        x = self.bn1(x)
        x = self.conv1(x)
        x1 = F.interpolate(x, size=(4, 4), mode='bilinear') 
        x2 = x1.flatten(start_dim=1)
        x2 = self.linear2(x2)
        x3 = self.act(x2)
        return x3

def value_net():
    return Value(bn_1=2048, output_size=1)





