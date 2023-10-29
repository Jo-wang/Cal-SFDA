import torch.nn as nn
import sys
import copy

sys.path.append('../')

class AC_model(nn.Module):
    def __init__(self, encoder, decoder, ac):
        super(AC_model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.value_net = ac
        
    def forward(self, x, reverse=False):
        N, C, H, W = x.shape   
        x_inter = self.encoder(x) 
        x_cla = self.decoder(x_inter, H, W, reverse)

        
        x_rl = self.value_net(x_inter)
        return x_cla, x_rl, x_inter
    
    
def model_all(encoder, decoder, ac):
    model = AC_model(encoder, decoder, ac)
    return model


class Fea_model(nn.Module):
    def __init__(self, encoder):
        super(Fea_model, self).__init__()
        self.encoder = encoder
        
       
        
    def forward(self, x):
        N, C, H, W = x.shape   
        x_inter = self.encoder(x)  
        
        return x_inter
    
    
def model_fea(encoder):
    model = Fea_model(encoder)
    return model


class Cla_model(nn.Module):
    def __init__(self, decoder):
        super(Cla_model, self).__init__()
        self.decoder = decoder
        
       
        
    def forward(self, x_inter, H, W):
            
        x_cla = self.decoder(x_inter, H, W)
        return x_cla
    
    
def model_cla(decoder):
    model = Cla_model(decoder)
    return model

class ED_model(nn.Module):
    def __init__(self, encoder, decoder):
        super(ED_model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x, reverse=False):
        N, C, H, W = x.shape   
        x_inter = self.encoder(x)  
        x_cla = self.decoder(x_inter, H, W, reverse)
        return x_cla
    
    
def model_ED(encoder, decoder):
    model = ED_model(encoder, decoder)
    return model