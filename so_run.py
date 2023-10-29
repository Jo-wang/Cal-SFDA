import os, sys
import wandb
import torch
import torch.backends.cudnn as cudnn
from dataset.dataset import * 
from model.model_builder import init_ed_model, init_decoder, init_encoder

from model import *
from init_config import *
import sys
from trainer.source_only_trainer import Trainer
import numpy as np 
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    cudnn.enabled = True
    cudnn.benchmark = True

    config = init_config("config/so_config.yml", sys.argv)
  
    wandb.init(config=config, project='Cal-SFDA', name='specify your prefered exp name here')
    
    encoder = init_encoder(config)
    decoder = init_decoder(config)
    
    model = init_ed_model(encoder, decoder, config)

    trainer = Trainer(model, config)

    trainer.train()


if __name__ == "__main__":
    main()
