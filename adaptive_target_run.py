import wandb
import numpy as np
from model.DeeplabV2 import *
import random
from trainer.adaptive_target_trainer import Trainer

import sys
from init_config import init_config
from model import *
from model.model_builder import init_decoder, init_encoder, init_ac_model, init_adaptive_tar

import torch.backends.cudnn as cudnn
import torch
import os
import sys
import datetime

print(sys.path)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    config = init_config("config/adaptive_target_config.yml", sys.argv)

    wandb.init(config=config, project='', name='')
    if config.source == 'synthia':
        config.num_classes = 16
    else:
        config.num_classes = 19

    encoder = init_encoder(config)
    decoder = init_decoder(config)
    ac = init_ac_model(config)
    model = init_adaptive_tar(encoder, decoder, ac, config)

    trainer = Trainer(model, config)
    
    trainer.train()


if __name__ == "__main__":
    main()
