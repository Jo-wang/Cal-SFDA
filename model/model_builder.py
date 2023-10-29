from collections import OrderedDict
from .sync_batchnorm import convert_model
import torch
from .DeeplabV2 import *
from .UNet import *
from rl.actor_critic import Value, value_net, value_unet
from rl.model import model_all, model_cla, model_fea, model_ED
from rl.model_unet import model_unet_all
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def freeze_bn(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            for i in module.parameters():
                i.requires_grad = False


def release_bn(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            for i in module.parameters():
                i.requires_grad = True


def init_model(cfg):

    model = Res_Deeplab(num_classes=cfg.num_classes).cuda()

    if cfg.fixbn:
        freeze_bn(model)
    else:
        release_bn(model)

    if cfg.model == 'deeplab' and cfg.init_weight != 'None':
        params = torch.load(cfg.init_weight)
        print('Model restored with weights from : {}'.format(cfg.init_weight))
        if 'init-' in cfg.init_weight and cfg.model == 'deeplab':
            new_params = model.state_dict().copy()
            for i in params:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = params[i]
            model.load_state_dict(new_params, strict=True)

        else:
            new_params = model.state_dict().copy()
            for i in params:
                if 'module' in i:
                    i_ = i.replace('module.', '')
                    new_params[i_] = params[i]
                else:
                    new_params[i] = params[i]

            model.load_state_dict(new_params, strict=True)




    if cfg.restore_from != 'None':
        params = torch.load(cfg.restore_from)
        new_params = OrderedDict()
        for k, v in params.items():
            if 'module' in k:
                k_ = k.replace('module.', '')
                new_params[k_] = v
            else:
                new_params[k] = params[k]
        model.load_state_dict(new_params)
        print('Model initialize with weights from : {}'.format(
            cfg.restore_from))

    if cfg.multigpu:  
        model = convert_model(model)
        model = nn.DataParallel(model, device_ids=[0, 1])

    if cfg.ece_train:
        model.train().cuda()
        print('Mode --> Train')
    else:
        model.eval().cuda()
        print('Mode --> Eval')

    return model


def match_encoder(new_params):
    encoder = OrderedDict()
    for k, v in new_params.items():

        if 'layer5' not in k and 'layer6' not in k:
            encoder[k] = v
    return encoder


def match_decoder(new_params):
    decoder = OrderedDict()
    for k, v in new_params.items():

        if 'layer5' in k or 'layer6' in k:
            decoder[k] = v
    return decoder


def init_encoder(cfg):
    model = Res_Deeplab_encoder(num_classes=cfg.num_classes).cuda()

    if cfg.fixbn:
        freeze_bn(model)
    else:
        release_bn(model)

    if cfg.init_weight != 'None':
        params = torch.load(cfg.init_weight)
        print('Model restored with weights from : {}'.format(cfg.init_weight))
        if 'init-' in cfg.init_weight and cfg.model == 'deeplab':
            new_params = model.state_dict().copy()
            for i in params:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = params[i]

            model.load_state_dict(match_encoder(new_params), strict=True)

        else:
            new_params = model.state_dict().copy()
            for i in params:
                if 'module' in i:
                    i_ = i.replace('module.', '')
                    new_params[i_] = params[i]
                else:
                    new_params[i] = params[i]

    if cfg.restore_from != 'None':
        params = torch.load(cfg.restore_from)
        new_params = OrderedDict()
        for k, v in params.items():
            if 'module' in k:
                k_ = k.replace('module.', '')
                new_params[k_.replace('encoder.', '')] = v
            else:
                new_params[k.replace('encoder.', '')] = v
        if cfg.encoder:
            new_params = match_encoder(new_params)
        model.load_state_dict(new_params)
        print('Model initialize with weights from : {}'.format(
            cfg.restore_from))

    if cfg.encoder_train:
        model.train() 
        print('Mode --> Train')
    else:
        model.eval().cuda()
        print('Mode --> Eval')

    return model

def init_ac_model(cfg):
    model = value_net().cuda()
    if cfg.train:
        model.train()  
        print('Mode --> Train')
    else:
        model.eval().cuda()
        print('Mode --> Eval')

    return model


def init_decoder(cfg):
    model = Res_Deeplab_decoder(num_classes=cfg.num_classes).cuda()

    if cfg.fixbn:
        freeze_bn(model)
    else:
        release_bn(model)

    if cfg.model == 'deeplab' and cfg.init_weight != 'None':
        params = torch.load(cfg.init_weight)
        print('Decoder Model restored with weights from : {}'.format(
            cfg.init_weight))
        if 'init-' in cfg.init_weight and cfg.model == 'deeplab':
            new_params = model.state_dict().copy()
            for i in params:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = params[i]
            if cfg.decoder:
                decoder_params = OrderedDict()
                for k, v in new_params.items():
                    if k[5] == '5' or k[5] == '6':
                        decoder_params[k] = v

            model.load_state_dict(decoder_params, strict=True)
        else:
            new_params = model.state_dict().copy()
            for i in params:
                if 'module' in i:
                    i_ = i.replace('module.', '')
                    new_params[i_] = params[i]
                else:
                    new_params[i] = params[i]
            model.load_state_dict(new_params, strict=True)

    if cfg.restore_from != 'None':
        params = torch.load(cfg.restore_from)
        new_params = OrderedDict()
        for k, v in params.items():
            if 'module' in k:
                k_ = k.replace('module.', '')
                new_params[k_.replace('decoder.', '')] = v
            else:
                new_params[k.replace('decoder.', '')] = params[k]
        new_params = match_decoder(new_params)
        model.load_state_dict(new_params)
        print('Model initialize with weights from : {}'.format(
            cfg.restore_from))

    if cfg.decoder_train:
        model.train() 
        print('Mode --> Train')
    else:
        model.eval().cuda()
        print('Mode --> Eval')

    return model


def init_whole_model(encoder, decoder, ac, cfg):
    model = model_all(encoder, decoder, ac).cuda()
    if cfg.rl_restore_from != 'None':
        params = torch.load(cfg.rl_restore_from)
        new_params = OrderedDict()
        for k, v in params.items():
            if 'module' in k:
                k_ = k.replace('module.', '')
                new_params[k_] = v
            else:
                new_params[k] = params[k]
        model.load_state_dict(new_params)
        print('Model initialize with weights from : {}'.format(
            cfg.rl_restore_from))

    if cfg.fixbn:
        freeze_bn(model)
    else:
        release_bn(model)

    if cfg.multigpu:
        model = convert_model(model)
        model = nn.DataParallel(model)

    if cfg.train:
        model.train().cuda()
        print('Mode --> Train')
    else:
        model.eval().cuda()
        print('Mode --> Eval')

    return model


def init_adaptive_tar(encoder, decoder, ac, cfg):

    assert cfg.rl_restore_from != 'None'
    params = torch.load(cfg.rl_restore_from)

    model = model_all(encoder, decoder, ac).cuda()

    new_params = OrderedDict()
    for k, v in params.items():
        if 'module' in k:
            k = k.replace('module.', '')
            new_params[k] = v
        else:
            new_params[k] = params[k]
    model.load_state_dict(new_params)

    if cfg.fixbn:
        freeze_bn(model)
    else:
        release_bn(model)

    if cfg.multigpu:
        model = convert_model(model)
        model = nn.DataParallel(model)

    if cfg.train:
        model.train().cuda()
        print('Mode --> Train')
    else:
        model.eval().cuda()
        print('Mode --> Eval')

    return model


def init_fea(encoder, cfg):
    assert cfg.restore_from != 'None'
    params = torch.load(cfg.restore_from)

    model = model_fea(encoder).cuda()

    new_params = OrderedDict()
    for k, v in params.items():
        if 'module' in k:
            k = k.replace('module.', '')
            new_params[k] = v
        else:
            new_params[k] = params[k]
    model.load_state_dict(new_params)

    if cfg.fixbn:
        freeze_bn(model)
    else:
        release_bn(model)

    if cfg.multigpu:
        model = convert_model(model)
        model = nn.DataParallel(model)

    if cfg.train:
        model.train().cuda()
        print('Mode --> Train')
    else:
        model.eval().cuda()
        print('Mode --> Eval')

    return model


def init_cla(decoder, cfg):
    assert cfg.restore_from != 'None'
    params = torch.load(cfg.restore_from)

    model = model_cla(decoder).cuda()

    new_params = OrderedDict()
    for k, v in params.items():
        if 'module' in k:
            k = k.replace('module.', '')
            new_params[k] = v
        else:
            new_params[k] = params[k]
    model.load_state_dict(new_params)

    if cfg.fixbn:
        freeze_bn(model)
    else:
        release_bn(model)

    if cfg.multigpu:
        model = convert_model(model)
        model = nn.DataParallel(model)

    if cfg.train:
        model.train().cuda()
        print('Mode --> Train')
    else:
        model.eval().cuda()
        print('Mode --> Eval')

    return model

def load_pse_init_model(cfg):
    model = Res_Deeplab(num_classes=cfg.num_classes).cuda()

    if cfg.fixbn:
        freeze_bn(model)
    else:
        release_bn(model)

    if cfg.model == 'deeplab' and cfg.init_weight != 'None':
        params = torch.load(cfg.init_weight)
        print('Model restored with weights from : {}'.format(cfg.init_weight))
        if 'init-' in cfg.init_weight and cfg.model == 'deeplab':
            new_params = model.state_dict().copy()
            for i in params:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = params[i]
            model.load_state_dict(new_params, strict=True)

        else:
            new_params = model.state_dict().copy()
            for i in params:
                if 'module' in i:
                    i_ = i.replace('module.', '')
                    new_params[i_] = params[i]
                else:
                    new_params[i] = params[i]

            model.load_state_dict(new_params, strict=True)

    if cfg.restore_from != 'None':
        params = torch.load(cfg.restore_from)
        new_params = OrderedDict()
        for k, v in params['encoder'].items():
            if 'module' in k:
                k_ = k.replace('module.', '')
                new_params[k_] = v
            else:
                new_params[k] = params['encoder'][k]
        for k, v in params['decoder'].items():
            if 'module' in k:
                k_ = k.replace('module.', '')
                new_params[k_] = v
            else:
                new_params[k] = params['decoder'][k]
        model.load_state_dict(new_params)
        print('Model initialize with weights from : {}'.format(
            cfg.restore_from))

    if cfg.multigpu:  
        model = convert_model(model)
        model = nn.DataParallel(model, device_ids=[0, 1])

    if cfg.train:
        model.train().cuda()
        print('Mode --> Train')
    else:
        model.eval().cuda()
        print('Mode --> Eval')

    return model

def init_ed_model(encoder, decoder, cfg):
    model = model_ED(encoder, decoder).cuda()

    if cfg.fixbn:
        freeze_bn(model)
    else:
        release_bn(model)

    if cfg.multigpu:
        model = convert_model(model)
        model = nn.DataParallel(model)

    if cfg.train:
        model.train().cuda()
        print('Mode --> Train')
    else:
        model.eval().cuda()
        print('Mode --> Eval')

    return model