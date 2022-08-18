''' 
Borrow from t2t-vit(https://github.com/yitu-opensource/T2T-ViT
- load_for_transfer_learning: load pretrained paramters to model in transfer learning
- get_mean_and_std: calculate the mean and std value of dataset.
- msr_init: net parameter initialization.
- progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import torch

import torch.nn as nn
import torch.nn.init as init
import logging
import os
from collections import OrderedDict

_logger = logging.getLogger(__name__)

def convert_qkv(origin_dict):
    model_dict = OrderedDict()
    for k,v in origin_dict.items():
        if 'qkv' in k :
            #bias
            if v.dim() == 1 :
                dim = v.shape[-1] // 3
                tmp_bias = v
                b_q, b_k, b_v = [None] *3 if tmp_bias is None else [tmp_bias[i*dim:(i+1)*dim] for i in range(3)]
                print('bias q k v', b_q.shape, b_k.shape, b_v.shape)
                model_dict[k.replace('qkv', 'query')] = b_q
                model_dict[k.replace('qkv', 'key')] = b_k
                model_dict[k.replace('qkv', 'value')] = b_v
            else :
                dim = v.shape[-1]
                tmp_weight = v
                w_q, w_k, w_v = [tmp_weight[i*dim:(i+1)*dim] for i in range(3)]
                model_dict[k.replace('qkv', 'query')] = w_q
                model_dict[k.replace('qkv', 'key')] = w_k
                model_dict[k.replace('qkv', 'value')] = w_v
        else :
            model_dict[k] = v

    return model_dict

def ada_load_state_dict(checkpoint_path, model, use_qkv=False, strict=True):
    if not os.path.isfile(checkpoint_path) :
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict_ema' in checkpoint :
        checkpoint = checkpoint['state_dict_ema']
    elif 'state_dict' in checkpoint :
        checkpoint = checkpoint['state_dict']
    elif 'model' in checkpoint :
        checkpoint = checkpoint['model'] # load deit type model

    if not use_qkv :
        checkpoint = convert_qkv(checkpoint)

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        # strip `module.` prefix
        name = k[7:] if k.startswith('module') else k
        new_state_dict[name] = v
    checkpoint = new_state_dict
    
    info = model.load_state_dict(checkpoint, strict=strict)
    if not strict :
        print('state dict load info', info)


def deit_load_state_dict(checkpoint_path, model, strict=True):
    if not os.path.isfile(checkpoint_path) :
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint = checkpoint['model']

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        # strip `module.` prefix
        name = k[7:] if k.startswith('module') else k
        new_state_dict[name] = v
    checkpoint = new_state_dict
    
    info = model.load_state_dict(checkpoint, strict=strict)
    if not strict :
        print('state dict load info', info)


def resize_pos_embed(posemb, posemb_new): # example: 224:(14x14+1)-> 384: (24x24+1)
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]  # posemb_tok is for cls token, posemb_grid for the following tokens
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))     # 14
    gs_new = int(math.sqrt(ntok_new))             # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)  # [1, 196, dim]->[1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic') # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)   # [1, dim, 24, 24] -> [1, 24*24, dim]
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)   # [1, 24*24+1, dim]
    return posemb

def load_state_dict(checkpoint_path, model, use_ema=False, num_classes=1000, del_posemb=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        if num_classes != 1000:
            # completely discard fully connected for all other differences between pretrained and created model
            del state_dict['head' + '.weight']
            del state_dict['head' + '.bias']

        if del_posemb==True:
            del state_dict['pos_embed']

        old_posemb = state_dict['pos_embed']
        if model.pos_embed.shape != old_posemb.shape:  # need resize the position embedding by interpolate
            new_posemb = resize_pos_embed(old_posemb, model.pos_embed)
            state_dict['pos_embed'] = new_posemb

        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()



def load_for_transfer_learning(model, checkpoint_path, use_ema=False, strict=True, num_classes=1000):
    state_dict = load_state_dict(checkpoint_path, use_ema, num_classes)
    model.load_state_dict(state_dict, strict=strict)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width = 100  # since we're not using transfer_learning.py or progress bar, just set a constant to make no-terminal job submission happy

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
