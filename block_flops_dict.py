import torch
import fvcore
from fvcore.nn import FlopCountAnalysis
import argparse
import models
from models.ada_transformer_block import StepAdaBlock
from timm.models import create_model

parser = argparse.ArgumentParser(description='T2T-ViT Training and Evaluating')
parser.add_argument('--model', default='t2t-19', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--ada-layer', action='store_true', default=False, dest='ada_layer',
                    help='')
parser.add_argument('--ada-head', action='store_true', default=False, dest='ada_head',
                    help='')

parser.add_argument('--ada-token', action='store_true', default=False, dest='ada_token',
                    help='ada-token on self-attn')
parser.add_argument('--ada-token-with-mlp', action='store_true', default=False, dest='ada_token_with_mlp',
                    help='ada-token on both self-attn and ffn')
parser.add_argument('--keep-layers', type=int, default=1,
                    help='use layers to make selection decision')

def _parse_args():

    args = parser.parse_args()
    if args.ada_token_with_mlp :
        args.ada_token = True

    return args

def get_flops_dict(dim, num_heads, mlp_ratio, debug=False, **kwargs):
    block = StepAdaBlock(dim, num_heads, mlp_ratio, **kwargs)
    inputs = torch.rand((1,197,dim))
    
    num_tokens = 197
    num_layers = 2

    block.apply(lambda x : setattr(x, 'count_flops', True))
    flops_dict = torch.zeros(num_heads+1, num_tokens+1, 2, 2)
    for t in range(1, num_tokens+1) :
        for h in range(num_heads+1) :
            block.apply(lambda x : setattr(x, 'h_ratio', h/num_heads))
            block.apply(lambda x : setattr(x, 't_ratio', (t)/197))

            if debug :
                block.apply(lambda x : setattr(x, 'h_ratio', 5/7))
                block.apply(lambda x : setattr(x, 't_ratio', (197)/197))
                block.apply(lambda x : setattr(x, 'l_ratio', [1,1]))
                flops = FlopCountAnalysis(block, inputs).total() / (1000**3)
                print('flops', flops)
                exit()

            def fill_dict(l_select):
                block.apply(lambda x : setattr(x, 'l_ratio', l_select))
                
                xx = block(inputs)
                
                # flops = FlopCountAnalysis(block, inputs).total() / (1000**3)
                # print('flops', h, t, l_select, flops)
                # flops_dict[h,t,l_select[0],l_select[1]] = flops
            fill_dict([0,0])
            fill_dict([0,1])
            fill_dict([1,0])
            fill_dict([1,1])
    return flops_dict

def select_flops(flops_dict, head_select, layer_select, token_select, block_num, base_flops=0.33):
    '''
    head_select : None or tensor (ada_block_h, h_num)
    layer_select : None or tensor (ada_block_l, 2)
    token_select : None or tensor (ada_block_t, 197)
    '''

    h, t, _, _ = flops_dict.shape
    h = h-1
    t = t-1
    if head_select is None :
        head_select = [h] * block_num
    else :
        ada_h = head_select.shape[0]
        head_select = [h] * (block_num - ada_h) + head_select.sum(-1).int().tolist()
    if layer_select is None :
        layer_select = [[1,1]] * block_num
    else :
        ada_l = layer_select.shape[0]
        layer_select = [[1,1]] * (block_num-ada_l) + layer_select.int().tolist()
    if token_select is None :
        token_select = [t] * block_num
    else :
        ada_t = token_select.shape[0]
        token_select = [t] * (block_num - ada_t) + token_select.sum(-1).int().tolist()

    flops = base_flops
    for h, t, l in zip(head_select, token_select, layer_select) :
        flops += flops_dict[h, t, l[0], l[1]]
    return flops

def batch_select_flops(bs, flops_dict, head_select, layer_select, token_select, block_num=19, base_flops=0.33):
    def batch_select(x):
        return x if x is not None else [None] * bs
    head_select = batch_select(head_select)
    layer_select = batch_select(layer_select)
    token_select = batch_select(token_select)
    
    batch_flops = []
    for h, l, t in zip(head_select, layer_select, token_select):
        batch_flops.append(select_flops(flops_dict, h, l, t, block_num, base_flops))
    
    return torch.tensor(batch_flops)