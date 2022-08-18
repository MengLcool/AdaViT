import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
import datetime
from time import gmtime, strftime
import models
from utils import ada_load_state_dict
from models.losses import AdaHeadLoss, AdaLoss, TeacherLoss

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import Dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import load_checkpoint, create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

import pdb
from saver import MyCheckpointSaver

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='T2T-ViT Training and Evaluating')

parser.add_argument('--head-ratio', type=float, default=2.0,
                    help='')
parser.add_argument('--layer-ratio', type=float, default=2.0,
                    help='')
parser.add_argument('--attn-ratio', type=float, default=0.,
                    help='')
parser.add_argument('--hidden-ratio', type=float, default=0.,
                    help='')
parser.add_argument('--pred-ratio', type=float, default=0.,
                    help='')
parser.add_argument('--head-target-ratio', type=float, default=0.5,
                    help='')
parser.add_argument('--layer-target-ratio', type=float, default=0.5,
                    help='')
parser.add_argument('--head-diverse-ratio', type=float, default=0.,
                    help='')
parser.add_argument('--layer-diverse-ratio', type=float, default=0.,
                    help='')
parser.add_argument('--head-select-tau', type=float, default=5.,
                    help='')
parser.add_argument('--layer-select-tau', type=float, default=5.,
                    help='')
parser.add_argument('--token-select-tau', type=float, default=5.,
                    help='')
parser.add_argument('--head-entropy-weight', type=float, default=0.,
                    help='')
parser.add_argument('--layer-entropy-weight', type=float, default=0.,
                    help='')
parser.add_argument('--head-minimal-weight', type=float, default=0.,
                    help='')
parser.add_argument('--head-minimal', type=float, default=0.,
                    help='')
parser.add_argument('--layer-minimal-weight', type=float, default=0.,
                    help='')
parser.add_argument('--layer-minimal', type=float, default=0.,
                    help='')
parser.add_argument('--token-ratio', type=float, default=2.,
                    help='')
parser.add_argument('--token-target-ratio', type=float, default=0.5,
                    help='')
parser.add_argument('--token-minimal', type=float, default=0.,
                    help='')
parser.add_argument('--token-minimal-weight', type=float, default=0.,
                    help='')

parser.add_argument('--inner-loop', type=int, default=-1,
                    help='')

# Dataset / Model parameters
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--norm-policy', action='store_true', default=False, dest='norm_policy',
                    help='')
parser.add_argument('--keep-layers', type=int, default=1,
                    help='use layers to make selection decision')
parser.add_argument('--ada-layer', action='store_true', default=False, dest='ada_layer',
                    help='')
parser.add_argument('--ada-block', action='store_true', default=False, dest='ada_block',
                    help='')
parser.add_argument('--ada-head', action='store_true', default=False, dest='ada_head',
                    help='')
parser.add_argument('--ada-head-v2', action='store_true', default=False, dest='ada_head_v2',
                    help='')
parser.add_argument('--dyna-data', action='store_true', default=False, dest='dyna_data',
                    help='')
parser.add_argument('--ada-head-attn', action='store_true', default=False, dest='ada_head_attn',
                    help='')
parser.add_argument('--head-slowfast', action='store_true', default=False, dest='head_slowfast',
                    help='')

parser.add_argument('--flops-dict', type=str, default='', dest='flops_dict',
                    help='')
parser.add_argument('--ada-token', action='store_true', default=False, dest='ada_token',
                    help='ada-token on self-attn')
parser.add_argument('--ada-token-nonstep', action='store_true', default=False, dest='ada_token_nonstep',
                    help='using nonstep option for token selection, i.e. generate policies for all layers at once at the beginning')
parser.add_argument('--ada-token-with-mlp', action='store_true', default=False, dest='ada_token_with_mlp',
                    help='ada-token on both self-attn and ffn')
parser.add_argument('--ada-token-start-layer', type=int, default=0)
parser.add_argument('--ada-token-pre-softmax', action='store_true', default=True, dest='ada_token_pre_softmax')
parser.add_argument('--ada-token-post-softmax', action='store_false', default=True, dest='ada_token_pre_softmax')
parser.add_argument('--ada-token-detach-attn', action='store_true', default=True, dest='ada_token_detach_attn')
parser.add_argument('--ada-token-no-detach-attn', action='store_false', default=True, dest='ada_token_detach_attn')
parser.add_argument('--ada-token-detach-attn-at-mlp', action='store_true', default=False, dest='ada_token_detach_attn_at_mlp', 
                    help='whether detaching attn_policy at MLP, when using dynamic token selection (which overrides --ada-token-detach-attn at MLP)')

parser.add_argument('--no-head-select-bias', action='store_false', default=True, dest='head_select_bias',
                    help='')
parser.add_argument('--no-layer-select-bias', action='store_false', default=True, dest='layer_select_bias',
                    help='')
parser.add_argument('--model', default='T2t_vit_14', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--freeze-bn', action='store_true', default=False,
                    help='freeze bn in training')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--pretrain-path', default='', type=str,
                    help='Load pretrain file path')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--eval_checkpoint', default='', type=str, metavar='PATH',
                    help='path to eval checkpoint (default: none)')
parser.add_argument('--use-full-head', action='store_true', default=False,
                    help='use full model param')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.005 for adamw)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
parser.add_argument('--ada-lr-scaling', action='store_true', default=False,
                    help='rescale the lr of ada subnetworks')            
parser.add_argument('--ada-token-lr-scale', type=float, default=1.0, help='')
parser.add_argument('--ada-layer-lr-scale', type=float, default=1.0, help='')
parser.add_argument('--ada-head-lr-scale', type=float, default=1.0, help='')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.0)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=True,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--no-model-ema', action='store_false', dest='model_ema',
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.99996,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

# Random Baseline settings
parser.add_argument('--random-policy', action='store_true', default=False, dest='random_policy')
parser.add_argument('--random-head', action='store_true', default=False, dest='random_head')
parser.add_argument('--random-layer', action='store_true', default=False, dest='random_layer')
parser.add_argument('--random-token', action='store_true', default=False, dest='random_token')
parser.add_argument('--eval_random_baseline', action='store_true', default=False,
                    help='if True, evaluate random baselines given certain computational budget')
parser.add_argument('--eval_random_layer', action='store_true', default=False,
                    help='if True, test randomly generated policies for layer selection')
parser.add_argument('--eval_random_head', action='store_true', default=False,
                    help='if True, test randomly generated policies for head selection')
parser.add_argument('--eval_random_token', action='store_true', default=False,
                    help='if True, test randomly generated policies for token selection')
parser.add_argument('--eval_random_layer_ratio', type=float, default=1.0,
                    help='ratio of kept layers in random policies')
parser.add_argument('--eval_random_head_ratio', type=float, default=1.0,
                    help='ratio of kept layers in random policies')
parser.add_argument('--eval_random_token_ratio', type=float, default=1.0,
                    help='ratio of kept layers in random policies')
parser.add_argument('--dev', action='store_true', default=False,
                    help='skip some time-consuming steps when developing, e.g. loading full training set')
parser.add_argument('--print-head-option', action='store_true')
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    if args.ada_block :
        args.ada_layer = True
    if args.ada_head_attn :
        args.ada_head = True
    if args.ada_token_with_mlp :
        args.ada_token = True
    if args.ada_token_nonstep:
        args.ada_token = True
    if args.ada_head_v2 :
        args.ada_head = True

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            _logger.warning(
                'Using more than one GPU per process in distributed mode is not allowed.Setting num_gpu to 1.')
            args.num_gpu = 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    if args.distributed:
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on %d GPUs.' % args.num_gpu)

    torch.manual_seed(args.seed + args.rank)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path=args.initial_checkpoint,
        img_size=args.img_size,
        keep_layers=args.keep_layers,
        head_select_bias=args.head_select_bias,
        layer_select_bias=args.layer_select_bias,
        norm_policy = args.norm_policy,
        ada_layer = args.ada_layer,
        ada_block = args.ada_block,
        ada_head = args.ada_head,
        ada_head_v2 = args.ada_head_v2,
        dyna_data = args.dyna_data,
        ada_token = args.ada_token,
        ada_token_nonstep = args.ada_token_nonstep,
        ada_token_with_mlp = args.ada_token_with_mlp,
        ada_head_attn = args.ada_head_attn,
        head_slowfast = args.head_slowfast,
        ada_token_start_layer = args.ada_token_start_layer,
        ada_token_pre_softmax = args.ada_token_pre_softmax,
        ada_token_detach_attn = args.ada_token_detach_attn,
        ada_token_detach_attn_at_mlp = args.ada_token_detach_attn_at_mlp,
        layer_select_tau = args.layer_select_tau,
        head_select_tau = args.head_select_tau,
        token_select_tau = args.token_select_tau,
        )

    def set_model_attr(model, name, value):
        if hasattr(model, name) :
            setattr(model, name, value)
    if args.random_policy :
        model.apply(lambda x : set_model_attr(x, 'random_policy', True))
        model.apply(lambda x : set_model_attr(x, 'random_layer_ratio', args.layer_target_ratio))
        model.apply(lambda x : set_model_attr(x, 'random_head_ratio', args.head_target_ratio))
        model.apply(lambda x : set_model_attr(x, 'random_token_ratio', args.token_target_ratio))

    if args.random_head :
        model.apply(lambda x : set_model_attr(x, 'random_head', True))
        model.apply(lambda x : set_model_attr(x, 'random_head_ratio', args.head_target_ratio))
    if args.random_layer :
        model.apply(lambda x : set_model_attr(x, 'random_layer', True))
        model.apply(lambda x : set_model_attr(x, 'random_layer_ratio', args.layer_target_ratio))
    if args.random_token :
        model.apply(lambda x : set_model_attr(x, 'random_token', True))
        model.apply(lambda x : set_model_attr(x, 'random_token_ratio', args.token_target_ratio))

    if args.pretrain_path :
        _logger.info('load pretrain model {}'.format(args.pretrain_path))
        # model.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'), strict=False)
        ada_load_state_dict(args.pretrain_path, model, use_qkv=False, strict=False)

    if args.local_rank == 0:
        _logger.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    use_amp = None
    if args.amp:
        # for backwards compat, `--amp` arg tries apex before native amp
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    if args.num_gpu > 1:
        if use_amp == 'apex':
            _logger.warning(
                'Apex AMP does not work well with nn.DataParallel, disabling. Use DDP or Torch AMP.')
            use_amp = None
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        assert not args.channels_last, "Channels last not supported with DP, use DDP."
    else:
        model.cuda()
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)

    if not args.ada_lr_scaling:
        optimizer = create_optimizer(args, model)
    else:
        ada_params_names = ['token_select', 'head_select', 'layer_select']
        no_wd_params_names = list(model.no_weight_decay())
        model_params = [kv for kv in model.named_parameters() if kv[1].requires_grad]
        # ada_params_name_check = [kv[0] for kv in model.named_parameters() if any(sub in kv[0] for sub in ada_params_names)]  # just for sanity check
        ada_params = [kv for kv in model_params if any(sub in kv[0] for sub in ada_params_names)]
        base_params = [kv for kv in model_params if not any(sub in kv[0] for sub in ada_params_names)]

        base_params_wd, base_params_no_wd = [], []
        ada_params_token_wd, ada_params_token_no_wd, ada_params_head_wd, ada_params_head_no_wd, ada_params_layer_wd, ada_params_layer_no_wd = [], [], [], [], [], [] 

        for name, param in base_params:
            if len(param.shape) == 1 or name.endswith(".bias") or name in no_wd_params_names:
                base_params_no_wd.append(param)
            else:
                base_params_wd.append(param)
        for name, param in ada_params:
            if 'token_select' in name:
                if len(param.shape) == 1 or name.endswith(".bias") or name in no_wd_params_names:
                    ada_params_token_no_wd.append(param)
                else:
                    ada_params_token_wd.append(param)
            elif 'head_select' in name:
                if len(param.shape) == 1 or name.endswith(".bias") or name in no_wd_params_names:
                    ada_params_head_no_wd.append(param)
                else:
                    ada_params_head_wd.append(param)
            elif 'layer_select' in name:
                if len(param.shape) == 1 or name.endswith(".bias") or name in no_wd_params_names:
                    ada_params_layer_no_wd.append(param)
                else:
                    ada_params_layer_wd.append(param) 
          
        all_params = [
            {'params': ada_params_token_wd, 'lr': args.lr * args.ada_token_lr_scale, 'weight_decay': args.weight_decay},
            {'params': ada_params_token_no_wd, 'lr': args.lr * args.ada_token_lr_scale, 'weight_decay': 0.},
            {'params': ada_params_head_wd, 'lr': args.lr * args.ada_head_lr_scale, 'weight_decay': args.weight_decay},
            {'params': ada_params_head_no_wd, 'lr': args.lr * args.ada_head_lr_scale, 'weight_decay': 0.},
            {'params': ada_params_layer_wd, 'lr': args.lr * args.ada_layer_lr_scale, 'weight_decay': args.weight_decay},
            {'params': ada_params_layer_no_wd, 'lr': args.lr * args.ada_layer_lr_scale, 'weight_decay': 0.},
            {'params': base_params_wd, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': base_params_no_wd, 'lr': args.lr, 'weight_decay': 0.},
        ]
        optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    if args.distributed:
        if args.sync_bn:
            assert not args.split_bn
            try:
                if has_apex and use_amp != 'native':
                    # Apex SyncBN preferred unless native amp is activated
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    _logger.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception as e:
                _logger.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    train_dir = os.path.join(args.data, 'train' if not args.dev else 'val')
    if not os.path.exists(train_dir):
        _logger.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    dataset_train = Dataset(train_dir)

    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    print('mixup activte', mixup_active)
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    eval_dir = os.path.join(args.data, 'val')
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(args.data, 'validation')
        if not os.path.isdir(eval_dir):
            _logger.error('Validation folder does not exist at: {}'.format(eval_dir))
            exit(1)
    dataset_eval = Dataset(eval_dir)

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    # train_loss_fn = AdaHeadLoss(train_loss_fn, target_ratio=args.head_target_ratio,head_loss_ratio=args.head_ratio)
    train_loss_fn = AdaLoss(train_loss_fn, 
                                 head_target_ratio=args.head_target_ratio, head_loss_ratio=args.head_ratio, 
                                 layer_target_ratio=args.layer_target_ratio, layer_loss_ratio=args.layer_ratio,
                                 head_diverse_ratio=args.head_diverse_ratio, layer_diverse_ratio=args.layer_diverse_ratio,
                                 head_entropy_weight=args.head_entropy_weight, layer_entropy_weight=args.layer_entropy_weight,
                                 head_minimal_weight=args.head_minimal_weight, head_minimal=args.head_minimal,
                                 layer_minimal_weight=args.layer_minimal_weight, layer_minimal=args.layer_minimal,
                                 token_target_ratio=args.token_target_ratio, token_loss_ratio=args.token_ratio, token_minimal=args.token_minimal, token_minimal_weight=args.token_minimal_weight)
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None

    flops_dict = None
    if args.flops_dict :
        flops_dict = torch.load(args.flops_dict)
    if args.eval_random_baseline:
        assert args.eval_checkpoint, "Please provide path to the checkpoint when evaluating baselines."
        ada_load_state_dict(args.eval_checkpoint, model, use_qkv=False, strict=False)
        # set random policy configuration
        model.random_layer, model.random_head, model.random_token = args.eval_random_layer, args.eval_random_head, args.eval_random_token
        model.random_layer_ratio, model.random_head_ratio, model.random_token_ratio = \
                                                        args.eval_random_layer_ratio, args.eval_random_head_ratio, args.eval_random_token_ratio
        val_metrics = validate(model, loader_eval, validate_loss_fn, args, print_head_option=args.print_head_option, flops_dict=flops_dict)
        print(f"Top-1 accuracy of the model is: {val_metrics['top1']:.1f}%")
        return

    if args.eval_checkpoint:  # evaluate the model
        ada_load_state_dict(args.eval_checkpoint, model, use_qkv=False, strict=False)
        if args.use_full_head :
            model.apply(lambda m: setattr(m, 'use_full_linear', True))
        val_metrics = validate(model, loader_eval, validate_loss_fn, args, print_head_option=args.print_head_option, flops_dict=flops_dict)
        print(f"Top-1 accuracy of the model is: {val_metrics['top1']:.1f}%")
        if 'gflops' in val_metrics :
            print(f"avg flops is: {val_metrics['gflops']:.1f} GFLOPS")
        return

    saver = None
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model,
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = MyCheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:  # train the model
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn, total_epochs=num_epochs)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, flops_dict=flops_dict)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.ema, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)', flops_dict=flops_dict)
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.eval()

def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None, total_epochs=0):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    base_model = model.module if hasattr(model, 'module') else model
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    meta_loss_m = {}

    model.train()
    if args.freeze_bn :
        freeze_bn(model)

    end = time.time()
    last_idx = len(loader) - 1

    def update_meta_loss_m(meta_loss) :
        for k, v in meta_loss.items():
            if k not in meta_loss_m :
                meta_loss_m[k] = AverageMeter()
            meta_loss_m[k].update(v.item(), input.size(0))

    num_updates = epoch * len(loader)
    total_num_updates = total_epochs * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            outputs = model(input, training=True, ret_attn_list=False)
            loss, meta_loss = loss_fn(outputs, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
            update_meta_loss_m(meta_loss)
            
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(), create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
                for k, v in meta_loss.items():
                    meta_loss[k] = reduce_tensor(v.data, args.world_size)
                update_meta_loss_m(meta_loss)

            eta_seconds = batch_time_m.avg * (total_num_updates - num_updates)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
 
            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Current Time: {}  '
                    'ETA : {} '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                        eta_string,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict(loss= losses_m.avg, **{x : meta_loss_m[x].avg for x in meta_loss})

def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix='', ret_head_option=False, print_head_option=False, flops_dict=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    head_m = AverageMeter()
    layer_m = AverageMeter()
    token_m = AverageMeter()
    flops_m = AverageMeter()

    analyse_m = [[] for _ in range(1000)]

    head_option = None
    layer_option = None
    token_option = None

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output, head_select, layer_select, token_select = output[:4]
                # output = output[0]
                # head_select = output[1]
            else :
                head_select = None
                layer_select = None
                token_select = None

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
                if head_select is not None :
                    head_select = reduce_tensor(head_select, args.world_size)
                if layer_select is not None :
                    layer_select = reduce_tensor(layer_select, args.world_size)
                if token_select is not None :
                    token_select = reduce_tensor(token_select, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            if head_select is not None :
                head_m.update(head_select.mean().item(), output.size(0))
                if head_option is None :
                    head_option = AverageMeter()
                head_option.update(head_select.mean(0).cpu(), output.size(0))
            if layer_select is not None :
                layer_m.update(layer_select.mean().item(), output.size(0))
                if layer_option is None :
                    layer_option = AverageMeter()
                layer_option.update(layer_select.mean(0).cpu(), output.size(0))
            if token_select is not None :
                token_m.update(token_select.mean().item(), output.size(0))
                if token_option is None :
                    token_option = AverageMeter()
                token_option.update(token_select.mean((0,-1)), output.size(0))

            if flops_dict is not None :
                bs = output.size(0)
                from block_flops_dict import batch_select_flops
                if 'deit' in args.model:
                    b_flops = batch_select_flops(bs, flops_dict, head_select, layer_select, token_select, block_num=12,base_flops=0.06)
                else :
                    b_flops = batch_select_flops(bs, flops_dict, head_select, layer_select, token_select, block_num=19, base_flops=0.33)
                flops_m.update(b_flops.mean(), bs)
                def deal_none(x):
                    return x if x is not None else [None] * bs
                head_select = deal_none(head_select)
                layer_select = deal_none(layer_select)
                token_select = deal_none(token_select)
                for c, b, ht, lt, tt in zip(target, b_flops, head_select, layer_select, token_select):
                    lt = lt.cpu() if lt is not None else None
                    ht = ht.cpu() if ht is not None else None
                    tt = tt.cpu() if tt is not None else None
                    analyse_m[c].append(dict(flops=b.cpu(), layer_select=lt, head_select=ht, token_select=tt))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Head avg :{head_select.val:>7.4f} ({head_select.avg:>6.4f})  '
                    'Layer avg :{layer_m.val:>7.4f} ({layer_m.avg:>6.4f})  '
                    'Token avg :{token_m.val:>7.4f} ({token_m.avg:>6.4f})  '
                    'Flops avg :{flops_m.val:>7.4f} ({flops_m.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m,
                        head_select=head_m, layer_m=layer_m, token_m=token_m, flops_m=flops_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg), ('head', head_m.avg), ('layer', layer_m.avg)])
    if flops_dict is not None :
        metrics.update(gflops=flops_m.avg)
    if head_option is not None :
        if ret_head_option :
            metrics.update(head_option=head_option.avg)
        if print_head_option :
            for i, val in enumerate(head_option.avg) :
                if args.rank == 0 :
                    print('{}: {}'.format(i+args.keep_layers, ','.join(['{:.2f}'.format(float(x)) for x in val])))

    if layer_option is not None :
        if ret_head_option :
            metrics.update(layer_option=layer_option.avg)
        if print_head_option :
            for i, val in enumerate(layer_option.avg) :
                if args.rank == 0 :
                    print('{}: {}'.format(i+args.keep_layers, ','.join(['{:.2f}'.format(float(x)) for x in val])))

    if token_option is not None :
        if ret_head_option :
            metrics.update(token_option=token_option.avg)
        if print_head_option :
            if args.rank == 0 :
                print(' '.join(['{:.2f}'.format(float(x)) for x in token_option.avg]))

    if analyse_m is not None :
        torch.save(analyse_m, '{}_analyse.{}.pth'.format(args.model, top1_m.avg))
    return metrics


if __name__ == '__main__':
    main()
