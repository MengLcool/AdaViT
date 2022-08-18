from functools import partial
from re import L
import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed
import numpy as np
from .token_transformer import Token_transformer
from .token_performer import Token_performer
from .transformer_block import Block, get_sinusoid_encoding
from .ada_transformer_block import StepAdaBlock, get_random_policy


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
}


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            #self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            #self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(dim=in_chans*7*7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x


class AdaStepT2T_ViT(nn.Module):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 norm_policy=False, use_t2t = True, patch_size=16,
                 ada_head = True, ada_head_v2=False, dyna_data=False, ada_head_attn=False, head_slowfast=False,
                 ada_layer=False, 
                 ada_block=False, head_select_tau=5., layer_select_tau=5., token_select_tau=5.,
                 ada_token = False, ada_token_with_mlp=False, ada_token_start_layer=0, ada_token_pre_softmax=True, ada_token_detach_attn=True,  ada_token_detach_attn_at_mlp=False,
                 keep_layers = 1, 
                 head_select_bias=True, layer_select_bias=True,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.use_t2t = use_t2t
        if use_t2t :
            self.tokens_to_token = T2T_module(
                    img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)
            num_patches = self.tokens_to_token.num_patches
        else :
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert keep_layers >=1, 'keep layers must >=1'
        self.keep_layers = keep_layers
        self.head_dim = embed_dim // num_heads
        print('ada head, ada layer, ada token', ada_head, ada_layer, ada_token)
        
        self.blocks = nn.ModuleList([
            StepAdaBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                is_token_select=ada_token and i >= ada_token_start_layer, ada_token_with_mlp=ada_token_with_mlp,
                ada_token_pre_softmax = ada_token_pre_softmax, ada_token_detach_attn=ada_token_detach_attn, dyna_data=dyna_data, ada_head_v2=ada_head_v2, ada_token_detach_attn_at_mlp=ada_token_detach_attn_at_mlp,
                ada_head= ada_head and i>=keep_layers,
                ada_layer= ada_layer and i >=keep_layers,
                norm_policy=norm_policy,
                only_head_attn=ada_head_attn, head_slowfast=head_slowfast)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        if self.use_t2t :
            x = self.tokens_to_token(x)
        else :
            x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attn_list = []
        hidden_list = []
        token_select_list = []
        head_select_list = []
        layer_select_list = []
        head_select_logits_list = []
        layer_select_logits_list = []

        def filter_append(target_list, element):
            if element is not None :
                target_list.append(element)

        for blk in self.blocks :
            x, attn, this_head_select, this_layer_select, this_token_select, this_head_select_logits, this_layer_select_logits = blk(x)
            attn_list.append(attn)
            hidden_list.append(x)
            filter_append(head_select_list, this_head_select)
            filter_append(layer_select_list, this_layer_select)
            filter_append(token_select_list, this_token_select)
            filter_append(head_select_logits_list, this_head_select_logits)
            filter_append(layer_select_logits_list, this_layer_select_logits)

        def convert_list_to_tensor(list_convert):
            if len(list_convert) :
                result = torch.stack(list_convert, dim=1)
            else :
                result = None
            return result 

        head_select = convert_list_to_tensor(head_select_list)
        if head_select is not None :
            head_select = head_select.squeeze(-1)
        layer_select = convert_list_to_tensor(layer_select_list)
        token_select = convert_list_to_tensor(token_select_list)
        head_select_logits = convert_list_to_tensor(head_select_logits_list)
        layer_select_logits = convert_list_to_tensor(layer_select_logits_list)
        a = [head_select, layer_select, token_select, head_select_logits, layer_select_logits]
        x = self.norm(x)
        
        return x[:, 0], head_select, layer_select, token_select, attn_list, hidden_list, dict(head_select_logits=head_select_logits, layer_select_logits=layer_select_logits)

    def zero_classification_grad(self):
        for blk in self.blocks[self.keep_layers:] :
            blk.zero_grad()
        self.norm.zero_grad()
        self.head.zero_grad()

    def forward(self, x, training=False, ret_attn_list=False):
        x, head_select, layer_select, token_select, attn_list, hidden_list, select_logtis = self.forward_features(x)
        x = self.head(x)
        if ret_attn_list :
            return x, head_select, layer_select, token_select, attn_list, hidden_list, select_logtis
        return x, head_select, layer_select, token_select, select_logtis


class T2T_GroupNorm(nn.GroupNorm):
    def __init__(self, num_groups: int, num_channels: int, **kwargs) -> None:
        super().__init__(num_groups, num_channels, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() > 2 :
            input = input.transpose(1,-1)
        result = super().forward(input)
        if input.dim() > 2 :
            result = result.transpose(1,-1)
        return result


def t2t_group_norm(dim, num_groups):
    return T2T_GroupNorm(num_groups, dim)

@register_model
def ada_step_t2t_vit_14_lnorm(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = AdaStepT2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def ada_step_t2t_vit_19_lnorm(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = AdaStepT2T_ViT(tokens_type='performer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def ada_step_deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = AdaStepT2T_ViT(
        use_t2t=False,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def ada_step_deit_small_patch16_224(pretrained=False, **kwargs):
    model = AdaStepT2T_ViT(
        use_t2t=False,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def ada_step_deit_base_patch16_224(pretrained=False, **kwargs): # adopt performer for tokens to token
    model = AdaStepT2T_ViT(
        use_t2t=False,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
