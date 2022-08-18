import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
from torch.nn.modules.normalization import LayerNorm

def _gumbel_sigmoid(
    logits, tau=1, hard=False, eps=1e-10, training = True, threshold = 0.5
):
    if training :
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

def get_random_policy(policy, ratio):
    random_p = torch.empty_like(policy).fill_(ratio).bernoulli() + policy * 0.0  # add policy * 0.0 into the loop of loss calculation to avoid the DDP issue
    return random_p

class SimpleTokenSelect(nn.Module):
    def __init__(self, dim_in, tau=5, is_hard=True, threshold=0.5, bias=True, pre_softmax=True, mask_filled_value=float('-inf'), ada_token_nonstep=False, ada_token_detach_attn=True):
        super().__init__()
        self.count_flops = False
        self.ada_token_nonstep = ada_token_nonstep  # if using nonstep, no mlp_head is needed in each of these layers
        if not ada_token_nonstep:
            self.mlp_head = nn.Linear(dim_in, 1, bias=bias)
        self.norm = nn.Identity()
        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold
        self.add_noise = True
        self.pre_softmax = pre_softmax
        self.mask_filled_value = mask_filled_value
        self.ada_token_detach_attn = ada_token_detach_attn
        self.random_policy = False
        self.random_token = False
        self.random_token_ratio = 1.

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, x, attn, attn_pre_softmax, token_select=None):
        b, l = x.shape[:2]

        if not self.ada_token_nonstep:
            # generate token policy step by step in each layer, including the first (couple of) blocks
            logits = self.mlp_head(self.norm(x[:,1:]))
            token_select = _gumbel_sigmoid(logits, self.tau, self.is_hard, threshold=self.threshold, training=self.training)
            if self.random_policy or self.random_token:
                token_select = get_random_policy(token_select, self.random_token_ratio)
            token_select = torch.cat([token_select.new_ones(b,1,1), token_select], dim=1)
            # token_select = token_select.unsqueeze(-1) #(b,l,1)
            token_select = token_select.transpose(1,2) #(b,1,l)
        else:
            if token_select is None:
                # when token_select is not given in non-step setting, 
                # it means this layer is in the first (couple of) trans blocks before head/layer policy generation,
                # and thus we do not drop any token in this/these layers as well for consistency
                token_select = torch.ones((b, 1, l), device=x.device)
            else:
                token_select = token_select[:, None, :]

        if self.count_flops :
            return attn, token_select.squeeze(1)

        attn_policy = torch.bmm(token_select.transpose(-1,-2), token_select) #(b,l,l)
        attn_policy = attn_policy.unsqueeze(1) #(b,1,l,l)
        if self.ada_token_detach_attn :
            attn_policy = attn_policy.detach()

        # use pre_softmax during inference in both pre-softmax or pre-softmax training
        if self.pre_softmax or not self.training :
            eye_mat = attn.new_zeros((l,l))
            eye_mat = eye_mat.fill_diagonal_(1) #(1,1,l,l)
            attn = attn_pre_softmax * attn_policy + attn_pre_softmax.new_zeros(attn_pre_softmax.shape).masked_fill_((1 - attn_policy - eye_mat)>0, self.mask_filled_value)
            attn = attn.softmax(-1)
            assert not torch.isnan(attn).any(), 'token select pre softmax nan !'
        else :
            attn = nn.functional.normalize(attn * attn_policy, 1, -1)

        return attn, token_select.squeeze(1)


class BlockHeadSelect(nn.Module):
    def __init__(self, dim_in, num_heads, tau=5, is_hard=True, threshold=0.5, bias=True):
        super().__init__()
        self.mlp_head = nn.Linear(dim_in, num_heads, bias=bias)
        # self.norm = Lay   rNorm(dim_in)
        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold
        self.add_noise = True
        self.head_dim = dim_in // num_heads
        self.random_policy = False
        self.random_head = False
        self.random_head_ratio = 1.

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        '''
        ret : tensor(b, dim, 1)
        '''
        bsize = x.shape[0]
        logits = self.mlp_head(x)
        sample = _gumbel_sigmoid(logits, self.tau, self.is_hard, threshold=self.threshold, training=self.training)
        if self.random_policy or self.random_head:
            sample = get_random_policy(sample, self.random_head_ratio)
        sample = sample.unsqueeze(-1) #(b,h,1)

        width_select = sample.expand(-1,-1,self.head_dim)
        width_select = width_select.reshape(bsize, -1, 1)

        return sample, width_select, logits


class BlockLayerSelect(nn.Module):
    def __init__(self, dim_in, num_sub_layer, tau=5, is_hard=True, threshold=0.5, bias=True):
        super().__init__()
        self.mlp_head = nn.Linear(dim_in, num_sub_layer, bias=bias)
        # self.norm = LayerNorm(dim_in)
        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold
        self.add_noise = True
        self.random_policy = False
        self.random_layer = False
        self.random_layer_ratio = 1.

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        logits = self.mlp_head(x)
        sample = _gumbel_sigmoid(logits, self.tau, self.is_hard, threshold=self.threshold, training=self.training)
        if self.random_policy or self.random_layer:
            sample = get_random_policy(sample, self.random_layer_ratio)
        sample = sample #(b,2)

        return sample, logits


class DynaLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_heads=6, bias=True, dyna_dim=[True, True], dyna_data=False):
        super(DynaLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features

        self.num_heads = num_heads
        self.width_mult = 1.
        self.dyna_dim = dyna_dim
        self.in_features = in_features
        self.out_features = out_features
        self.use_full_linear = False
        self.dyna_data = dyna_data # if dyna_data is False, dyna weights
        self.count_flops = False

    def forward(self, input,  width_select=None, width_specify=None):
        """
        input : tensor (B,L,C)
        width_select : tensor(B,1,dims) or (B,dims,1)
        """
        if self.use_full_linear :
            return super().forward(input)

        if self.count_flops :
            if width_select is not None :
                assert width_select.shape[0] == 1
                width_specify = int(width_select.sum().item())
                width_select = None

        if self.dyna_data and width_select is not None :
            # only support input shape of (b,l,c)
            assert input.dim() == 3
            assert width_select.dim() == 3
            assert width_select.shape[1] == 1 or width_select.shape[2] == 1
            # if output is static, then input is dynamic
            if width_select.shape[1] == 1 :
                input_mask = width_select
            else :
                input_mask = 1
            if width_select.shape[2] == 1 :
                output_mask = width_select[...,0].unsqueeze(1) #(b,1,c)
            else :
                output_mask = 1
            input = input * input_mask
            result = super().forward(input) * output_mask
            return result

        if width_select is not None:
            weight = self.weight * width_select
            b, n, c = input.shape
            input = input.transpose(1,2).reshape(1,-1,n)
            weight = weight.view(-1,c,1)
            if self.bias is None :
                bias = self.bias
            elif width_select.shape[-1] == 1:
                bias = self.bias * width_select.squeeze()
                bias = bias.view(-1)
            else :
                bias = self.bias.unsqueeze(0).expand(b,-1)
                bias = bias.reshape(-1)
            result = nn.functional.conv1d(input, weight, bias, groups=b)
            result = result.view(b,-1,n).transpose(1,2) #(b,n,c)
            return result
        
        if width_specify is not None :
            if self.dyna_dim[0] :
                self.in_features = width_specify
            if self.dyna_dim[1] :
                self.out_features = width_specify
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias

        return nn.functional.linear(input, weight, bias)


class AdaAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ada_token=False, ada_token_nonstep=False, ada_token_pre_softmax=True, ada_token_detach_attn=True, dyna_data=False, ada_token_threshold=0.6):
        super().__init__()
        self.count_flops = False
        self.t_ratio = 1

        self.num_heads = num_heads
        # head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query = DynaLinear(dim, dim, bias=qkv_bias, dyna_dim=[False, True], dyna_data=dyna_data)
        self.key = DynaLinear(dim, dim, bias=qkv_bias, dyna_dim=[False, True], dyna_data=dyna_data)
        self.value = DynaLinear(dim, dim, bias=qkv_bias, dyna_dim=[False, True], dyna_data=dyna_data)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = DynaLinear(dim, dim, dyna_dim=[True, True], dyna_data=dyna_data)
        self.proj_drop = nn.Dropout(proj_drop)

        if ada_token :
            self.token_select = SimpleTokenSelect(dim, pre_softmax=ada_token_pre_softmax, ada_token_detach_attn=ada_token_detach_attn, ada_token_nonstep=ada_token_nonstep, threshold=ada_token_threshold)
        else :
            self.token_select = None

    def forward_count_flops(self, x, width_select=None, head_select=None, token_select=None, only_head_attn=False):
        width_specify=None
        B, N, C = x.shape
        width_select_qk = width_select
        if only_head_attn :
            assert head_select is not None
            width_select = None

        if self.token_select is not None :
            token_active = int(x.shape[1] * self.t_ratio)
        else :
            token_active = x.shape[1]

        q = self.query(x[:,:token_active], width_select=width_select_qk, width_specify=width_specify).reshape(B, token_active, -1, C//self.num_heads).permute(0,2,1,3)
        k = self.key(x[:,:token_active], width_select=width_select_qk, width_specify=width_specify).reshape(B, token_active, -1, C//self.num_heads).permute(0,2,1,3)
        v = self.value(x[:,:token_active], width_select=width_select, width_specify=width_specify).reshape(B, token_active, -1, C//self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_pre_softmax = attn
        attn = attn.softmax(dim=-1)

        attn_origin = attn

        if self.token_select is not None :
            attn, token_select = self.token_select(x, attn, attn_pre_softmax, token_select=token_select)

        attn = self.attn_drop(attn)
        if only_head_attn :
            v[:,:attn.shape[1]] = (attn @ v[:,:attn.shape[1]])
            x = v.transpose(1, 2).reshape(B, token_active, -1)
        else :
            x = (attn @ v).transpose(1, 2).reshape(B, token_active, -1)
        if width_select is not None :
            width_select = width_select.transpose(-1,-2)
        x = self.proj(x, width_select, width_specify=width_specify)
        
        x[:, :token_active] = self.proj_drop(x[:, :token_active])
        
        return x, attn_origin, token_select

    def forward(self, x, mask=None, value_mask_fill=-1e4, head_mask=None, width_select=None, head_select=None, token_select=None, width_specify=None, token_keep_ratio=None, only_head_attn=False,
                      random_token_select_ratio=1.0):

        if self.count_flops :
            return self.forward_count_flops(x, width_select, head_select, token_select, only_head_attn)
        B, N, C = x.shape
        if only_head_attn :
            assert head_select is not None
            width_select = None

        q = self.query(x, width_select=width_select, width_specify=width_specify).reshape(B, N, -1, C//self.num_heads).permute(0,2,1,3)
        k = self.key(x, width_select=width_select, width_specify=width_specify).reshape(B, N, -1, C//self.num_heads).permute(0,2,1,3)
        v = self.value(x, width_select=width_select, width_specify=width_specify).reshape(B, N, -1, C//self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None :
            mask = mask.view(B, 1, N, 1).expand_as(attn)
            attn[~mask] = value_mask_fill
        attn_pre_softmax = attn
        attn = attn.softmax(dim=-1)
        if only_head_attn :
            head_select = head_select.view(*head_select.shape, *([1]*(4-head_select.dim())))
            eye_mat = attn.new_zeros(attn.shape[-2:])
            eye_mat.fill_diagonal_(1).view(1,1,*eye_mat.shape) #(1,1,l,l)
            attn = attn * head_select + eye_mat * (1 - head_select)

        attn_origin = attn
        if head_mask is not None :
            attn = attn * head_mask

        if self.token_select is not None :
            attn, token_select = self.token_select(x, attn, attn_pre_softmax, token_select=token_select)
            if only_head_attn and not self.training :
                head_select = head_select.view(*head_select.shape, *([1]*(4-head_select.dim())))
                eye_mat = attn.new_zeros(attn.shape[-2:])
                eye_mat.fill_diagonal_(1).view(1,1,*eye_mat.shape) #(1,1,l,l)
                attn = attn * head_select + eye_mat * (1 - head_select)

        if random_token_select_ratio != 1.0:
            # test random baseline with predefined token select ratio
            token_select = (torch.rand((B, N), device=x.device) < random_token_select_ratio).float()
            token_select[:, 0] = 1.0  # CLS token is always kept
            attn_policy = torch.bmm(token_select[:, None, :].transpose(-1,-2), token_select[:, None, :]) #(b,l,l)
            attn = nn.functional.normalize(attn * attn_policy.unsqueeze(1), 1, -1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        if width_select is not None :
            width_select = width_select.transpose(-1,-2)
        x = self.proj(x, width_select, width_specify=width_specify)
        
        if token_select is not None :
            x = x * token_select.unsqueeze(-1)
        x = self.proj_drop(x)
        
        return x, attn_origin, token_select


class AdaMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., dyna_data=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.fc1 = DynaLinear(in_features, hidden_features, dyna_dim=[True, False], dyna_data=dyna_data)
        self.fc2 = DynaLinear(hidden_features, out_features, dyna_dim=[False, False], dyna_data=dyna_data)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None, width_select=None, width_specify=None):
        if mask is not None :
            assert mask.shape[:2] == x.shape[:2]
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            if mask.dtype != x.dtype :
                mask = mask.type_as(x)
        else :
            mask = x.new_ones(x.shape[:2]).unsqueeze(-1)
        x = self.fc1(x, width_select=width_select, width_specify=width_specify)
        x = x * mask
        x = self.act(x)
        x = self.drop(x)
        width_select = None
        x = self.fc2(x, width_select=width_select, width_specify=width_specify)
        x = x * mask
        x = self.drop(x)
        return x


class StepAdaBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                    ada_head=False, ada_layer=False, is_token_select=False, ada_token_pre_softmax=True, ada_token_detach_attn=True,
                    ada_token_with_mlp=False, ada_token_detach_attn_at_mlp=True,
                    dyna_data=False, ada_head_v2=False, only_head_attn=False, head_slowfast=False,
                    norm_policy=False):
        super().__init__()
        self.count_flops = False
        self.h_ratio, self.t_ratio = 1., 1.
        self.l_ratio = [1, 1]
        self.is_token_select = is_token_select

        self.norm_policy = None
        if norm_policy and (ada_head or ada_layer):
            self.norm_policy = norm_layer(dim)
        self.norm1 = norm_layer(dim)
        self.attn = AdaAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, ada_token=is_token_select, ada_token_pre_softmax=ada_token_pre_softmax, ada_token_detach_attn=ada_token_detach_attn,
            dyna_data=dyna_data)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.ada_head_v2 = ada_head_v2
        self.mlp = AdaMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, dyna_data=dyna_data)
        self.ada_token_with_mlp = ada_token_with_mlp
        self.ada_token_detach_attn_at_mlp = ada_token_detach_attn_at_mlp
        self.only_head_attn = only_head_attn and ada_head
        self.head_slowfast = head_slowfast

        self.head_select = None 
        self.layer_select = None 
        if ada_head :
            self.head_select = BlockHeadSelect(dim, num_heads)
        if ada_layer :
            self.layer_select = BlockLayerSelect(dim, 2)

    def forward_count_flops(self, x) :
        if self.norm_policy is not None :
            policy_token = self.norm_policy(x)[:,0]
        else :
            policy_token = x[:,0]
        if self.layer_select is not None :
            sub_layer_select, layer_logits = self.layer_select(policy_token)
        else :
            sub_layer_select, layer_logits = None, None
        if self.head_select  is not None :
            head_select, width_select, head_logits = self.head_select(policy_token)
        else :
            head_select, width_select, head_logits = None, None, None

        def mask_policy(policy, ratio) :
            if policy is not None :
                policy = torch.zeros_like(policy)
                policy[:, :int(policy.shape[1] * ratio)] = 1
            return policy
        h_ratio, l_ratio, t_ratio = [1, 1, 1]
        if head_select is not None :
            h_ratio = self.h_ratio
            head_select = mask_policy(head_select, h_ratio)
        if width_select is not None :
            width_select = mask_policy(width_select, h_ratio)
        if sub_layer_select is not None :
            l_ratio = self.l_ratio
            sub_layer_select[:,0] = l_ratio[0]
            sub_layer_select[:,1] = l_ratio[1]
        if self.is_token_select :
            t_ratio = self.t_ratio

        if width_select is not None :
            # TODO
            if width_select.sum() == 0 :
                return x
            width_select_attn = width_select #(b,c,1)
            if self.only_head_attn :
                assert head_select is not None
                width_select_mlp = None
            elif self.ada_head_v2 :
                bs = width_select.shape[0]
                width_select_mlp = width_select.expand(-1,-1,int(self.mlp_ratio)).reshape(bs,-1,1)
            else :
                width_select_mlp = width_select.transpose(-1,-2)
        else :
            width_select_attn, width_select_mlp = [None] * 2

        if sub_layer_select is None or sub_layer_select[0,0] :
            attn_x, attn_origin, token_select = self.attn(self.norm1(x), width_select=width_select_attn,
                                                        head_select=head_select, only_head_attn=self.only_head_attn)
            x[:,:int(x.shape[1] * t_ratio),:attn_x.shape[-1]] = x[:,:int(x.shape[1] * t_ratio),:attn_x.shape[-1]] + attn_x
        elif sub_layer_select[0,0] == 0:
            attn_x = 0
            x = x + attn_x

        x = self.norm2(x)
        if self.only_head_attn :
            mlp_x = x 
        else :
            mlp_x = x

        if sub_layer_select is not None and sub_layer_select[0,1] == 0 :
            x = x + 0
        else :
            if self.ada_token_with_mlp :
                token_active = int(x.shape[1] * t_ratio)
            else :
                token_active = x.shape[1]

            mlp_x = self.mlp(mlp_x[:,:token_active], width_select=width_select_mlp)
            x[:,:token_active] = x[:,:token_active] + mlp_x

        return x

    def forward(self, x, mask=None, head_mask=None, width_specify=None, # only_head_attn=False, head_slowfast=False,
                    random_token_select_ratio=1.0):
        """
        width_select : (b,c,1)
        """
        if self.count_flops :
            return self.forward_count_flops(x)

        if self.norm_policy is not None :
            policy_token = self.norm_policy(x)[:,0]
        else :
            policy_token = x[:,0]
        if self.layer_select is not None :
            sub_layer_select, layer_logits = self.layer_select(policy_token)
        else :
            sub_layer_select, layer_logits = None, None
        if self.head_select  is not None :
            head_select, width_select, head_logits = self.head_select(policy_token)
        else :
            head_select, width_select, head_logits = None, None, None

        # start 
        if self.only_head_attn :
            assert head_select is not None
            width_select = None
        if width_select is not None :
            # TODO
            width_select_attn = width_select #(b,c,1)
            if self.ada_head_v2 :
                bs = width_select.shape[0]
                width_select_mlp = width_select.expand(-1,-1,int(self.mlp_ratio)).reshape(bs,-1,1)
            else :
                width_select_mlp = width_select.transpose(-1,-2)
        else :
            width_select_attn, width_select_mlp = [None] * 2

        attn_x, attn_origin, token_select = self.attn(self.norm1(x), mask=mask, head_mask=head_mask, width_select=width_select_attn,
                                                width_specify=width_specify, head_select=head_select, only_head_attn=self.only_head_attn, 
                                                random_token_select_ratio=random_token_select_ratio)

        if sub_layer_select is None :
            x = x + self.drop_path(attn_x)
            mlp_x = self.mlp(self.norm2(x), width_select=width_select_mlp, width_specify=width_specify)
            if self.ada_token_with_mlp and token_select is not None :
                t_select = token_select.unsqueeze(-1)
                if self.ada_token_detach_attn_at_mlp:
                    t_select = t_select.detach()
                mlp_x = t_select * mlp_x
            x = x + self.drop_path(mlp_x)
        else :
            x = x + sub_layer_select[:,0][:,None,None] * attn_x
            mlp_x = self.mlp(self.norm2(x), width_select=width_select_mlp, width_specify=width_specify)
            if self.ada_token_with_mlp and token_select is not None :
                t_select = token_select.unsqueeze(-1)
                if self.ada_token_detach_attn_at_mlp:
                    t_select = t_select.detach()
                mlp_x = t_select * mlp_x
            x = x + sub_layer_select[:,1][:,None,None] * mlp_x
        return x, attn_origin, head_select, sub_layer_select, token_select, head_logits, layer_logits
