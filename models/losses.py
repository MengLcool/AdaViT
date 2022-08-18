from numpy.lib.arraysetops import isin
from timm import loss
from timm.data.transforms_factory import transforms_imagenet_train
import torch
from torch.functional import Tensor
import torch.nn as nn

def binaray_entropy(prob, eps=1e-7):
    neg_entro = prob * prob.clamp(min=eps).log() + (1-prob) * (1-prob).clamp(min=eps).log()
    return - neg_entro

class BgLoss(nn.Module):
    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion

    def forward(self, outputs, y) :
        assert isinstance(outputs, tuple) and len(outputs) == 2, 'err {} {}'.format(type(outputs), len(outputs))
        cls_pred, bg_pred = outputs
        if y.dim() == 2 :
            bsize, c = y.shape
            y_min = y.min(-1, keepdim=True)[0]
            y = torch.cat([y_min, y], dim=1)
            y_bg = torch.cat([y.new_ones(bsize,1), y.new_ones(bsize,c) * y_min], dim=1)
            y = nn.functional.normalize(y, 1, -1)
            y_bg = nn.functional.normalize(y_bg, 1, -1)
        else :
            y = y+1
            y_bg = y.new_zeros(y.shape)
        base_loss = self.base_criterion(cls_pred, y)
        bg_loss = self.base_criterion(bg_pred, y_bg)

        return (base_loss + bg_loss).mean()


class AdaHeadLoss(nn.Module):
    def __init__(self, base_criterion, target_ratio=0.5, head_loss_ratio=2., diverse_ratio=0.1):
        super().__init__()
        self.base_criterion = base_criterion
        self.target_ratio = target_ratio
        self.head_loss_ratio = head_loss_ratio
        self.diverse_ratio = diverse_ratio

    def forward(self, outputs, y):
        '''
        head_select: (b, num_layers, num_head)
        '''
        assert len(outputs) >= 2 
        x, head_select = outputs[:2]
        base_loss = self.base_criterion(x, y)
        head_mean = head_select.mean()
        flops_loss = (head_mean - self.target_ratio).abs().mean()

        head_mean = head_select.mean(0) # (num_layers, num_head)
        diverse_loss = (head_mean - self.target_ratio).abs().mean()

        head_loss = flops_loss + self.diverse_ratio * diverse_loss

        loss = base_loss + self.head_loss_ratio * head_loss

        return loss, dict(base_loss=base_loss, head_loss=head_loss)


class AdaLoss(nn.Module):
    def __init__(self, base_criterion, head_target_ratio=0.5, layer_target_ratio=0.5, head_loss_ratio=2.,layer_loss_ratio=2., head_diverse_ratio=0.1, layer_diverse_ratio=0.1,
                    head_entropy_weight=0.1, layer_entropy_weight=0.1, 
                    head_minimal_weight=0., head_minimal=0.,
                    layer_minimal_weight=0., layer_minimal=0.,
                    token_target_ratio=0.5, token_loss_ratio=2., token_minimal=0.1, token_minimal_weight=1.):
        super().__init__()
        self.base_criterion = base_criterion
        self.head_target_ratio = head_target_ratio
        self.layer_target_ratio = layer_target_ratio

        self.head_loss_ratio = head_loss_ratio
        self.layer_loss_ratio = layer_loss_ratio

        self.head_diverse_ratio = head_diverse_ratio
        self.layer_diverse_ratio = layer_diverse_ratio

        self.head_entropy_weight = head_entropy_weight
        self.layer_entropy_weight = layer_entropy_weight

        self.head_minimal_weight = head_minimal_weight
        self.head_minimal = head_minimal

        self.layer_minimal_weight = layer_minimal_weight
        self.layer_minimal = layer_minimal

        self.token_target_ratio = token_target_ratio
        self.token_loss_ratio = token_loss_ratio
        self.token_minimal = token_minimal
        self.token_minimal_weight = token_minimal_weight

    def forward(self, outputs, y):
        '''
        head_select: (b, num_layers, num_head)
        '''
        assert len(outputs) >= 3
        x, head_select, layer_select, token_select = outputs[:4]
        logits_set = outputs[-1]

        base_loss = self.base_criterion(x, y)
        layer_loss = self._get_layer_loss(x, layer_select, logits_set)
        head_loss = self._get_head_loss(x, head_select, logits_set, layer_select)
        token_loss = self._get_token_loss(x, token_select)
        
        loss = base_loss + self.head_loss_ratio * head_loss + self.layer_loss_ratio * layer_loss + self.token_loss_ratio * token_loss

        return loss, dict(base_loss=base_loss, head_loss=head_loss, layer_loss=layer_loss, token_loss=token_loss)

    def _get_head_loss(self, x, head_select, logits_set, layer_select):
        eps = 1e-6
        if head_select is not None :
            if layer_select is not None :
                block_select = layer_select.sum(-1, keepdim=True)
                head_select_mask = (block_select > 0).type_as(block_select)
                head_select_mask = head_select_mask.expand(-1,-1,head_select.shape[-1])
                assert head_select.shape == head_select_mask.shape
            else :
                head_select_mask = head_select.new_ones(head_select.shape)
            head_mean = (head_select * head_select_mask).sum() / (head_select_mask.sum() + eps)
            head_flops_loss = (head_mean - self.head_target_ratio).abs().mean()

            if self.head_diverse_ratio > 0 :
                # head_mean = head_select.mean(0) # (num_layers, num_head)
                head_mean = (head_select * head_select_mask).sum(0) / (head_select_mask.sum(0) + eps)
                head_diverse_loss = (head_mean - self.head_target_ratio).abs().mean()
            else :
                head_diverse_loss = 0

            if self.head_minimal_weight > 0 :
                # head_per_layer = head_select.sum(-1) #(b, num_layers)
                # head_minimal_loss = (1 - head_per_layer).clamp(min=0.).sum(-1).mean()
                # head_mean = head_select.mean(0) # (num_layers, num_head)
                head_mean = (head_select * head_select_mask).sum(0) / (head_select_mask.sum(0) + eps)
                head_minimal_loss = (self.head_minimal - head_mean).clamp(min=0.).sum()
            else :
                head_minimal_loss = 0

            if self.head_entropy_weight > 0 :
                head_select_logits = logits_set['head_select_logits']
                head_entropy = binaray_entropy(head_select_logits.sigmoid()).mean()
            else :
                head_entropy = 0

            head_loss = head_flops_loss + self.head_diverse_ratio * head_diverse_loss - self.head_entropy_weight * head_entropy \
                        + self.head_minimal_weight * head_minimal_loss
        else :
            head_loss = x.new_zeros(1).mean()
        
        return head_loss
    
    def _get_layer_loss(self, x, layer_select, logits_set):
        if layer_select is not None :
            layer_mean = layer_select.mean()
            layer_flops_loss = (layer_mean - self.layer_target_ratio).abs().mean()

            if self.layer_diverse_ratio > 0 :
                layer_mean = layer_select.mean((0,-1))
                layer_diverse_loss = (layer_mean - self.layer_target_ratio).abs().mean()
            else :
                layer_diverse_loss = 0

            if self.layer_entropy_weight > 0 :
                layer_select_logits = logits_set['layer_select_logits']
                layer_entropy = binaray_entropy(layer_select_logits.sigmoid()).mean()
            else :
                layer_entropy = 0

            if self.layer_minimal_weight > 0 :
                layer_mean = layer_select.mean(0) #(num_layers, 2)
                layer_minimal_loss = (self.layer_minimal - layer_mean).clamp(min=0.).sum()
            else :
                layer_minimal_loss = 0

            layer_loss = layer_flops_loss + self.layer_diverse_ratio * layer_diverse_loss - self.layer_entropy_weight * layer_entropy \
                            + self.layer_minimal_weight * layer_minimal_loss
        else :
            layer_loss = x.new_zeros(1).mean()

        return layer_loss

    def _get_token_loss(self, x, token_select):
        """
        token_select : tensor (b, num_layer, l)

        """
        if token_select is not None :
            token_mean = token_select.mean()
            # token_flops_loss = (token_mean - self.token_target_ratio).abs().mean()
            # token_flops_loss = (token_mean - self.token_target_ratio).clamp(min=0.).mean()
            token_flops_loss = ((token_mean - self.token_target_ratio)**2).mean()

            if self.token_minimal_weight > 0 :
                token_mean = token_select.mean(-1)
                token_minimal_loss = (self.token_minimal - token_mean).clamp(min=0.).sum()
            else :
                token_minimal_loss = 0

            token_loss = token_flops_loss + self.token_minimal_weight * token_minimal_loss
        else :
            token_loss = x.new_zeros(1).mean()

        return token_loss

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).sum(-1).mean()

# TODO : hard or soft distill loss
class TeacherLoss(nn.Module):
    def __init__(self, teacher_model, base_criterion, kd_ratio=1., tau=5., attn_ratio=.5, hidden_ratio=.1, pred_ratio=.5, keep_layers=1):
        super().__init__()
        self.kd_ratio = kd_ratio
        print('self tau', tau)
        self.tau = tau
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        self.attn_ratio = attn_ratio
        self.hidden_ratio = hidden_ratio
        # self.layer_ratio = layer_ratio
        self.pred_ratio = pred_ratio
        self.teacher_model.eval()
        self.keep_layers = keep_layers

    def forward(self, x, outputs, y):
        assert len(outputs) >= 5
        logits, head_select, layer_select, token_select, attn_list, hidden_list = outputs[:6]
        base_loss, meta_loss = self.base_criterion(outputs, y)

        with torch.no_grad():
            logits_teacher, attn_list_teacher, hidden_list_teacher = self.teacher_model(x, ret_attn_list=True)

        attn_loss = x.new_zeros(1).mean()
        hidden_loss = x.new_zeros(1).mean()
        if head_select is not None :
            head_select_start = len(attn_list) - head_select.shape[1]
        for i, (attn_s, attn_t, hidden_s, hidden_t) in enumerate(zip(attn_list, attn_list_teacher, hidden_list, hidden_list_teacher)) :
            assert attn_s.dim() == 4 and attn_t.dim() ==4
            if i >= self.keep_layers and layer_select is not None :
                this_select = layer_select[:,i-self.keep_layers].detach() #(b,2)
                this_select = (this_select > 0.5).float().unsqueeze(-1)
                attn_select = this_select[:,0]
                hidden_select = this_select[:,1]
            else :
                this_select = 1.
                attn_select = 1.
                hidden_select = 1.
            
            if head_select is not None and i >= head_select_start :
                attn_mask = head_select.detach()[:,i-head_select_start] #(b,head,1)
                if attn_mask.dim() == 2 :
                    attn_mask = attn_mask.unsqueeze(-1)
            else :
                attn_mask = 1

            attn_s = attn_s[:,:,0] * attn_mask #(b,head,len)
            # TODO : normalize teacher attn
            attn_t = attn_t[:,:,0] * attn_mask
            hidden_s = hidden_s[:,0] #(b, c)
            hidden_t = hidden_t[:,0]
            # attn_s = torch.where(attn_s <=1e-2, attn_s.new_zeros(attn_s.shape), attn_s)
            # attn_t = torch.where(attn_t <=1e-2, attn_t.new_zeros(attn_t.shape), attn_t)
            # TODO : how to design attn loss
            if self.attn_ratio > 0 :
                this_attn_loss = self.l1_loss(attn_s, attn_t).sum(-1)
                this_attn_loss = (attn_select * this_attn_loss).mean()
                attn_loss +=this_attn_loss
            if self.hidden_ratio > 0 :
                this_hidden_loss = self.mse_loss(hidden_s, hidden_t)
                this_hidden_loss = (hidden_select * this_hidden_loss).mean()
                hidden_loss += this_hidden_loss

        if self.pred_ratio > 0 :
            T = self.tau
            pred_loss = soft_cross_entropy(logits/T, logits_teacher/T)
        else :
            pred_loss = 0

        loss = base_loss + self.attn_ratio*attn_loss+ self.hidden_ratio*hidden_loss + self.pred_ratio*pred_loss
        meta_loss.update(attn_loss=attn_loss, hidden_loss=hidden_loss, pred_loss=pred_loss)
        return loss, meta_loss