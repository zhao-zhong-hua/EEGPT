import torch
from torch.cuda.amp import autocast
import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn

import math

import torch

from logging import getLogger

logger = getLogger()


CHANNEL_DICT = {k.upper():v for v,k in enumerate(
                     [      'FP1', 'FPZ', 'FP2', 
                        "AF7", 'AF3', 'AF4', "AF8", 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 'PO8', 
                               'O1', 'OZ', 'O2', ])}

################################# Utils ######################################

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def apply_mask(mask, x):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), C, D (feature-dim)]
    :param mask: tensor [mN, mC] containing indices of patches in [N, C] to keep 
    """    
    B, N, C, D = x.shape
    if len(mask.shape)==2:
        mN, mC = mask.shape
        
        mask_keep = mask.reshape((1,mN*mC,1)).repeat((B, 1, D))
        masked_x = torch.gather(x.reshape((B, N*C, D)), dim=-2, index=mask_keep)
        masked_x = masked_x.contiguous().view((B,mN,mC,D))
    else:
        mN = mask.shape[0]
        
        mask_keep = mask.reshape((1,mN,1)).repeat((B, 1, D))
        masked_x = torch.gather(x.reshape((B, N*C, D)), dim=-2, index=mask_keep)
    return masked_x

def apply_mask_t(mask_t, x):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), C, D (feature-dim)]
    :param mask: tensor [mN, mC] containing indices of patches in [N, C] to keep 
    """    
    B, N, D = x.shape
    mN = mask_t.shape[0]
    
    mask_keep = mask_t.reshape((1,mN,1)).repeat((B, 1, D))
    masked_x = torch.gather(x, dim=1, index=mask_keep)
    return masked_x

def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x

# helper functions
def exists(val):
    return val is not None

# rotary embedding helper functions

def rotate_half(x):
    
    # x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x = x.reshape((*x.shape[:-1],x.shape[-1]//2, 2))
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    # return rearrange(x, '... d r -> ... (d r)')
    return x.flatten(-2)

def apply_rotary_emb(freqs, t, start_index = 0, scale = 1.):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim = -1)

################################# RoPE Model Begin ######################################
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        theta = 10000,
        learned_freq = False,
        interpolate_factor = 1.
    ):
        super().__init__()
        
        self.cache = dict()
        self.cache_scale = dict()
        self.freqs = nn.Parameter(
            1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)), 
            requires_grad = learned_freq)
        
        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        self.register_buffer('scale', None)
        
    def prepare_freqs(self, num_patches = (1, 8), device='cuda', dtype=torch.float, offset = 0):
        # num_patches (C, N)
        C, N = num_patches
        cache_key = f'freqs:{num_patches}'
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        seq_pos = torch.arange(N, device = device, dtype = dtype)
        seq_pos = seq_pos.repeat_interleave(repeats=C, dim=0) # correspond to x (B, N, C, D)
        seq_pos = (seq_pos + offset) / self.interpolate_factor
        
        freqs = self.freqs
        freqs = torch.outer(seq_pos.type(freqs.dtype), freqs) # (n_seq_pos, n_freqs)
        freqs = freqs.repeat_interleave(repeats=2, dim=-1)    # (n_seq_pos, n_freqs*2)

        self.cache[cache_key] = freqs

        return freqs
    
################################# EEGPT Model Begin ######################################

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
    
    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., is_causal=False, use_rope=False, return_attention=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.use_rope = use_rope
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal
        self.return_attention= return_attention

    def forward(self, x, freqs=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3,B,nh,t,d
        q, k, v = qkv[0], qkv[1], qkv[2] # B,nh,t,d
        
        if self.use_rope:# RoPE
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
        if self.return_attention:
            if self.is_causal:
                attn_mask = torch.ones(q.size(-2), q.size(-2), dtype=torch.bool).tril(diagonal=0)
                attn_maak = torch.zeros(q.size(-2), q.size(-2))
                attn_mask = attn_maak.masked_fill(torch.logical_not(attn_mask), -float('inf'))
                attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
            else:
                attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
            return attn_weight
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=self.is_causal)
        x = y.transpose(1, 2).contiguous().view(B, T, C) #(B, nh, T, hs) -> (B, T, hs*nh)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, is_causal=False, use_rope=False, return_attention=False):
        super().__init__()
        
        self.return_attention= return_attention
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, is_causal=is_causal, use_rope=use_rope, return_attention = return_attention)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, freqs=None):
        y = self.attn(self.norm1(x), freqs)
        if self.return_attention: return y
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(64, 1000), patch_size=16, patch_stride=None, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.num_patches = ((img_size[0]), (img_size[1] // patch_size))
        else:
            self.num_patches = ((img_size[0]), ((img_size[1] - patch_size) // patch_stride + 1))

        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1,patch_size), 
                              stride=(1, patch_size if patch_stride is None else patch_stride))
        
    def forward(self, x):
        # x: B,C,T
        x = x.unsqueeze(1)# B, 1, C, T
        x = self.proj(x).transpose(1,3) # B, T, C, D
        return x

################################# Finetune Model Begin ######################################
class EEGTransformerReconstructor(nn.Module):
    """ EEG Transformer """
    def __init__(
        self,
        num_patches,
        patch_size=64,
        embed_num=1,
        use_pos_embed = False,
        use_inp_embed = True,
        embed_dim=768,
        reconstructor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        interpolate_factor = 2.,
        return_attention_layer=-1,
        **kwargs
    ):
        super().__init__()
        self.use_inp_embed = use_inp_embed
        self.use_pos_embed = use_pos_embed
        self.num_patches = num_patches
        
        
        # --
        self.cls_token = nn.Parameter(torch.zeros(1, 1, reconstructor_embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        # --
        if use_inp_embed:
            self.reconstructor_embed = nn.Linear(embed_dim, reconstructor_embed_dim, bias=True)
        
        if use_pos_embed:
            self.pos_embed           = nn.Parameter(torch.zeros(1, 1, embed_num, reconstructor_embed_dim))
            trunc_normal_(self.pos_embed, std=init_std)
        
        self.mask_token          = nn.Parameter(torch.zeros(1, 1, reconstructor_embed_dim))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.time_embed_dim = (reconstructor_embed_dim//num_heads)//2
        self.time_embed = RotaryEmbedding(dim=self.time_embed_dim, interpolate_factor=interpolate_factor)
        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), reconstructor_embed_dim)
        # --
        self.reconstructor_blocks = nn.ModuleList([
            Block(
                dim=reconstructor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, is_causal=False, use_rope=True, 
                return_attention=(i+1)==return_attention_layer)
            for i in range(depth)])
        self.reconstructor_norm = norm_layer(reconstructor_embed_dim)
        self.reconstructor_proj = nn.Linear(reconstructor_embed_dim, patch_size, bias=True)
        # ------
        self.init_std = init_std


    def get_num_layers(self):
        return len(self.reconstructor_blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed', 'chan_embed'}

    def forward(self, x, use_out_proj=False):
        # -- map from encoder-dim to pedictor-dim
        if self.use_inp_embed:
            x = self.reconstructor_embed(x)

        C, N        = self.num_patches
        B, mN, eN, D= x.shape
        
        # assert mN == N, f"{mN},{N}"
        # -- get freqs for RoPE
        freqs_x      = self.time_embed.prepare_freqs((eN, N), x.device, x.dtype) # NC, time_dim
        freqs_y      = self.time_embed.prepare_freqs((1, 1), x.device, x.dtype) # NC, time_dim
        
        y = self.cls_token.repeat((B, 1, 1))
        
        if self.use_pos_embed:
            x        = x + self.pos_embed.repeat((B, x.shape[1], 1, 1)).to(x.device)
            
        # -- concat query mask_token ys
        x           = x.flatten(1,2) # B N E D -> B NE D
        x           = torch.cat([y, x], dim=1)
        freqs_x     = torch.cat([freqs_y, freqs_x], dim=0).to(x)
        
        
        # -- fwd prop
        for blk in self.reconstructor_blocks:
            x = blk(x, freqs_x) # B, NC, D
            if blk.return_attention==True: return x
        
        if use_out_proj:
            x = self.reconstructor_norm(x) 
                
            x = self.reconstructor_proj(x)
        
        return x
    
    
class EEGTransformerPredictor(nn.Module):
    """ EEG Transformer """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        embed_num=1,
        use_pos_embed = False,
        use_inp_embed = True,
        use_part_pred = False,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        interpolate_factor = 2.,
        return_attention_layer=-1,
        **kwargs
    ):
        super().__init__()
        self.use_part_pred = use_part_pred
        self.use_pos_embed = use_pos_embed
        self.use_inp_embed = use_inp_embed
        self.num_patches = num_patches
        self.embed_num = embed_num
        
        # --
        self.cls_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        # --
        if use_inp_embed:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        
        if use_pos_embed:
            self.pos_embed   = nn.Parameter(torch.zeros(1, 1, embed_num, predictor_embed_dim))
            trunc_normal_(self.pos_embed, std=init_std)
        
        self.mask_token      = nn.Parameter(torch.zeros(1, 1, embed_num, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.time_embed_dim = (predictor_embed_dim//num_heads)//2
        self.time_embed = RotaryEmbedding(dim=self.time_embed_dim, interpolate_factor=interpolate_factor)
        
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, is_causal=False, use_rope=True, 
                return_attention=(i+1)==return_attention_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)

    def get_num_layers(self):
        return len(self.predictor_blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed', 'chan_embed'}

    def forward(self, x, use_out_proj=False):
        # conditions: (Nq, D) as qurey for downstream 
        # mask_t: mN one number index like (n*C+c) in matrix (N,1)
        
        # -- map from encoder-dim to pedictor-dim
        if self.use_part_pred:
            inp_x = x
            
        if self.use_inp_embed:
            x = self.predictor_embed(x)

        C, N        = self.num_patches
        B, mN, eN, D    = x.shape
        
        ############## Mask x ###############
        # -- get freqs for RoPE
        freqs_x      = self.time_embed.prepare_freqs((eN, N), x.device, x.dtype) # NC, time_dim
        freqs_y      = self.time_embed.prepare_freqs((1, 1), x.device, x.dtype) # NC, time_dim
        
        y = self.cls_token.repeat((B, 1, 1))

        if self.use_pos_embed:
            x                = x + self.pos_embed.repeat((B, x.shape[1], 1, 1)).to(x.device)
            
        # -- concat query mask_token ys
        x           = x.flatten(1,2) # B N E D -> B NE D
        x           = torch.cat([y, x], dim=1)
        freqs_x     = torch.cat([freqs_y, freqs_x], dim=0).to(x)
        
        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x, freqs_x) # B, NC, D
            if blk.return_attention==True: return x
        
        # -- reshape back
        # x = x.reshape((B, N, eN, D))
        
        if use_out_proj:
            x = self.predictor_norm(x) 
                
            x = self.predictor_proj(x)
        return x
    
           
class EEGTransformer(nn.Module):
    """ EEG Transformer """
    def __init__(
        self,
        img_size=(64,1000),
        patch_size=64,
        patch_stride=None,
        embed_dim=768,
        embed_num=1,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_module=PatchEmbed,# PatchNormEmbed
        init_std=0.02,
        interpolate_factor = 2.,
        return_attention_layer=-1,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.embed_num = embed_num
        
        self.num_heads = num_heads
        
        # --
        self.patch_embed = patch_module(
            img_size=img_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        # --
        
        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), embed_dim)
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                is_causal=False, use_rope= False, return_attention=(i+1)==return_attention_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))
            
        trunc_normal_(self.summary_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        
    def prepare_chan_ids(self, channels):
        chan_ids = []
        for ch in channels:
            ch = ch.upper().strip('.')
            assert ch in CHANNEL_DICT, ch
            chan_ids.append(CHANNEL_DICT[ch])
        return torch.tensor(chan_ids).unsqueeze_(0).long()
    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'chan_embed', 'summary_token'}
    
    def forward(self, x, chan_ids=None, mask_x=None, mask_t=None):
        # x.shape B, C, T
        # mask_x.shape mN, mC
        # mask_t.shape mN
        
        # -- patchify x
        x = self.patch_embed(x) #
        B, N, C, D = x.shape
        
        assert N==self.num_patches[1] and C==self.num_patches[0], f"{N}=={self.num_patches[1]} and {C}=={self.num_patches[0]}"
        
        if chan_ids is None:
            chan_ids = torch.arange(0,C)     
        chan_ids = chan_ids.to(x)
        
        # -- add channels positional embedding to x
        x = x + self.chan_embed(chan_ids.long()).unsqueeze(0) # (1,C) -> (1,1,C,D)
        
        if mask_x is not None:
            mask_x = mask_x.to(x.device)
            x = apply_mask(mask_x, x)# B, mN, mC, D
            B, N, C, D = x.shape
            
        
        x = x.flatten(0, 1) # BmN, mC, D
        
        # -- concat summary token
        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([x,summary_token], dim=1)  # BmN, mC+embed_num, D
        
        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x) # B*N, mC+1, D
            if blk.return_attention==True: return x

        x = x[:, -summary_token.shape[1]:, :]
        
        if self.norm is not None:
            x = self.norm(x) 

        
        x = x.flatten(-2)
        x = x.reshape((B, N, -1))
        # -- reshape back
            
        if mask_t is not None:
            mask_t = mask_t.to(x.device)
            x = apply_mask_t(mask_t, x)# B, mN, D        
        
        x = x.reshape((B, N, self.embed_num, -1))
        
        return x


class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)
        
    @autocast(True)
    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)
    
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    @autocast(True)
    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)
def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    # print(x.shape)
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

class EEGPTClassifier(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels = 22,
                 img_size=[58, 256*4], 
                 patch_stride=64, 
                 use_channels_names=None,
                 use_mean_pooling=True, 
                 norm_layer=nn.LayerNorm, 
                 use_chan_conv=False,
                 max_norm_chan_conv = 1,
                 max_norm_head = 1,
                 
                 qkv_bias=True,
                 
                 enc_drop_rate=0.0,
                 enc_attn_drop_rate=0.0,
                 enc_drop_path_rate=0.0,
                 
                 rec_drop_rate=0.0,
                 rec_attn_drop_rate=0.0,
                 rec_drop_path_rate=0.0,
                 
                 use_freeze_encoder=False,
                 use_freeze_reconstructor=False,
                 
                 interpolate_factor = 2.,
                 desired_time_len = 200 * 10,
                 use_avg = False,
                 
                 use_predictor = False,
                 use_out_proj = False,
                 **kwargs):
        
        super().__init__()    
        self.num_classes = num_classes
        self.max_norm_chan_conv = max_norm_chan_conv
        self.max_norm_head = max_norm_head
        self.desired_time_len = desired_time_len
        self.use_avg = use_avg
        self.use_out_proj = use_out_proj
        self.use_chan_conv = use_chan_conv
        if use_chan_conv:
            self.chan_conv      = torch.nn.Sequential(
                Conv1dWithConstraint(in_channels, img_size[0], 1, max_norm=max_norm_chan_conv, doWeightNorm=(max_norm_chan_conv>0)),
                # nn.BatchNorm1d(self.chans_num, track_running_stats=False)
            )
        
        target_encoder = EEGTransformer(
            img_size=img_size,
            patch_size= 32*2,
            patch_stride=patch_stride,
            embed_dim = 512,
            embed_num = 4,
            depth     = 8,
            num_heads = 8,
            mlp_ratio =4.0,
            drop_rate =enc_drop_rate,
            attn_drop_rate=enc_attn_drop_rate,
            drop_path_rate=enc_drop_path_rate,
            init_std=0.02,
            qkv_bias=qkv_bias, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        if use_freeze_encoder:
            for p in target_encoder.parameters():
                p.requires_grad_(False)
        
        self.use_predictor = use_predictor
        if not use_predictor:
            reconstructor = EEGTransformerReconstructor(
                num_patches=target_encoder.num_patches,
                patch_size             =32*2,
                embed_dim              =512,
                embed_num              =4,
                reconstructor_embed_dim=512,
                depth                  =8,
                num_heads              =8,
                mlp_ratio               =4.0,
                
                interpolate_factor      = interpolate_factor,
                drop_rate               =rec_drop_rate,
                attn_drop_rate          =rec_attn_drop_rate,
                drop_path_rate          =rec_drop_path_rate,
                init_std                =0.02,
                qkv_bias                =qkv_bias, 
                norm_layer              =partial(nn.LayerNorm, eps=1e-6))
        else:
            reconstructor = EEGTransformerPredictor(
                num_patches             =target_encoder.num_patches,
                embed_dim               =512,
                embed_num               =4,
                use_part_pred           =True,
                predictor_embed_dim     =512,
                depth                   =8,
                num_heads               =8,
                mlp_ratio               =4.0,
                
                interpolate_factor      = interpolate_factor,
                drop_rate               =rec_drop_rate,
                attn_drop_rate          =rec_attn_drop_rate,
                drop_path_rate          =rec_drop_path_rate,
                init_std                =0.02,
                qkv_bias                =qkv_bias, 
                norm_layer              =partial(nn.LayerNorm, eps=1e-6))
            
        
        if use_freeze_reconstructor:
            for p in reconstructor.parameters():
                p.requires_grad_(False)
            reconstructor.cls_token.requires_grad_(True)
        
        self.target_encoder = target_encoder
        
        if not self.use_predictor:
            self.reconstructor  = reconstructor
        else:
            self.predictor  = reconstructor
            
        self.chans_id       = target_encoder.prepare_chan_ids(use_channels_names)
        
        if (not use_predictor) and use_out_proj:
            embed_dim = 64
        else:
            embed_dim = 512
        self.embed_dim = embed_dim
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = LinearWithConstraint(embed_dim, num_classes, max_norm=max_norm_head, doWeightNorm=max_norm_head>0) if num_classes > 0 else nn.Identity()
        # if max_norm_head>0:
            
        # else:
        #     self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def get_num_layers(self):
        if self.use_predictor:
            return self.target_encoder.get_num_layers() + self.predictor.get_num_layers()
        else:
            return self.target_encoder.get_num_layers() + self.reconstructor.get_num_layers()
    
    def get_classifier(self):
        return self.head
    
    @torch.jit.ignore
    def no_weight_decay(self):
        if self.use_predictor:
            return set(["target_encoder."+x for x in self.target_encoder.no_weight_decay()] + \
                        ["predictor."+x for x in self.predictor.no_weight_decay()])
        else:
            return set(["target_encoder."+x for x in self.target_encoder.no_weight_decay()] + \
                        ["reconstructor."+x for x in self.reconstructor.no_weight_decay()])

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = LinearWithConstraint(self.embed_dim, num_classes, max_norm=self.max_norm_head, doWeightNorm=self.max_norm_head>0) if self.num_classes > 0 else nn.Identity()

    def forward_features(self, x, chan_ids=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        if chan_ids is None:
            chan_ids = self.chans_id
        
        if self.use_chan_conv:
            x = self.chan_conv(x)
            
        x = self.target_encoder(x, chan_ids.to(x))
        
        if not self.use_predictor:
            x = self.reconstructor(x, self.use_out_proj)
        else:
            x = self.predictor(x, self.use_out_proj)
        
        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]
    def forward(self, x, chan_ids=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        '''
        x: [batch size, number of electrodes, Times]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4*256]
        '''
        if len(x.shape)==4: x = x.flatten(2)
        if (self.desired_time_len>0) and self.desired_time_len != x.shape[-1]:
            x = temporal_interpolation(x, self.desired_time_len, use_avg=self.use_avg)
        x = self.forward_features(x, chan_ids=chan_ids, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x)
        return x
    
    # def load_state_dict(self, state_dict, strict: bool = False):
    #     return super().load_state_dict(state_dict, strict)
        
if __name__=="__main__":
    use_channels_names = [      
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2' ]
    ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
    # use_channels_names = ch_names
    model = EEGPTClassifier(4, in_channels=len(ch_names), img_size=[len(use_channels_names),2000], use_channels_names=use_channels_names, use_chan_conv=True, use_predictor=True)
    
    x = torch.zeros((2,len(ch_names),2000))
    with torch.no_grad():
        z = model(x)
        print(z.shape)
    # target_encoder = EEGTransformer(
    # img_size        =[len(use_channels_names), 1024],
    # patch_size      =32*2,
    # embed_num       =4,
    # embed_dim       =512,
    # depth           =8,
    # num_heads       =8,
    # mlp_ratio       =4.0,
    # drop_rate       =0.0,
    # attn_drop_rate  =0.0,
    # drop_path_rate  =0.0,
    # init_std        =0.02,
    # qkv_bias        =True, 
    # norm_layer      =partial(nn.LayerNorm, eps=1e-6))
    
    # reconstructor = EEGTransformerReconstructor(
    # num_patches            =target_encoder.num_patches,
    # patch_size             =32*2,
    # embed_dim              =512,
    # embed_num              =4,
    # reconstructor_embed_dim=512,
    # depth                  =8,
    # num_heads              =8,
    # mlp_ratio              =4.0,
    # drop_rate              =0.0,
    # attn_drop_rate         =0.0,
    # drop_path_rate         =0.0,
    # init_std               =0.02,
    # qkv_bias               =True, 
    # norm_layer             =partial(nn.LayerNorm, eps=1e-6))
    
    # x = torch.zeros((2,19,1024))
    # chans_id = target_encoder.prepare_chan_ids(use_channels_names)
    # with torch.no_grad():
    #     z = target_encoder(x, chans_id.to(x))
    #     r = reconstructor(z)