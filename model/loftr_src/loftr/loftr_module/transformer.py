import copy
from typing import Optional
from mamba_ssm.modules.mamba_simple import Mamba
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention
from .vmamba import VSSBlock
from einops import rearrange
import math

class TokenDownLayer(nn.Module):
    def __init__(self, shape1,shape2) -> None:
        super().__init__()
        self.dwn = nn.Sequential(
            nn.AdaptiveAvgPool2d((shape1,shape2))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.dwn(x)

        return x

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask: Optional[torch.Tensor] = None, source_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()
        self.block_type = config['block_type']
        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.config['block_type'] = config['block_type']
        
        if config['block_type'] == 'loftr':
            encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
            self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
            self._reset_parameters()
        elif config['block_type'] == 'mamba':
            dpr = [x.item() for x in torch.linspace(0, 0.1, 8)]
            # encoder_layer = VSSBlock(
            #                     hidden_dim=256, 
            #                     drop_path=0.1,)
            self.layers = nn.ModuleList([copy.deepcopy(VSSBlock(
                                hidden_dim=256, 
                                d_model = config['d_model'],nhead = config['nhead'],
                                drop_path=dpr[i],)) for i in range(len(self.layer_names))])



    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0: Optional[torch.Tensor] = None, mask1: Optional[torch.Tensor] = None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        
        if len(feat0.shape) == 4 :
            B, C, H, W = feat0.shape

            
            feat0 = rearrange(feat0, 'b c h w -> b (h w) c')
            feat1 = rearrange(feat1, 'b c h w -> b (h w) c')
        else:
            H0, W0, H1, W1 = 0, 0, 0, 0

        if self.block_type == 'loftr':
            for layer, name in zip(self.layers, self.layer_names):
                if name == 'self':
                    feat0 = layer(feat0, feat0, mask0, mask0)
                    feat1 = layer(feat1, feat1, mask1, mask1)
                elif name == 'cross':
                    feat0 = layer(feat0, feat1, mask0, mask1)
                    feat1 = layer(feat1, feat0, mask1, mask0)
                else:
                    raise KeyError
       
        # elif self.block_type == 'mamba':
        #     for layer, name in zip(self.layers, self.layer_names):
        #         feat0=rearrange(feat0, 'b h w c-> b (h w) c')
        #         feat1=rearrange(feat1, 'b h w c-> b (h w) c')
        #         if name == 'self':
        #             feat0 = layer(feat0, feat0, H1, W1, H1, W1)
        #             feat1 = layer(feat1, feat1, H2, W2, H2, W2)
        #         elif name == 'cross':
        #             feat0 = layer(feat0, feat1, H1, W1, H2, W2)
        #             feat1 = layer(feat1, feat0, H2, W2, H1, W1)
        #     feat0 = rearrange(feat0, 'b h w c-> b (h w) c')
        #     feat1 = rearrange(feat1, 'b h w c-> b (h w) c')
        elif self.block_type == 'mamba':


            feat0=rearrange(feat0, 'b c h w -> b h w c')
            feat1=rearrange(feat1, 'b c h w -> b h w c')
            for layer, name in zip(self.layers, self.layer_names):
                if name == 'self':
                    feat0 = layer(feat0, feat0, H, W)
                    feat1 = layer(feat1, feat1, H, W)
                elif name == 'cross':
                    feat0 = layer(feat0, feat1, H, W)
                    feat1 = layer(feat1, feat0, H, W)
            feat0 = rearrange(feat0, 'b h w c-> b (h w) c')
            feat1 = rearrange(feat1, 'b h w c-> b (h w) c')
        return feat0, feat1
