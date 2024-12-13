from .vision_transformer import PatchEmbed, PosEmbedding, FeedForwardBlock, MultiHeadAttentionBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def get_vitae(**kwargs):
    in_channels = kwargs.pop('in_channels', 3)
    out_channels = kwargs.pop('out_channels', 384)
    spatial_res = kwargs.pop('spatial_res', 1)
    hidden_dim = kwargs.pop('hidden_dim', 64)
    bottleneck_dim = kwargs.pop('bottleneck_dim', 64)
    
    patch_size = kwargs.pop('patch_size', 2)
    num_enc_layers = kwargs.pop('num_enc_layers', 3)
    num_dec_layers = kwargs.pop('num_dec_layers', 3)
    num_heads = kwargs.pop('num_heads', 8)
    mlp_ratio = kwargs.pop('mlp_ratio', 4)
    
    in_res = kwargs.pop('in_res', 56)
    
    return VisionTransformerAutoencoder(in_res=in_res, in_channels=in_channels, out_channels=out_channels, patch_size=patch_size, spatial_res=spatial_res, \
        hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim, num_enc_layers=num_enc_layers, num_dec_layers=num_dec_layers, num_heads=num_heads, mlp_ratio=mlp_ratio, \
    **kwargs)

class VisionTransformerAutoencoder(nn.Module):
    def __init__(self, in_res, in_channels, patch_size, hidden_dim, bottleneck_dim, spatial_res, num_enc_layers, num_dec_layers, num_heads, mlp_ratio, residual=False, **kwargs):
        super(VisionTransformerAutoencoder, self).__init__()
        
        self.in_res = in_res
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.spatial_res = spatial_res
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.residual = residual
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.num_patches = (in_res // patch_size) ** 2
        self.patch_embed = PatchEmbed(in_channels, patch_size, hidden_dim)
        self.pos_embed = PosEmbedding(hidden_dim, self.num_patches)
        
        self.encoder = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttentionBlock(hidden_dim, num_heads),
                FeedForwardBlock(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim)
            ])
            for _ in range(num_enc_layers)
        ])
        
        self.bottleneck = nn.Linear(hidden_dim, bottleneck_dim)
        
        self.decoder_pos_embed = PosEmbedding(bottleneck_dim, self.num_patches)
        self.decoder_embed = nn.Linear(bottleneck_dim, hidden_dim)
        self.decoder_token = nn.Parameter(torch.randn(1, bottleneck_dim))
        self.decoder = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttentionBlock(hidden_dim, num_heads),
                FeedForwardBlock(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim)
            ])
            for _ in range(num_dec_layers)
        ])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor of shape (b, c, h, w)
        Returns:
            torch.Tensor: tensor of shape (b, c, h, w)
        """
        
        # patch embedding
        x = self.patch_embed(x)  # (b, n, e)
        
        # positional embedding
        x = self.pos_embed(x)
        
        # encoder
        for attn, ffn in self.encoder:
            # Multi-head attention
            if self.residual:
                residual = x
                x, _ = attn(self.layer_norm(x))
                x = x + residual
                
                residual = x
                x = ffn(self.layer_norm(x))
                x = x + residual
            else:
                x, _ = attn(self.layer_norm(x))
                x = ffn(self.layer_norm(x))
        
        # bottleneck
        x = self.bottleneck(x)  # (b, n, bottleneck_dim)
        # spatial reduction
        h, w = self.in_res // self.patch_size, self.in_res // self.patch_size
        x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)  # (b, bottleneck_dim, h, w)
        assert h % self.spatial_res == 0 and w % self.spatial_res == 0, f"Invalid spatial resolution: {h}, {w}"
        ksize = h // self.spatial_res
        x = F.avg_pool2d(x, kernel_size=ksize, stride=ksize)  # (b, bottleneck_dim, h', w')
        h_, w_ = x.size(2), x.size(3)
        x = rearrange(x, 'b d h w -> b (h w) d', h=h_, w=w_)  # (b, n', bottleneck_dim)
        
        # We use learnable token for spatial expansion. 
        x_rec = self.decoder_token.unsqueeze(1).repeat(1, self.num_patches, 1)  # (1, n, bottleneck_dim)
        x_rec = x_rec.repeat(x.size(0), 1, 1)  # (b, n, bottleneck_dim)
        x_rec = self.decoder_pos_embed(x_rec)
        
        x_cat = torch.cat([x_rec, x], dim=1)  # (b, n + n', bottleneck_dim)
        x_cat = self.decoder_embed(x_cat)  # (b, n + n', hidden_dim)
        
        # decoder, compressed features intaracts with the learnable tokens
        for attn, ffn in self.decoder:
            if self.residual:
                residual = x_cat
                x_cat, _ = attn(self.layer_norm(x_cat))
                x_cat = x_cat + residual
                
                residual = x_cat
                x_cat = ffn(self.layer_norm(x_cat))
                x_cat = x_cat + residual
            else:
                x_cat, _ = attn(self.layer_norm(x_cat))
                x_cat = ffn(self.layer_norm(x_cat))
        
        # extract reconstructed features
        x_rec = x_cat[:, :self.num_patches]  # (b, n, hidden_dim)
        x_rec = rearrange(x_rec, 'b (h w) d -> b d h w', h=h, w=w)
        return x_rec
        
        
        
        
        
        