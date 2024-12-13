
import math
import torch
from torch import nn
from einops import rearrange

from .conv_ae import ConvEncoderBlock, ConvDecoderBlock

class ConvMatryoshkaAE(nn.Module):
    def __init__(
        self, in_channels, out_channels, spatial_res, hidden_dim, bottleneck_dim, split_strategy="log", **kwargs
    ):
        super(ConvMatryoshkaAE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_res = spatial_res
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.split_strategy = split_strategy
        
        self.split_dims = self.get_split_dims()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=4, stride=2,
                    padding=1),  # (384, 56, 56) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=False),
            ConvEncoderBlock(hidden_dim, hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 14, 14)
            ConvEncoderBlock(hidden_dim, hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 14, 14) -> (hidden_dim, 7, 7)  # (hidden_dim, 7, 7) -> (hidden_dim, 4, 4)
        )
        # self.bottleneck = nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck_dim, kernel_size=8, padding=1) # (hidden_dim, 7, 7) -> (hidden_dim, 1, 1)
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck_dim, kernel_size=1, padding=0)
        )
        
        self.decoder_list = nn.ModuleList()
        for i in range(len(self.split_dims)):
            self.decoder_list.append(
                nn.Sequential(
                    ConvDecoderBlock(in_channels=self.split_dims[i], outchannels=hidden_dim, upsample_size=4, upsample_mode='bilinear', kernel_size=3, stride=1, padding=2, dropout=0.2),  # (bottleneck_dim, 1, 1) -> (hidden_dim, 5, 5)
                    ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=8, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 5, 5) -> (hidden_dim, 9, 9)
                    ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=16, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 9, 9) -> (hidden_dim, 17, 17)
                    ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=32, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 17, 17) -> (hidden_dim, 33, 33)
                    ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=64, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 33, 33) -> (hidden_dim, 65, 65)
                    ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=127, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 65, 65) -> (hidden_dim, 129, 129)
                    ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=56, upsample_mode='bilinear', kernel_size=3, stride=1, padding=1, dropout=0.2),  # (hidden_dim, 129, 129) -> (hidden_dim, 56, 56)
                )
            )
        self.out_proj = nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3,
                    stride=1, padding=1)  # (hidden_dim, 56, 56) -> (out_channels, 56, 56)
        
    def get_split_dims(self):
        if self.split_strategy == "log":
            return self.get_log_split_dims()
        elif self.split_strategy == "linear":
            return self.get_linear_split_dims()
        else:
            raise ValueError(f"Invalid split strategy: {self.split_strategy}")
        
    def get_log_split_dims(self):
        split_dims = [] 
        assert math.log2(self.bottleneck_dim) % 1 == 0, "Bottleneck dimension must be a power of 2"
        for i in range(1, int(math.log2(self.bottleneck_dim)) + 1):
            split_dims.append(2 ** i)
        return split_dims
    
    def get_linear_split_dims(self, num_splits=4):
        split_dims = []
        split_dim = self.bottleneck_dim // num_splits
        for i in range(1, num_splits + 1):
            split_dims.append(split_dim * i)
        return split_dims
    
    def forward(self, x, return_dict=False):
        org_x = x
        x = self.encoder(x)
        x = self.bottleneck(x)  # (B, bottleneck_dim, 1, 1)

        mat_embs = []
        preds = []
        for i, dims in enumerate(self.split_dims):
            mat_emb = x[:, :dims, :, :]
            out = self.decoder_list[i](mat_emb)
            mat_embs.append(mat_emb)
            preds.append(self.out_proj(out))
        
        if return_dict:
            return {
                "mat_embs": mat_embs,
                "preds": preds,
                "org_x": org_x
            }
        else:
            return preds
    
    def calculate_loss(self, x, preds, weights=None):
        """Calculate loss for MatryoshkaAE.
        Args:
            x (torch.Tensor): Input tensor (B, C, H, W)
            preds (List[torch.Tensor]): List of predicted tensors (B, C, H, W)
        Returns:
            torch.Tensor: Loss tensor
        """
        N = len(preds)
        assert N == len(self.split_dims), "Number of predictions must match number of split dims"
        
        if weights is None:
            weights = [1.0] * N
        else:
            assert len(weights) == N, "Number of weights must match number of predictions"
        
        loss = 0
        for i in range(N):
            loss += weights[i] * nn.MSELoss()(x, preds[i])
        return loss
    
def get_mat_ae(**kwargs):
    return ConvMatryoshkaAE(**kwargs)    

if __name__ == "__main__":
    model = ConvMatryoshkaAE(in_channels=3, out_channels=3, spatial_res=56, hidden_dim=128, bottleneck_dim=256)
    x = torch.randn(2, 3, 56, 56)
    preds = model(x)
    loss = model.calculate_loss(x, preds)
    