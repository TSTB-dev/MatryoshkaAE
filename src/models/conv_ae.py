import torch
from torch import nn

class ConvEncoderBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvEncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)  # w/o spatial reduction
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu2 = nn.ReLU(inplace=False)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

class ConvDecoderBlock(nn.Module):
    def __init__(self, in_channels, outchannels, upsample_size, upsample_mode='bilinear', \
        kernel_size=3, stride=1, padding=1, dropout=0.2):
        super(ConvDecoderBlock, self).__init__()
        self.upsample = nn.Upsample(size=upsample_size, mode=upsample_mode)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=outchannels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

def get_convae(**kwargs):
    in_channels = kwargs.pop('in_channels', 3)
    out_channels = kwargs.pop('out_channels', 384)
    spatial_res = kwargs.pop('spatial_res', 1)
    hidden_dim = kwargs.pop('hidden_dim', 64)
    bottleneck_dim = kwargs.pop('bottleneck_dim', 64)
    
    if spatial_res==1:
        # model = nn.Sequential(
        #     # Input conv
        #     nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=4, stride=2,
        #             padding=1),  # (384, 56, 56) -> (hidden_dim, 28, 28)
        #     nn.ReLU(inplace=False),
            
        #     # Encoder
        #     ConvEncoderBlock(in_channels=hidden_dim, hidden_dim=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 14, 14)
        #     ConvEncoderBlock(in_channels=hidden_dim, hidden_dim=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 14, 14) -> (hidden_dim, 7, 7)

        #     # bottleneck
        #     nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck_dim, kernel_size=8, padding=1),  # (hidden_dim, 7, 7) -> (hidden_dim, 1, 1)
            
        #     # decoder
        #     ConvDecoderBlock(in_channels=bottleneck_dim, outchannels=hidden_dim, upsample_size=4, upsample_mode='bilinear', kernel_size=3, stride=1, padding=2, dropout=0.2),  # (bottleneck_dim, 1, 1) -> (hidden_dim, 5, 5)
        #     ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=8, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 5, 5) -> (hidden_dim, 9, 9)
        #     ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=16, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 9, 9) -> (hidden_dim, 17, 17)
        #     ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=32, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 17, 17) -> (hidden_dim, 33, 33)
        #     ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=64, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 33, 33) -> (hidden_dim, 65, 65)
        #     ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=127, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 65, 65) -> (hidden_dim, 129, 129)
        #     ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=56, upsample_mode='bilinear', kernel_size=3, stride=1, padding=1, dropout=0.2),  # (hidden_dim, 129, 129) -> (hidden_dim, 56, 56)
            
        #     nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3,
        #             stride=1, padding=1)  # (hidden_dim, 56, 56) -> (out_channels, 56, 56)
        # )
        # return model
        model = nn.Sequential(
            # Input conv
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1,
                    padding=0),  
            nn.ReLU(inplace=False),
            
            # Encoder
            ConvEncoderBlock(in_channels=hidden_dim, hidden_dim=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),  # (hidden_dim, 28, 28) -> (hidden_dim, 14, 14)
            ConvEncoderBlock(in_channels=hidden_dim, hidden_dim=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),  # (hidden_dim, 14, 14) -> (hidden_dim, 7, 7)
            ConvEncoderBlock(in_channels=hidden_dim, hidden_dim=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            
            # bottleneck
            nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck_dim, kernel_size=1, padding=0),  # (hidden_dim, 7, 7) -> (hidden_dim, 1, 1)
            
            # decoder
            ConvDecoderBlock(in_channels=bottleneck_dim, outchannels=hidden_dim, upsample_size=56, upsample_mode='bilinear', kernel_size=1, stride=1, padding=0, dropout=0.2),  # (hidden_dim, 33, 33) -> (hidden_dim, 65, 65)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=56, upsample_mode='bilinear', kernel_size=1, stride=1, padding=0, dropout=0.2),  # (hidden_dim, 65, 65) -> (hidden_dim, 129, 129)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=56, upsample_mode='bilinear', kernel_size=1, stride=1, padding=0, dropout=0.2),  # (hidden_dim, 129, 129) -> (hidden_dim, 56, 56)
            
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                    stride=1, padding=0)  # (hidden_dim, 56, 56) -> (out_channels, 56, 56)
        )
        return model
    elif spatial_res==2:
        model = nn.Sequential(
            # input conv
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (384, 56, 56) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=False),

            # Encoder
            ConvEncoderBlock(in_channels=hidden_dim, hidden_dim=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 14, 14)
            ConvEncoderBlock(in_channels=hidden_dim, hidden_dim=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 14, 14) -> (hidden_dim, 7, 7)
            
            # bottleneck
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck_dim, kernel_size=4, stride=4, padding=1), # (hidden_dim, 7, 7) -> (botleneck_dim, 2, 2)
        
            # Decoder
            ConvDecoderBlock(in_channels=bottleneck_dim, outchannels=hidden_dim, upsample_size=4, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (botleneck_dim, 2, 2) -> (hidden_dim, 5, 5)    
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=8, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 5, 5) -> (hidden_dim, 9, 9)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=16, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 9, 9) -> (hidden_dim, 17, 17)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=32, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 17, 17) -> (hidden_dim, 33, 33)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=64, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 33, 33) -> (hidden_dim, 65, 65)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=127, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 65, 65) -> (hidden_dim, 129, 129)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=56, upsample_mode='bilinear', kernel_size=3, stride=1, padding=1, dropout=0.2),  # (hidden_dim, 129, 129) -> (hidden_dim, 56, 56)
            
            # output conv
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1)  # (hidden_dim, 56, 56) -> (out_channels, 56, 56)
        )
        return model
            
    elif spatial_res == 4:
        model = nn.Sequential(
            # Input conv
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (384, 56, 56) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=False),
            
            # Encoder 
            ConvEncoderBlock(in_channels=hidden_dim, hidden_dim=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 14, 14)
            ConvEncoderBlock(in_channels=hidden_dim, hidden_dim=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 14, 14) -> (hidden_dim, 7, 7)
            
            # bottleneck
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck_dim, kernel_size=2, stride=2, padding=1), # (hidden_dim, 7, 7) -> (botleneck_dim, 4, 4)
        
            # decoder
            ConvDecoderBlock(in_channels=bottleneck_dim, outchannels=hidden_dim, upsample_size=4, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (botleneck_dim, 4, 4) -> (hidden_dim, 5, 5)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=8, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 5, 5) -> (hidden_dim, 9, 9)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=16, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 9, 9) -> (hidden_dim, 17, 17)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=32, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 17, 17) -> (hidden_dim, 33, 33)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=64, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 33, 33) -> (hidden_dim, 65, 65)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=127, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 65, 65) -> (hidden_dim, 129, 129)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=56, upsample_mode='bilinear', kernel_size=3, stride=1, padding=1, dropout=0.2),  # (hidden_dim, 129, 129) -> (hidden_dim, 56, 56)
            
            # output conv
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1)  # (hidden_dim, 56, 56) -> (out_channels, 56, 56)
        )
        return model
    elif spatial_res==8:
        model = nn.Sequential(
            
            # input conv
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (384, 56, 56) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=False),
            
            # Encoder
            ConvEncoderBlock(in_channels=hidden_dim, hidden_dim=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 14, 14)
            ConvEncoderBlock(in_channels=hidden_dim, hidden_dim=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 14, 14) -> (bottleneck_dim, 7, 7)
            
            # bottleneck
            nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck_dim, kernel_size=3, stride=1, padding=2),  # (bottleneck_dim, 7, 7) -> (bottleneck_dim, 8, 8)
            
            # decoder
            ConvDecoderBlock(in_channels=bottleneck_dim, outchannels=hidden_dim, upsample_size=8, upsample_mode='bilinear', kernel_size=3, stride=1, padding=1, dropout=0.2),  # (bottleneck_dim, 8, 8) -> (hidden_dim, 8, 8)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=8, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 8, 8) -> (hidden_dim, 9, 9)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=16, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 9, 9) -> (hidden_dim, 17, 17)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=32, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 17, 17) -> (hidden_dim, 33, 33)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=64, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 33, 33) -> (hidden_dim, 65, 65)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=127, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 65, 65) -> (hidden_dim, 129, 129)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=56, upsample_mode='bilinear', kernel_size=3, stride=1, padding=1, dropout=0.2),  # (hidden_dim, 129, 129) -> (hidden_dim, 56, 56)
            
            # output conv
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1)  # (hidden_dim, 56, 56) -> (out_channels, 56, 56)
        )
        return model
    elif spatial_res == 16:
        model = nn.Sequential(
            # input conv
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (384, 56, 56) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=False),
            
            # Encoder   
            ConvEncoderBlock(in_channels=hidden_dim, hidden_dim=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 14, 14)
            ConvEncoderBlock(in_channels=hidden_dim, hidden_dim=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),  # (hidden_dim, 14, 14) -> (bottleneck_dim, 16, 16)
            
            # bottleneck
            nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck_dim, kernel_size=3, stride=1, padding=1),  # (hidden_dim, 16, 16) -> (hidden_dim, 16, 16)
            nn.ReLU(inplace=False),
            
            # decoder
            ConvDecoderBlock(in_channels=bottleneck_dim, outchannels=hidden_dim, upsample_size=16, upsample_mode='bilinear', kernel_size=3, stride=1, padding=1, dropout=0.2),  # (bottleneck_dim, 16, 16) -> (hidden_dim, 16, 16)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=16, upsample_mode='bilinear', kernel_size=3, stride=1, padding=1, dropout=0.2),  # (bottleneck_dim, 16, 16) -> (hidden_dim, 16, 16)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=16, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (bottleneck_dim, 16, 16) -> (hidden_dim, 16, 16)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=32, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 16, 16) -> (hidden_dim, 33, 33)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=64, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 33, 33) -> (hidden_dim, 65, 65)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=127, upsample_mode='bilinear', kernel_size=4, stride=1, padding=2, dropout=0.2),  # (hidden_dim, 65, 65) -> (hidden_dim, 129, 129)
            ConvDecoderBlock(in_channels=hidden_dim, outchannels=hidden_dim, upsample_size=56, upsample_mode='bilinear', kernel_size=3, stride=1, padding=1, dropout=0.2),  # (hidden_dim, 129, 129) -> (hidden_dim, 56, 56)
            
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1)  # (hidden_dim, 56, 56) -> (out_channels, 56, 56)
        )
        return model
            
            

