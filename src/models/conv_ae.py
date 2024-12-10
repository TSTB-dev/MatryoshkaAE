import torch
from torch import nn

def get_convae(**kwargs):
    in_channels = kwargs.pop('in_channels', 3)
    out_channels = kwargs.pop('out_channels', 384)
    spatial_res = kwargs.pop('spatial_res', 1)
    hidden_dim = kwargs.pop('hidden_dim', 64)
    bottleneck_dim = kwargs.pop('bottleneck_dim', 64)
    
    if spatial_res==1:
        model = nn.Sequential(
            # encoder
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=4, stride=2,
                    padding=1),  # (384, 56, 56) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1,
                    padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2,
                    padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1,
                    padding=1),  # (hidden_dim, 14, 14) -> (hidden_dim, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2,
                    padding=1),  # (hidden_dim, 14, 14) -> (hidden_dim, 7, 7)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck_dim, kernel_size=8, padding=1),  # (hidden_dim, 7, 7) -> (hidden_dim, 1, 1)
            
            # decoder
            nn.Upsample(size=3, mode='bilinear'),  # (bottleneck_dim, 1, 1) -> (bottleneck_dim, 3, 3)
            nn.Conv2d(in_channels=bottleneck_dim, out_channels=hidden_dim, kernel_size=4, stride=1,
                    padding=2),  # (bottleneck_dim, 3, 3) -> (hidden_dim, 4, 4)
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Upsample(size=8, mode='bilinear'),  # (hidden_dim, 4, 4) -> (hidden_dim, 8, 8)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1,
                    padding=2),  # (hidden_dim, 8, 8) -> (hidden_dim, 10, 10)
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Upsample(size=15, mode='bilinear'),  # (hidden_dim, 10, 10) -> (hidden_dim, 15, 15)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1,
                    padding=2),  # (hidden_dim, 15, 15) -> (hidden_dim, 17, 17)
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Upsample(size=32, mode='bilinear'),  # (hidden_dim, 17, 17) -> (hidden_dim, 32, 32)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1,
                    padding=2),  # (hidden_dim, 32, 32) -> (hidden_dim, 34, 34)
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Upsample(size=63, mode='bilinear'),  # (hidden_dim, 34, 34) -> (hidden_dim, 63, 63)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1,
                    padding=2),  # (hidden_dim, 63, 63) -> (hidden_dim, 65, 65)
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Upsample(size=127, mode='bilinear'),  # (hidden_dim, 65, 65) -> (hidden_dim, 127, 127)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1,
                    padding=2),  # (hidden_dim, 127, 127) -> (hidden_dim, 129, 129)
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Upsample(size=56, mode='bilinear'),  # (hidden_dim, 129, 129) -> (hidden_dim, 56, 56)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1,
                    padding=1),  # (hidden_dim, 56, 56) -> (hidden_dim, 56, 56)
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3,
                    stride=1, padding=1)  # (hidden_dim, 56, 56) -> (out_channels, 56, 56)
        )
        return model
    elif spatial_res==2:
        model = nn.Sequential(
            # Encoder
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (384, 56, 56) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),  # (hidden_dim, 14, 14) -> (hidden_dim, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 14, 14) -> (hidden_dim, 7, 7)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck_dim, kernel_size=4, stride=4, padding=1), # (hidden_dim, 7, 7) -> (botleneck_dim, 2, 2)
        
            # Decoder
            nn.Upsample(size=2, mode='bilinear'),  # (botleneck_dim, 2, 2) -> (botleneck_dim, 4, 4)
            nn.Conv2d(in_channels=bottleneck_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (botleneck_dim, 4, 4) -> (hidden_dim, 5, 5)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=8, mode='bilinear'),  # (hidden_dim, 5, 5) -> (hidden_dim, 10, 10)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 10, 10) -> (hidden_dim, 11, 11)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=15, mode='bilinear'),  # (hidden_dim, 11, 11) -> (hidden_dim, 15, 15)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 15, 15) -> (hidden_dim, 16, 16)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=32, mode='bilinear'),  # (hidden_dim, 16, 16) -> (hidden_dim, 32, 32)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 32, 32) -> (hidden_dim, 33, 33)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=63, mode='bilinear'),  # (hidden_dim, 33, 33) -> (hidden_dim, 63, 63)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 63, 63) -> (hidden_dim, 64, 64)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=127, mode='bilinear'),  # (hidden_dim, 64, 64) -> (hidden_dim, 127, 127)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 127, 127) -> (hidden_dim, 128, 128)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=56, mode='bilinear'),  # (hidden_dim, 128, 128) -> (hidden_dim, 56, 56)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),  # (hidden_dim, 56, 56) -> (hidden_dim, 56, 56)
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1)  # (hidden_dim, 56, 56) -> (out_channels, 56, 56)
        )
        return model
            
    elif spatial_res == 4:
        model = nn.Sequential(
            # encoder
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (384, 56, 56) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),  # (hidden_dim, 14, 14) -> (hidden_dim, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 14, 14) -> (hidden_dim, 7, 7)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck_dim, kernel_size=2, stride=2, padding=1), # (hidden_dim, 7, 7) -> (botleneck_dim, 4, 4)
        
            # decoder
            nn.Upsample(size=4, mode='bilinear'),  # (botleneck_dim, 4, 4) -> (botleneck_dim, 8, 8)
            nn.Conv2d(in_channels=bottleneck_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (botleneck_dim, 8, 8) -> (hidden_dim, 9, 9)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=16, mode='bilinear'),  # (hidden_dim, 9, 9) -> (hidden_dim, 16, 16)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 16, 16) -> (hidden_dim, 17, 17)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=32, mode='bilinear'),  # (hidden_dim, 17, 17) -> (hidden_dim, 32, 32)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 32, 32) -> (hidden_dim, 33, 33)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=63, mode='bilinear'),  # (hidden_dim, 33, 33) -> (hidden_dim, 63, 63)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 63, 63) -> (hidden_dim, 64, 64)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=127, mode='bilinear'),  # (hidden_dim, 64, 64) -> (hidden_dim, 127, 127)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 127, 127) -> (hidden_dim, 128, 128)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=56, mode='bilinear'),  # (hidden_dim, 128, 128) -> (hidden_dim, 56, 56)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),  # (hidden_dim, 56, 56) -> (hidden_dim, 56, 56)
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1)  # (hidden_dim, 56, 56) -> (out_channels, 56, 56)
        )
        return model
    elif spatial_res==8:
        model = nn.Sequential(
            # encoder
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (384, 56, 56) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),  # (hidden_dim, 14, 14) -> (hidden_dim, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 16, 16) -> (hidden_dim, 8, 8)
            nn.ReLU(inplace=True),
            
            # decoder
            nn.Upsample(size=8, mode='bilinear'),  # (hidden_dim, 8, 8) -> (hidden_dim, 8, 8)
            nn.Conv2d(in_channels=bottleneck_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 8, 8) -> (hidden_dim, 9, 9)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=32, mode='bilinear'),  # (hidden_dim, 9, 9) -> (hidden_dim, 18, 18)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 18, 18) -> (hidden_dim, 19, 19)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=63, mode='bilinear'),  # (hidden_dim, 19, 19) -> (hidden_dim,63, 63)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 63, 63) -> (hidden_dim, 64, 64)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=127, mode='bilinear'),  # (hidden_dim, 64, 64) -> (hidden_dim, 128, 128)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 128, 128) -> (hidden_dim, 129, 129)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=56, mode='bilinear'),  # (hidden_dim, 129, 129) -> (hidden_dim, 56, 56)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),  # (hidden_dim, 56, 56) -> (hidden_dim, 56, 56)
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1)  # (hidden_dim, 56, 56) -> (out_channels, 56, 56)
        )
        return model
    elif spatial_res == 16:
        model = nn.Sequential(
            # encoder
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (384, 56, 56) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),  # (hidden_dim, 28, 28) -> (hidden_dim, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),  # (hidden_dim, 14, 14) -> (hidden_dim, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck_dim, kernel_size=3, stride=1, padding=1),  # (hidden_dim, 16, 16) -> (hidden_dim, 16, 16)
            nn.ReLU(inplace=True),
            
            # decoder
            nn.Upsample(size=32, mode='bilinear'),  # (hidden_dim, 16, 16) -> (hidden_dim, 32, 32)
            nn.Conv2d(in_channels=bottleneck_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 32, 32) -> (hidden_dim, 33, 33)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=63, mode='bilinear'),  # (hidden_dim, 33, 33) -> (hidden_dim, 63, 63)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 63, 63) -> (hidden_dim, 64, 64)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=127, mode='bilinear'),  # (hidden_dim, 64, 64) -> (hidden_dim, 127, 127)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1, padding=1),  # (hidden_dim, 127, 127) -> (hidden_dim, 128, 128)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Upsample(size=56, mode='bilinear'),  # (hidden_dim, 128, 128) -> (hidden_dim, 56, 56)
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),  # (hidden_dim, 56, 56) -> (hidden_dim, 56, 56)
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1)  # (hidden_dim, 56, 56) -> (out_channels, 56, 56)
        )
        return model
            
            

