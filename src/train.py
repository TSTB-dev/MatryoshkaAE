import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

import random
import argparse

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models import get_autoencoder, get_backbone
from datasets import build_dataset, build_transforms
from util import AverageMeter

import tensorboardX
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def parser_args():
    parser = argparse.ArgumentParser(description='Convolutional Autoencoder [Training]')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--class_name', type=str, required=True, help='Class name of the dataset')
    parser.add_argument('--num_normal_samples', type=int, default=-1, help='Number of normal samples')
    parser.add_argument('--ae_model', type=str, default='conv_ae_sp1', help='Autoencoder model')
    parser.add_argument('--ae_hidden_dim', type=int, default=384, help='Autoencoder hidden dimension')
    parser.add_argument('--bottleneck_dim', type=int, default=64, help='Bottleneck dimension of the autoencoder')
    parser.add_argument('--backbone_model', type=str, default='pdn_small', help='Backbone model')
    parser.add_argument('--backbone_resume_path', type=str, default=None, help='Path to resume backbone weights')
    parser.add_argument('--feature_dim', type=int, default=384, help='Feature dimension of the backbone')
    parser.add_argument('--feature_res', type=int, default=1, help='Feature resolution of the backbone')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--split', type=str, default='train', help='Data split')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--transform', type=str, default='default', help='Transform type')
    
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient clipping')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    parser.add_argument('--log_interval', type=int, default=1, help='Log interval')
    parser.add_argument('--save_dir', type=str, default='weights', help='Save directory')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to resume weights')
    
    return parser.parse_args()

def train(args):
    assert args.split in ['train', 'test'], f"Invalid split: {args.split}"
    
    proj_dir = os.path.join(args.save_dir, f"{args.class_name}_{args.ae_model}_dim{args.bottleneck_dim}_{args.backbone_model}")
    tb_writer = tensorboardX.SummaryWriter(logdir=os.path.join(proj_dir, 'logs'))
    logger.info(f"Arguments: {args}")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = build_transforms(args)
    dataset = build_dataset(args, transform)
    dataset_name = dataset.__class__.__name__
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Number of samples: {len(dataset)}")
    
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    total_iter = len(train_dataloader) * args.num_epochs
    iter_per_epoch = len(train_dataloader)
    
    # Build autoencoder
    backbone = get_backbone(args.backbone_model)
    autoencoder = get_autoencoder(args.ae_model, in_channels=args.feature_dim, out_channels=args.feature_dim, spatial_res=args.feature_res, hidden_dim=args.ae_hidden_dim, \
        bottleneck_dim=args.bottleneck_dim)
    
    # Build optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(autoencoder.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")
    
    # Build scheduler
    if args.scheduler is not None:
        if args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iter)
        else:
            raise ValueError(f"Invalid scheduler: {args.scheduler}")
    else:
        scheduler = None
    
    # Load weights
    if args.resume_path is not None:
        autoencoder.load_state_dict(torch.load(args.resume_path))
    
    if args.backbone_resume_path is not None:
        backbone.load_state_dict(torch.load(args.backbone_resume_path, map_location="cpu"))
    else:
        raise ValueError("Backbone resume path is required")
    
    autoencoder.to(args.device)
    backbone.to(args.device)
    
    autoencoder.train()
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    
    logger.info("Start training...")
    criterion = nn.MSELoss()
    for epoch in range(args.num_epochs):
        loss_meter = AverageMeter()
        for j, batch in enumerate(train_dataloader):
            images = batch['samples'].to(args.device)
            
            # Foward pass
            
            # 1. Feature extraction
            with torch.no_grad():
                x = backbone(images)  # (b, c, h, w)
            # 2. Autoencoding
            x_rec = autoencoder(x)  
            loss = criterion(x_rec, x)
            loss_meter.update(loss.item(), x.size(0))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            if args.grad_clip is not None:
                nn.utils.clip_grad_norm_(autoencoder.parameters(), args.grad_clip)
            optimizer.step()
            
            # Log
            if j % args.log_interval == 0:
                logger.info(f"Epoch [{epoch+1}/{args.num_epochs}] Iter [{j}/{iter_per_epoch}] Loss: {loss_meter.avg:.4f}")
                tb_writer.add_scalar("Loss", loss_meter.avg, epoch * iter_per_epoch + j)
            
            # we update lr scheuler in a step-wise manner
            if scheduler is not None:
                scheduler.step()
            
            assert not torch.isnan(loss).any(), "Loss is NaN"
            
        logger.info(f"Epoch [{epoch+1}/{args.num_epochs}] Loss: {loss_meter.avg:.4f}")
    
    tb_writer.close()
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(proj_dir, 'weights.pth')
    torch.save(autoencoder.state_dict(), model_path)
    logger.info(f"Model saved at {model_path}")
    
    logger.info("Training finished")

if __name__ == '__main__':
    args = parser_args()
    train(args)
            
    
    
    
    