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
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from models import get_autoencoder, get_backbone
from datasets import build_dataset, build_transforms, AD_CLASSES, LOCO_CLASSES
from util import AverageMeter, gaussian_kernel, load_config

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def parser_args():
    parser = argparse.ArgumentParser(description='Convolutional Autoencoder [Evaluation]')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--class_name', type=str, required=True, help='Class name of the dataset')
    parser.add_argument('--ae_model', type=str, default='conv_ae_sp1', help='Autoencoder model')
    parser.add_argument('--ae_hidden_dim', type=int, default=384, help='Autoencoder hidden dimension')
    parser.add_argument('--bottleneck_dim', type=int, default=64, help='Bottleneck dimension of the autoencoder')
    parser.add_argument('--backbone_model', type=str, default='pdn_small', help='Backbone model')
    parser.add_argument('--ae_resume_path', type=str, required=True, help='Path to resume autoencoder weights')
    parser.add_argument('--backbone_resume_path', type=str, default=None, help='Path to resume backbone weights')
    parser.add_argument('--feature_dim', type=int, default=384, help='Feature dimension of the backbone')
    parser.add_argument('--feature_res', type=int, default=1, help='Feature resolution of the backbone')
    
    # for vit ae
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size')
    parser.add_argument('--num_enc_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--num_dec_layers', type=int, default=3, help='Number of decoder layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='MLP ratio')
    parser.add_argument('--in_res', type=int, default=56, help='Input resolution')
    
    # for mat ae
    parser.add_argument('--split_strategy', type=str, default='log', help='Split strategy')
    
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--gaussian_filter', action='store_true', default=False, help='Apply Gaussian filter')
    parser.add_argument('--gaussian_sigma', type=float, default=4, help='Gaussian sigma')
    parser.add_argument('--gaussian_ksize', type=int, default=7, help='Gaussian kernel size')
    parser.add_argument('--split', type=str, default='test', help='Data split')
    parser.add_argument('--multi_category', action='store_true', default=False, help='Multi-category dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--transform', type=str, default='default', help='Transform type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ensemble', action='store_true', default=False, help='Ensemble evaluation')
    # for ensemble evaluation
    parser.add_argument('--ensemble_save_dirs', type=str, nargs='+', default=None, help='Path to ensemble save directories')
    
    args = parser.parse_args()
    return args

def compute_err_map(org_features, recon):
    """Coompute Error map between original features and reconstructed features
    Args:
        org_features (tensor): Original features. Shape: (B, C, H', W')
        recon (tensor): Reconstructed features. Shape: (B, C, H', W')
    Returns:
        err_map (tensor): Error map. Shape: (B, H', W')
    """
    err_l2 = F.mse_loss(org_features, recon, reduction='none')  # (B, C, H', W')
    err_l2 = torch.mean(err_l2, dim=1)  # (B, H', W')
    
    err_cos = 1 - F.cosine_similarity(org_features, recon, dim=1)  # (B, H', W')
    assert torch.all(err_cos > 0), f"Cosine similarity should be positive: {err_cos}"
    assert torch.all(err_l2 > 0), f"MSE loss should be positive: {err_l2}"
    err_map = err_l2 * err_cos  # (B, H', W')
    return err_map

def gaussian_filter(err_map, sigma=1.4, ksize=7):
    """Apply Gaussian filter to the error map

    Args:
        err_map (tensor): Error map. Shape: (B, H, W)
        sigma (float, optional): Standard deviation of the Gaussian filter. Defaults to 1.4.
        ksize (int, optional): Kernel size of the Gaussian filter. Defaults to 7.
    Returns:
        err_map (tensor): Error map after applying Gaussian filter, Shape: (B, H, W)
    """
    err_map = err_map.detach().cpu()
    kernel = gaussian_kernel(ksize, sigma) 
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(err_map.device)  # (1, 1, ksize, ksize)
    padding = ksize // 2
    err_map = F.pad(err_map, (padding, padding, padding, padding), mode='reflect')
    err_map = F.conv2d(err_map.unsqueeze(1), kernel, padding=0).squeeze(1)
    return err_map

def evaluate(args):
    
    if args.ensemble:
        evaluate_ensemble(args)
        return
    
    assert args.split in ['train', 'test'], f"Invalid split: {args.split}"
    assert args.ae_resume_path is not None, "Autoencoder weights should be provided"
    assert args.backbone_resume_path is not None, "Backbone weights should be provided"
    args.num_normal_samples = -1
    args.log_interval = 1
    logger.info(args)
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Build dataset
    transform = build_transforms(args.img_size, args.transform)
    dataset = build_dataset(args.img_size, args.split, args.num_normal_samples, args.data_root, args.class_name, transform, False, multi_category=args.multi_category)
    dataset_name = args.data_root.split('/')[-1]
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Number of samples: {len(dataset)}")
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)
    
    # Build autoencoder
    backbone = get_backbone(args.backbone_model)
    autoencoder = get_autoencoder(args.ae_model, in_channels=args.feature_dim, out_channels=args.feature_dim, spatial_res=args.feature_res, hidden_dim=args.ae_hidden_dim, \
        bottleneck_dim=args.bottleneck_dim, patch_size=args.patch_size, num_enc_layers=args.num_enc_layers, num_dec_layers=args.num_dec_layers, num_heads=args.num_heads, \
        mlp_ratio=args.mlp_ratio, in_res=args.in_res, split_strategy=args.split_strategy
    )
    # Load weights
    autoencoder.load_state_dict(torch.load(args.ae_resume_path, map_location="cpu", weights_only=True))
    backbone.load_state_dict(torch.load(args.backbone_resume_path, map_location="cpu", weights_only=True))
    autoencoder.to(args.device)
    backbone.to(args.device)
    
    autoencoder.eval()
    backbone.eval()
    
    # Evaluation
    loss_meter = AverageMeter()
    logger.info("Start evaluation...")
    results = {
        "err_maps": [],
        "filenames": [],
        "cls_names": [],
        "labels": [],
        "anom_types": []
    }
    for i, batch in enumerate(test_dataloader):
        images = batch["samples"].to(args.device)  # (B, C, H, W)
        results["labels"].append(batch["labels"][0].item())
        results["anom_types"].append(batch["anom_type"][0])
        results["filenames"].append(batch["filenames"][0])
        results["cls_names"].append(batch["clsnames"][0])
        
        with torch.no_grad():
            x = backbone(images)  # (B, c, h, w)
            x_rec = autoencoder(x)
            # import pdb; pdb.set_trace()
            if "mat_ae" in args.ae_model:
                x_rec = x_rec[-1]  # (B, C, H, W)
            loss = F.mse_loss(x, x_rec)
            loss_meter.update(loss.item(), images.size(0))
            err_map = compute_err_map(x, x_rec)  # (B, H', W')

            if args.gaussian_filter:
                err_map = gaussian_filter(err_map, sigma=args.gaussian_sigma, ksize=args.gaussian_ksize)
            err_map = torch.mean(err_map, dim=0)  # (H, W)
            results["err_maps"].append(err_map)
    
        if i % args.log_interval == 0:
            logger.info(f'Iter: [{i}/{len(test_dataloader)}]\t'
                        f'Loss: {loss_meter.avg:.4f}\t')
    
    logger.info(f"Loss: {loss_meter.avg:.4f}")
            
    # Calculate metrics
    global_err_scores = [torch.max(err_map) for err_map in results["err_maps"]]
    global_err_scores = torch.stack(global_err_scores).cpu().numpy()
    
    auc = roc_auc_score(results["labels"], global_err_scores)
    logger.info(f'auROC: {auc:.4f} on {args.class_name}')
    
    # Calculate the auROC score for each anomaly type
    unique_anom_types = list(sorted(set(results["anom_types"])))
    normal_indices = [i for i, x in enumerate(results["anom_types"]) if x == "good"]
    for anom_type in unique_anom_types:
        if anom_type == "good":
            continue
        anom_indices = [i for i, x in enumerate(results["anom_types"]) if x == anom_type]
        normal_scores = global_err_scores[normal_indices]
        anom_scores = global_err_scores[anom_indices]
        scores = np.concatenate([normal_scores, anom_scores])
        labels = [0] * len(normal_scores) + [1] * len(anom_scores)
        auc = roc_auc_score(labels, scores)
        logger.info(f'auROC: {auc:.4f} on {anom_type}')
    logger.info(f'Evaluation Finished🎉')
    
def evaluate_ensemble(args):
    assert args.split in ['train', 'test'], f"Invalid split: {args.split}"
    assert args.ae_resume_path is not None, "Autoencoder weights should be provided"
    assert args.backbone_resume_path is not None, "Backbone weights should be provided"
    args.num_normal_samples = -1
    args.log_interval = 1
    logger.info(args)
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Build dataset
    transform = build_transforms(args.img_size, args.transform)
    dataset = build_dataset(args.img_size, args.split, args.num_normal_samples, args.data_root, args.class_name, transform, False, multi_category=args.multi_category)
    dataset_name = args.data_root.split('/')[-1]
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Number of samples: {len(dataset)}")
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)
    
    # Build autoencoder
    backbone = get_backbone(args.backbone_model)
    backbone.load_state_dict(torch.load(args.backbone_resume_path, map_location="cpu", weights_only=True))
    backbone.to(args.device)
    
    autoencoders = []
    for save_dir in args.ensemble_save_dirs:
        config = load_config(os.path.join(save_dir, 'config.yml'))
        model_config = config['model']
        
        autoencoder = get_autoencoder(model_config['ae_model'], in_channels=model_config['feature_dim'], out_channels=model_config['feature_dim'], spatial_res=model_config['feature_res'], hidden_dim=model_config['ae_hidden_dim'], \
            bottleneck_dim=model_config['bottleneck_dim'], patch_size=model_config['patch_size'], num_enc_layers=model_config['num_enc_layers'], num_dec_layers=model_config['num_dec_layers'], num_heads=model_config['num_heads'], \
            mlp_ratio=model_config['mlp_ratio'], in_res=model_config['in_res']
        )
        autoencoder.load_state_dict(torch.load(os.path.join(save_dir, 'weights.pth'), map_location="cpu", weights_only=True))
        autoencoder.to(args.device)
        autoencoders.append(autoencoder)
    
    # Evaluation
    loss_meter = AverageMeter()
    logger.info("Start Ensemble evaluation...")
    results = {
        "err_maps": [],
        "filenames": [],
        "cls_names": [],
        "labels": [],
        "anom_types": []
    }

    for i, batch in enumerate(test_dataloader):
        images = batch["samples"].to(args.device)  # (B, C, H, W)
        results["labels"].append(batch["labels"][0].item())
        results["anom_types"].append(batch["anom_type"][0])
        results["filenames"].append(batch["filenames"][0])
        results["cls_names"].append(batch["clsnames"][0])
        
        with torch.no_grad():
            x = backbone(images)  # (B, c, h, w)
    
            err_maps = []
            loss = 0
            for autoencoder in autoencoders:
                x_rec = autoencoder(x)
                loss += F.mse_loss(x, x_rec)
                err_map = compute_err_map(x, x_rec)        
                if args.gaussian_filter:
                    err_map = gaussian_filter(err_map, sigma=args.gaussian_sigma, ksize=args.gaussian_ksize)
                err_maps.append(err_map)    
            err_maps = torch.stack(err_maps)  # (N, B, H', W')
            err_maps = torch.max(err_maps, dim=0)  # (B, H', W')
            err_map = torch.mean(err_map, dim=0)  # (H, W)
            results["err_maps"].append(err_map)

        loss_meter.update(loss.item(), images.size(0))
        if i % args.log_interval == 0:
            logger.info(f'Iter: [{i}/{len(test_dataloader)}]\t'
                        f'Loss: {loss_meter.avg:.4f}\t')
    
    logger.info(f"Loss: {loss_meter.avg:.4f}")
    
    # Calculate metrics
    global_err_scores = [torch.max(err_map) for err_map in results["err_maps"]]
    global_err_scores = torch.stack(global_err_scores).cpu().numpy()
    
    auc = roc_auc_score(results["labels"], global_err_scores)
    logger.info(f'auROC: {auc:.4f} on {args.class_name}')
    
    # Calculate the auROC score for each anomaly type
    unique_anom_types = list(sorted(set(results["anom_types"])))
    normal_indices = [i for i, x in enumerate(results["anom_types"]) if x == "good"]
    for anom_type in unique_anom_types:
        if anom_type == "good":
            continue
        anom_indices = [i for i, x in enumerate(results["anom_types"]) if x == anom_type]
        normal_scores = global_err_scores[normal_indices]
        anom_scores = global_err_scores[anom_indices]
        scores = np.concatenate([normal_scores, anom_scores])
        labels = [0] * len(normal_scores) + [1] * len(anom_scores)
        auc = roc_auc_score(labels, scores)
        logger.info(f'auROC: {auc:.4f} on {anom_type}')
    logger.info(f'Evaluation Finished🎉')
    
            
if __name__ == '__main__':
    args = parser_args()
    
    if args.multi_category:
        for cat in AD_CLASSES:
            args.class_name = cat
            args.multi_category = False
            evaluate(args)
    else:
        evaluate(args)
    
    
    
    
    