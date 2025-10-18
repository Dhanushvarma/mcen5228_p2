import argparse
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import config
from data import ImageDepthDataset
from model import U_Net

# Set environment for OpenEXR support
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_datasets(dataset_dir, dataset_configs, cache=True):
    """
    Create datasets from configuration.
    
    Args:
        dataset_dir: Base directory for datasets
        dataset_configs: List of (name, is_blender, scale_factor) tuples
        cache: Whether to cache dataset in memory
    
    Returns:
        Dictionary of dataset name -> dataset object
    """
    datasets = {}
    for dataset_name, is_blender, scale_factor in dataset_configs:
        datasets[dataset_name] = ImageDepthDataset(
            base=dataset_dir,
            path=dataset_name,
            codedDir=config.coded_dir,
            cache=cache,
            is_blender=is_blender,
            image_size=config.image_size,
            scale_factor=scale_factor
        )
    return datasets


def sigma_metric(estimated_depth, ground_truth_depth, threshold):
    """
    Compute sigma accuracy metric (standard depth estimation metric).
    
    Args:
        estimated_depth: Predicted depth values
        ground_truth_depth: Ground truth depth values
        threshold: Threshold for accuracy (typically 1.25, 1.25^2, or 1.25^3)
    
    Returns:
        Percentage of pixels within threshold
    """
    ratio = torch.max(estimated_depth / ground_truth_depth, ground_truth_depth / estimated_depth)
    return torch.mean((ratio < threshold).float())


def evaluate_dataset(model, dataloader, output_dir, dataset_name):
    """
    Evaluate model on a single dataset.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for evaluation
        output_dir: Directory to save predictions
        dataset_name: Name of the dataset being evaluated
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    L1 = nn.L1Loss()
    
    # Create output directory for predictions
    pred_dir = os.path.join(output_dir, dataset_name, "pred_depth")
    os.makedirs(pred_dir, exist_ok=True)
    
    metrics = {
        "l1_error": 0,
        "l1_error_under3": 0,
        "abs_rel": 0,
        "sq_rel": 0,
        "rmse": 0,
        "rmse_log": 0,
        "sigma_1.25": 0,
        "sigma_1.25^2": 0,
        "sigma_1.25^3": 0,
        "sample_count": 0,
        "total_inference_time": 0,
    }
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            # Forward pass with timing
            coded = batch["Coded"].to(device)
            depth_gt = batch["Depth"].to(device)
            
            start_time = time.time()
            reconstruction = config.post_forward(model(coded))
            inference_time = time.time() - start_time
            
            metrics["total_inference_time"] += inference_time
            
            # Get predictions
            pred = reconstruction[:, 0]  # (B, H, W)
            gt = depth_gt  # (B, H, W)
            
            # Apply valid mask (depth > 0)
            valid_mask = gt > 0
            
            # Clamp values to valid range [0, 6] meters
            pred_valid = torch.clamp(pred[valid_mask], 0, config.depth_max).to(device)
            gt_valid = torch.clamp(gt[valid_mask], 0, config.depth_max)
            
            # Compute metrics on valid pixels
            batch_size = len(batch["Coded"])
            
            # L1 errors
            metrics["l1_error"] += L1(pred_valid, gt_valid).item() * batch_size
            
            # L1 error for depth < 3m
            mask_under3 = gt_valid < 3
            if torch.any(mask_under3):
                metrics["l1_error_under3"] += L1(pred_valid[mask_under3], gt_valid[mask_under3]).item() * batch_size
            
            # Relative errors
            metrics["abs_rel"] += torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid).item() * batch_size
            metrics["sq_rel"] += torch.mean(((pred_valid - gt_valid) ** 2) / gt_valid).item() * batch_size
            
            # RMSE
            metrics["rmse"] += torch.sqrt(torch.mean((pred_valid - gt_valid) ** 2)).item() * batch_size
            
            # RMSE log
            log_diff = torch.log(pred_valid) - torch.log(gt_valid)
            metrics["rmse_log"] += torch.sqrt(torch.mean(log_diff ** 2)).item() * batch_size
            
            # Sigma metrics
            metrics["sigma_1.25"] += sigma_metric(pred_valid, gt_valid, 1.25).item() * batch_size
            metrics["sigma_1.25^2"] += sigma_metric(pred_valid, gt_valid, 1.25 ** 2).item() * batch_size
            metrics["sigma_1.25^3"] += sigma_metric(pred_valid, gt_valid, 1.25 ** 3).item() * batch_size
            
            metrics["sample_count"] += batch_size
            
            # Save predicted depth map
            # Scale back to original scale for saving (use 5000 as standard for PNG)
            pred_np = pred[0].cpu().numpy()  # First image in batch
            pred_scaled = (pred_np * 5000).astype(np.uint16)
            output_path = os.path.join(pred_dir, f"{idx:05d}.png")
            cv2.imwrite(output_path, pred_scaled)
    
    # Compute averages
    avg_metrics = {k: v / metrics["sample_count"] for k, v in metrics.items() 
                   if k not in ["sample_count", "total_inference_time"]}
    
    # Add timing metrics
    avg_metrics["avg_inference_time"] = metrics["total_inference_time"] / len(dataloader)
    avg_metrics["fps"] = 1.0 / avg_metrics["avg_inference_time"]
    
    return avg_metrics


def main(args):
    print("="*60)
    print("Evaluation Configuration")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model = U_Net(img_ch=config.img_channels, output_ch=config.output_channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Load test datasets
    print("\nLoading test datasets...")
    test_dataset_dict = create_datasets(args.dataset_dir, config.test_datasets, cache=True)
    
    for name, dataset in test_dataset_dict.items():
        print(f"  - {name}: {len(dataset)} samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate on each test dataset
    print("\n" + "="*60)
    print("Starting Evaluation")
    print("="*60)
    
    all_results = {}
    
    for dataset_name, dataset in test_dataset_dict.items():
        print(f"\nEvaluating on: {dataset_name}")
        print("-"*60)
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        metrics = evaluate_dataset(model, dataloader, args.output_dir, dataset_name)
        all_results[dataset_name] = metrics
        
        # Print results
        print(f"Results for {dataset_name}:")
        print(f"  L1 Error:              {metrics['l1_error']:.4f} m")
        print(f"  L1 Error (<3m):        {metrics['l1_error_under3']:.4f} m")
        print(f"  Absolute Rel Error:    {metrics['abs_rel']:.4f}")
        print(f"  Squared Rel Error:     {metrics['sq_rel']:.4f}")
        print(f"  RMSE:                  {metrics['rmse']:.4f} m")
        print(f"  RMSE (log):            {metrics['rmse_log']:.4f}")
        print(f"  Sigma 1.25:            {metrics['sigma_1.25']:.4f}")
        print(f"  Sigma 1.25^2:          {metrics['sigma_1.25^2']:.4f}")
        print(f"  Sigma 1.25^3:          {metrics['sigma_1.25^3']:.4f}")
        print(f"  Avg Inference Time:    {metrics['avg_inference_time']:.4f} s")
        print(f"  FPS:                   {metrics['fps']:.2f}")
        print(f"  Predictions saved to: {os.path.join(args.output_dir, dataset_name, 'pred_depth')}")
    
    # Summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    
    avg_l1 = np.mean([m['l1_error'] for m in all_results.values()])
    avg_l1_under3 = np.mean([m['l1_error_under3'] for m in all_results.values()])
    avg_sigma_125 = np.mean([m['sigma_1.25'] for m in all_results.values()])
    
    print(f"Average L1 Error:       {avg_l1:.4f} m")
    print(f"Average L1 Error (<3m): {avg_l1_under3:.4f} m")
    print(f"Average Sigma 1.25:     {avg_sigma_125:.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate depth estimation model")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset_dir", "-d", type=str, default="datasets",
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", "-o", type=str, default="evaluation_results",
                        help="Path to save evaluation results")
    
    args = parser.parse_args()
    main(args)