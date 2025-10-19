import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path

from config import config

# Color schemes
COLORMAPS = {
    'viridis': plt.cm.viridis,
    'plasma': plt.cm.plasma,
    'magma': plt.cm.magma,
    'inferno': plt.cm.inferno,
}

# Set the environment variable for OpenEXR support in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def load_depth_map(filepath, is_blender=False, scale_factor=1):
    """
    Load a depth map from file.
    
    Args:
        filepath: Path to depth file
        is_blender: Whether it's a Blender EXR file
        scale_factor: Scale factor to convert to meters
    
    Returns:
        Depth map in meters as numpy array
    """
    if is_blender:
        # Read EXR file
        depth = cv2.imread(str(filepath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Could not read depth file: {filepath}")
        return depth[:, :, 0]  # Take first channel
    else:
        # Read PNG file
        depth = cv2.imread(str(filepath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Could not read depth file: {filepath}")
        return depth / scale_factor


def load_predicted_depth(filepath, scale_factor=5000):
    """
    Load predicted depth map (saved as PNG with scale 5000).
    
    Args:
        filepath: Path to predicted depth PNG
        scale_factor: Scale factor used when saving (default 5000)
    
    Returns:
        Depth map in meters
    """
    depth = cv2.imread(str(filepath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Could not read predicted depth: {filepath}")
    return depth / scale_factor


def compute_metrics(pred, gt, valid_mask=None):
    """
    Compute depth metrics between prediction and ground truth.
    
    Args:
        pred: Predicted depth map
        gt: Ground truth depth map
        valid_mask: Boolean mask of valid pixels
    
    Returns:
        Dictionary of metrics
    """
    if valid_mask is None:
        valid_mask = (gt > 0) & (gt <= 6)
    
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    
    # Absolute Relative Error
    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
    
    # RMSE
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
    
    # Additional metrics
    sq_rel = np.mean(((pred_valid - gt_valid) ** 2) / gt_valid)
    mae = np.mean(np.abs(pred_valid - gt_valid))
    
    # Sigma accuracy
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    sigma_1 = np.mean(ratio < 1.25)
    sigma_2 = np.mean(ratio < 1.25 ** 2)
    sigma_3 = np.mean(ratio < 1.25 ** 3)
    
    return {
        'abs_rel': abs_rel,
        'rmse': rmse,
        'sq_rel': sq_rel,
        'mae': mae,
        'sigma_1.25': sigma_1,
        'sigma_1.25^2': sigma_2,
        'sigma_1.25^3': sigma_3,
    }


def apply_colormap(depth, vmin=0, vmax=6, colormap='viridis', invalid_color=[0, 0, 0]):
    """
    Apply colormap to depth map.
    
    Args:
        depth: Depth map in meters
        vmin: Minimum depth value for colormap
        vmax: Maximum depth value for colormap
        colormap: Name of colormap to use
        invalid_color: RGB color for invalid pixels
    
    Returns:
        RGB image with colormap applied
    """
    # Normalize depth to [0, 1]
    depth_norm = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    
    # Apply colormap
    cmap = COLORMAPS.get(colormap, plt.cm.viridis)
    colored = cmap(depth_norm)[:, :, :3]  # Remove alpha channel
    
    # Convert to 0-255 range
    colored = (colored * 255).astype(np.uint8)
    
    # Set invalid pixels to black (or specified color)
    invalid_mask = (depth <= 0) | (depth > vmax)
    colored[invalid_mask] = invalid_color
    
    return colored


def create_comparison_figure(gt, pred, metrics, title="Depth Comparison", 
                            colormap='viridis', vmin=0, vmax=6):
    """
    Create a comparison figure with ground truth, prediction, and error map.
    
    Args:
        gt: Ground truth depth map
        pred: Predicted depth map
        metrics: Dictionary of computed metrics
        title: Figure title
        colormap: Colormap to use
        vmin, vmax: Depth range for visualization
    
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Ground Truth
    gt_colored = apply_colormap(gt, vmin, vmax, colormap)
    axes[0].imshow(gt_colored)
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Prediction
    pred_colored = apply_colormap(pred, vmin, vmax, colormap)
    axes[1].imshow(pred_colored)
    axes[1].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Error Map
    valid_mask = (gt > 0) & (gt <= vmax)
    error = np.abs(pred - gt)
    error[~valid_mask] = 0
    
    # Use a different colormap for error (red for high error)
    error_colored = apply_colormap(error, 0, 2, 'hot', invalid_color=[0, 0, 0])
    axes[2].imshow(error_colored)
    axes[2].set_title('Absolute Error', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add colorbar
    fig.subplots_adjust(right=0.85, wspace=0.05)
    
    # Colorbar for depth
    cbar_ax1 = fig.add_axes([0.88, 0.55, 0.02, 0.35])
    norm1 = colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm1, cmap=colormap), 
                       cax=cbar_ax1)
    cb1.set_label('Depth (m)', fontsize=12)
    
    # Colorbar for error
    cbar_ax2 = fig.add_axes([0.88, 0.1, 0.02, 0.35])
    norm2 = colors.Normalize(vmin=0, vmax=2)
    cb2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm2, cmap='hot'), 
                       cax=cbar_ax2)
    cb2.set_label('Error (m)', fontsize=12)
    
    # Add metrics text
    metrics_text = (
        f"Abs Rel: {metrics['abs_rel']:.4f}\n"
        f"RMSE: {metrics['rmse']:.4f} m\n"
        f"MAE: {metrics['mae']:.4f} m\n"
        f"δ < 1.25: {metrics['sigma_1.25']:.3f}"
    )
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11, 
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def visualize_dataset(dataset_name, dataset_dir, pred_dir, output_dir, 
                     is_blender, scale_factor, colormap='viridis', 
                     num_samples=None, save_individual=True):
    """
    Visualize predictions for an entire dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_dir: Base directory containing datasets
        pred_dir: Directory containing predicted depth maps
        output_dir: Directory to save visualizations
        is_blender: Whether dataset is Blender format
        scale_factor: Scale factor for ground truth depth
        colormap: Colormap to use for visualization
        num_samples: Number of samples to visualize (None for all)
        save_individual: Whether to save individual comparison images
    """
    # Setup paths
    gt_depth_dir = Path(dataset_dir) / dataset_name / "depth"
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of prediction files
    pred_files = sorted(Path(pred_dir).glob("*.png"))
    
    if num_samples is not None:
        pred_files = pred_files[:num_samples]
    
    print(f"\n{'='*60}")
    print(f"Visualizing: {dataset_name}")
    print(f"{'='*60}")
    print(f"Number of samples: {len(pred_files)}")
    print(f"Colormap: {colormap}")
    
    # Accumulate metrics
    all_metrics = []
    
    for i, pred_file in enumerate(pred_files):
        # Load prediction
        pred = load_predicted_depth(pred_file, scale_factor=5000)
        
        # Extract the index from prediction filename (e.g., 00000.png -> 0)
        # This handles both 00000.png and 0.png formats
        pred_idx = int(pred_file.stem)
        
        # Construct ground truth filename
        # Try both with and without .exr extension for different naming conventions
        if is_blender:
            # Try exact index match first (e.g., 0.exr, 1.exr, 100.exr)
            gt_file = gt_depth_dir / f"{pred_idx}.exr"
            if not gt_file.exists():
                # Try with leading zeros (e.g., 00000.exr)
                gt_file = gt_depth_dir / f"{pred_idx:05d}.exr"
        else:
            # For PNG files, try exact index first
            gt_file = gt_depth_dir / f"{pred_idx}.png"
            if not gt_file.exists():
                # Try with leading zeros
                gt_file = gt_depth_dir / f"{pred_idx:05d}.png"
        
        # Check if file exists
        if not gt_file.exists():
            print(f"  Warning: Ground truth not found for {pred_file.name}, trying alternate naming...")
            # Last resort: try with the same name as prediction
            gt_file = gt_depth_dir / pred_file.name.replace('.png', '.exr' if is_blender else '.png')
            if not gt_file.exists():
                print(f"  Skipping {pred_file.name} - no matching ground truth found")
                continue
        
        # Load ground truth
        gt = load_depth_map(gt_file, is_blender, scale_factor)
        
        # Clip to valid range
        pred = np.clip(pred, 0, 6)
        gt = np.clip(gt, 0, 6)
        
        # Compute metrics
        metrics = compute_metrics(pred, gt)
        all_metrics.append(metrics)
        
        # Create and save comparison figure
        if save_individual:
            fig = create_comparison_figure(
                gt, pred, metrics, 
                title=f"{dataset_name} - Sample {i:03d}",
                colormap=colormap
            )
            fig.savefig(output_path / f"comparison_{i:03d}.png", 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(pred_files)} samples")
    
    if not all_metrics:
        print(f"  ERROR: No valid samples found for {dataset_name}")
        return None
    
    # Compute average metrics
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) 
                   for k in all_metrics[0].keys()}
    
    # Print results
    print(f"\nAverage Metrics for {dataset_name}:")
    print(f"  Abs Rel:     {avg_metrics['abs_rel']:.4f}")
    print(f"  RMSE:        {avg_metrics['rmse']:.4f} m")
    print(f"  MAE:         {avg_metrics['mae']:.4f} m")
    print(f"  Sq Rel:      {avg_metrics['sq_rel']:.4f}")
    print(f"  δ < 1.25:    {avg_metrics['sigma_1.25']:.3f}")
    print(f"  δ < 1.25²:   {avg_metrics['sigma_1.25^2']:.3f}")
    print(f"  δ < 1.25³:   {avg_metrics['sigma_1.25^3']:.3f}")
    
    # Save metrics to file
    metrics_file = output_path / "metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write(f"Metrics for {dataset_name}\n")
        f.write("="*50 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"{key:20s}: {value:.6f}\n")
    
    print(f"\nVisualizations saved to: {output_path}")
    print(f"Metrics saved to: {metrics_file}")
    
    return avg_metrics


def main(args):
    print("="*60)
    print("Depth Visualization Tool")
    print("="*60)
    print(f"Prediction directory: {args.pred_dir}")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Colormap: {args.colormap}")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize each test dataset
    all_dataset_metrics = {}
    
    for dataset_name, is_blender, scale_factor in config.test_datasets:
        pred_dataset_dir = Path(args.pred_dir) / dataset_name / "pred_depth"
        
        if not pred_dataset_dir.exists():
            print(f"\nWarning: Predictions not found for {dataset_name} at {pred_dataset_dir}")
            print("Skipping...")
            continue
        
        metrics = visualize_dataset(
            dataset_name=dataset_name,
            dataset_dir=args.dataset_dir,
            pred_dir=pred_dataset_dir,
            output_dir=args.output_dir,
            is_blender=is_blender,
            scale_factor=scale_factor,
            colormap=args.colormap,
            num_samples=args.num_samples,
            save_individual=not args.no_individual
        )
        
        all_dataset_metrics[dataset_name] = metrics
    
    # Summary across all datasets
    if len(all_dataset_metrics) > 1:
        print("\n" + "="*60)
        print("Summary Across All Datasets")
        print("="*60)
        
        avg_abs_rel = np.mean([m['abs_rel'] for m in all_dataset_metrics.values()])
        avg_rmse = np.mean([m['rmse'] for m in all_dataset_metrics.values()])
        avg_sigma = np.mean([m['sigma_1.25'] for m in all_dataset_metrics.values()])
        
        print(f"Average Abs Rel:  {avg_abs_rel:.4f}")
        print(f"Average RMSE:     {avg_rmse:.4f} m")
        print(f"Average δ < 1.25: {avg_sigma:.3f}")
        print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize depth prediction results")
    parser.add_argument("--pred_dir", "-p", type=str, required=True,
                        help="Directory containing predicted depth maps (output from eval.py)")
    parser.add_argument("--dataset_dir", "-d", type=str, default="datasets",
                        help="Path to dataset directory (for ground truth)")
    parser.add_argument("--output_dir", "-o", type=str, default="visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--colormap", "-c", type=str, default="viridis",
                        choices=['viridis', 'plasma', 'magma', 'inferno'],
                        help="Colormap for depth visualization")
    parser.add_argument("--num_samples", "-n", type=int, default=None,
                        help="Number of samples to visualize (default: all)")
    parser.add_argument("--no_individual", action="store_true",
                        help="Skip saving individual comparison images")
    
    args = parser.parse_args()
    main(args)