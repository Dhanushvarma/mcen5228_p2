import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from config import config
from data import ImageDepthDataset
from model import U_Net
from utils import init_weights, count_parameters

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


def evaluate(model, dataloader):
    """
    Evaluate model on a dataloader.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for evaluation
    
    Returns:
        Tuple of (avg_l1_error, avg_l1_error_under_3m)
    """
    model.eval()
    L1 = nn.L1Loss()
    
    total_l1 = 0
    total_l1_under3 = 0
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Forward pass
            coded = batch["Coded"].to(device)
            depth_gt = batch["Depth"].to(device)
            
            reconstruction = config.post_forward(model(coded))
            
            # Calculate L1 error
            total_l1 += L1(reconstruction[:, 0], depth_gt).item() * len(batch["Coded"])
            
            # Calculate L1 error for depth < 3m
            mask = depth_gt < 3
            if torch.any(mask):
                total_l1_under3 += L1(reconstruction[mask, 0], depth_gt[mask]).item() * len(batch["Coded"])
            
            sample_count += len(batch["Coded"])
    
    model.train()
    return total_l1 / sample_count, total_l1_under3 / sample_count


def train(args):
    """
    Main training function.
    """
    print("="*60)
    print("Training Configuration")
    print("="*60)
    print(config)
    print("="*60)
    print(f"Device: {device}")
    print("="*60)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, "MetricWeightedLossBlenderNYU")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load training datasets
    print("\nLoading training datasets...")
    train_dataset_dict = create_datasets(args.dataset_dir, config.train_datasets, cache=True)
    
    # Combine training datasets
    train_dataset = ConcatDataset(list(train_dataset_dict.values()))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    print(f"Training samples: {len(train_dataset)}")
    for name, dataset in train_dataset_dict.items():
        print(f"  - {name}: {len(dataset)} samples")
    
    # Load test datasets
    print("\nLoading test datasets...")
    test_dataset_dict = create_datasets(args.dataset_dir, config.test_datasets, cache=True)
    test_loaders = {
        name: DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        for name, dataset in test_dataset_dict.items()
    }
    
    for name, dataset in test_dataset_dict.items():
        print(f"  - {name}: {len(dataset)} samples")
    
    # Initialize model
    print("\nInitializing model...")
    model = U_Net(img_ch=config.img_channels, output_ch=config.output_channels).to(device)
    init_weights(model)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Initialize Weights & Biases if requested
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name="MetricWeightedLossBlenderNYU",
            config={
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "model": "U_Net",
            }
        )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    best_test_loss = float('inf')
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        model.train()
        
        # Training
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            coded = batch["Coded"].to(device)
            depth_gt = batch["Depth"].to(device)
            
            # Forward pass
            reconstruction = config.post_forward(model(coded))
            
            # Compute loss
            loss = config.compute_loss(depth_gt, reconstruction)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Evaluation on test sets
        test_results = {}
        total_test_l1 = 0
        total_test_l1_under3 = 0
        
        for name, test_loader in test_loaders.items():
            l1, l1_under3 = evaluate(model, test_loader)
            test_results[name] = {"L1": l1, "L1_under3": l1_under3}
            total_test_l1 += l1
            total_test_l1_under3 += l1_under3
        
        avg_test_l1 = total_test_l1 / len(test_loaders)
        avg_test_l1_under3 = total_test_l1_under3 / len(test_loaders)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1}/{config.epochs} | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Test L1: {avg_test_l1:.4f} | Test L1 <3m: {avg_test_l1_under3:.4f}")
        for name, results in test_results.items():
            print(f"    {name}: L1={results['L1']:.4f}, L1<3m={results['L1_under3']:.4f}")
        
        # Log to wandb
        if args.use_wandb:
            log_dict = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "test_L1": avg_test_l1,
                "test_L1_under3": avg_test_l1_under3,
            }
            for name, results in test_results.items():
                log_dict[f"{name}_L1"] = results["L1"]
                log_dict[f"{name}_L1_under3"] = results["L1_under3"]
            wandb.log(log_dict)
        
        # Save best model
        if avg_test_l1_under3 < best_test_loss:
            best_test_loss = avg_test_l1_under3
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pt"))
            print(f"  ✓ Saved best model (L1<3m: {best_test_loss:.4f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % config.save_every == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt"))
            print(f"  ✓ Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final.pt"))
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best test L1 <3m: {best_test_loss:.4f}")
    print(f"Models saved to: {checkpoint_dir}")
    print("="*60)
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train depth estimation model")
    parser.add_argument("--dataset_dir", type=str, default="datasets", 
                        help="Path to dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Path to save checkpoints")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="codedvo",
                        help="W&B project name")
    
    args = parser.parse_args()
    train(args)