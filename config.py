import torch
import torch.nn as nn

# Loss Functions
def weighted_mse_loss(input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
    """
    Weighted Mean Squared Error Loss.
    
    Args:
        input: Predicted depth values
        target: Ground truth depth values
        weight: Weight tensor for each pixel
    
    Returns:
        Weighted MSE loss
    """
    return torch.sum(weight * (input - target) ** 2) / torch.sum(weight)


class Config:
    """
    Simplified configuration for MetricWeightedLossBlenderNYU experiment.
    """
    # Model parameters
    model_name = "U_Net"
    img_channels = 3
    output_channels = 1
    
    # Training parameters
    epochs = 80
    batch_size = 3
    learning_rate = 1e-4
    
    # Dataset parameters
    coded_dir = "Codedphasecam-27Linear"
    image_size = (480, 640)
    
    # Training datasets configuration
    # Format: (dataset_name, is_blender, scale_factor)
    train_datasets = [
        ("LivingRoom1", True, 1),      # Blender dataset, scale_factor=1
        ("nyu_data", False, 1000),     # NYU dataset, scale_factor=1000
    ]
    
    # Test datasets configuration
    # Format: (dataset_name, is_blender, scale_factor)
    test_datasets = [
        ("DiningRoom", True, 1),       # Blender test dataset
        ("Corridor", True, 1),         # Blender test dataset
    ]
    
    # Checkpoint saving
    save_every = 10
    checkpoint_dir = "checkpoints"
    
    # Depth range (in meters)
    depth_min = 0.0
    depth_max = 6.0
    
    @staticmethod
    def compute_loss(ground_truth: torch.Tensor, reconstruction: torch.Tensor):
        """
        Compute weighted MSE loss with depth-dependent weighting.
        
        Args:
            ground_truth: Ground truth depth map (B, H, W) or (B, 1, H, W)
            reconstruction: Predicted depth map (B, 1, H, W)
        
        Returns:
            Loss value
        """
        # Ensure correct shape
        if reconstruction.dim() == 4:
            reconstruction = reconstruction[:, 0]  # (B, H, W)
        
        if ground_truth.dim() == 4:
            ground_truth = ground_truth[:, 0]  # (B, H, W)
        
        # Compute depth-dependent weights: 2^(-0.3 * depth)
        # This gives more weight to closer objects
        weight = 2 ** (-0.3 * ground_truth)
        
        return weighted_mse_loss(reconstruction, ground_truth, weight)
    
    @staticmethod
    def post_forward(reconstruction: torch.Tensor) -> torch.Tensor:
        """
        Post-processing after model forward pass.
        For metric depth, no transformation needed.
        
        Args:
            reconstruction: Model output
        
        Returns:
            Post-processed output
        """
        return reconstruction
    
    @staticmethod
    def get_dataset_config(dataset_name: str, dataset_list: list):
        """
        Get configuration for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            dataset_list: List of dataset configurations to search
        
        Returns:
            Tuple of (is_blender, scale_factor) or None if not found
        """
        for name, is_blender, scale_factor in dataset_list:
            if name == dataset_name:
                return is_blender, scale_factor
        return None
    
    @classmethod
    def get_train_dataset_names(cls):
        """Get list of training dataset names."""
        return [name for name, _, _ in cls.train_datasets]
    
    @classmethod
    def get_test_dataset_names(cls):
        """Get list of test dataset names."""
        return [name for name, _, _ in cls.test_datasets]
    
    def __repr__(self):
        train_info = "\n    ".join([f"{name} (blender={bl}, scale={sf})" 
                                    for name, bl, sf in self.train_datasets])
        test_info = "\n    ".join([f"{name} (blender={bl}, scale={sf})" 
                                   for name, bl, sf in self.test_datasets])
        return (
            f"Config(\n"
            f"  model={self.model_name}\n"
            f"  epochs={self.epochs}\n"
            f"  batch_size={self.batch_size}\n"
            f"  learning_rate={self.learning_rate}\n"
            f"  train_datasets=[\n    {train_info}\n  ]\n"
            f"  test_datasets=[\n    {test_info}\n  ]\n"
            f"  coded_dir={self.coded_dir}\n"
            f")"
        )


# Create a global config instance
config = Config()