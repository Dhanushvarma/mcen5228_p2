# mcen5228_p2 (Depth Estimation with Coded Aperture)
Project 2 for MCEN 5228 (Advanced Computer Vision)

## Setup

### Requirements
```bash
pip install torch torchvision opencv-python numpy natsort
pip install wandb  # Optional, for experiment tracking
```

### Dataset Structure

Organize your data as follows:
```
datasets/
├── LivingRoom1/           # Training (Blender)
│   ├── rgb/
│   ├── depth/             # .exr files
│   └── Codedphasecam-27Linear/
├── nyu_data/              # Training (NYU)
│   ├── rgb/
│   ├── depth/             # .png files
│   └── Codedphasecam-27Linear/
├── DiningRoom/            # Test (Blender)
│   └── ...
└── Corridor/              # Test (Blender)
    └── ...
```

## Training

Basic training:
```bash
python train.py --dataset_dir ./datasets --checkpoint_dir ./checkpoints
```

With Weights & Biases logging:
```bash
python train.py --dataset_dir ./datasets --checkpoint_dir ./checkpoints --use_wandb --wandb_project your_project_name
```

Training will save:
- `best.pt` - Best model based on L1 error < 3m
- `epoch_N.pt` - Checkpoint every 10 epochs
- `final.pt` - Final model after all epochs

## Evaluation

Evaluate a trained model:
```bash
python eval.py --checkpoint ./checkpoints/MetricWeightedLossBlenderNYU/best.pt --dataset_dir ./datasets --output_dir ./results
```

This will:
- Evaluate on DiningRoom and Corridor test sets
- Save predicted depth maps to `results/{dataset_name}/pred_depth/`
- Print metrics: L1 error, RMSE, sigma accuracy, and FPS

## Configuration

Edit `config.py` to modify:
- Training/test datasets
- Hyperparameters (epochs, batch size, learning rate)
- Dataset scale factors
- Loss function parameters

**Training datasets:**
```python
train_datasets = [
    ("LivingRoom1", True, 1),   # (name, is_blender, scale_factor)
    ("nyu_data", False, 1000),
]
```

**Test datasets:**
```python
test_datasets = [
    ("DiningRoom", True, 1),
    ("Corridor", True, 1),
]
```

## Model

- Architecture: U-Net
- Input: 3-channel coded aperture RGB images (480×640)
- Output: 1-channel metric depth map
- Loss: Depth-weighted MSE (prioritizes closer objects)

## File Structure

```
├── config.py      # Configuration and loss functions
├── data.py        # Dataset loader
├── model.py       # U-Net architecture
├── train.py       # Training script
├── eval.py        # Evaluation script
├── datasets/      # Your data (create this)
└── checkpoints/   # Saved models (auto-created)
```

## Citation

Based on CodedVO: Coded Visual Odometry (IEEE RA-L 2024)

```bibtex
@ARTICLE{codedvo2024,
  author={Shah, Sachin and Rajyaguru, Naitri and Singh, Chahat Deep and Metzler, Christopher and Aloimonos, Yiannis},
  journal={IEEE Robotics and Automation Letters}, 
  title={CodedVO: Coded Visual Odometry}, 
  year={2024},
  doi={10.1109/LRA.2024.3416788}
}
```