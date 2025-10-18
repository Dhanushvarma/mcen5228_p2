# MCEN5228 Project-2 (Depth Estimation with Coded Aperture)
Project 2 for MCEN 5228 (Advanced Computer Vision)

## Setup

### Requirements
```bash
pip install torch torchvision opencv-python numpy natsort matplotlib
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

## Visualization

Generate qualitative visualizations with viridis/plasma colormaps:
```bash
python visualize.py --pred_dir ./results --dataset_dir ./datasets --output_dir ./visualizations --colormap viridis
```

Options for colormaps: `viridis`, `plasma`, `magma`, `inferno`

This creates:
- Side-by-side comparisons (Ground Truth | Prediction | Error Map)
- Depth visualizations with 0-6m metric range
- Metrics overlaid on each comparison (Abs Rel, RMSE, MAE, δ<1.25)
- Individual comparison images for each sample
- Summary metrics file for the dataset

**Quick visualization of 10 samples:**
```bash
python visualize.py --pred_dir ./results --dataset_dir ./datasets --output_dir ./vis_quick --colormap plasma --num_samples 10
```

**Comparing Multiple Models:**
To compare different models, run eval.py with different checkpoints to different output directories, then visualize each:
```bash
# Model 1
python eval.py --checkpoint ./checkpoints/model1/best.pt --output_dir ./results_model1
python visualize.py --pred_dir ./results_model1 --output_dir ./vis_model1

# Model 2
python eval.py --checkpoint ./checkpoints/model2/best.pt --output_dir ./results_model2
python visualize.py --pred_dir ./results_model2 --output_dir ./vis_model2
```

## Complete Workflow

```bash
# 1. Train the model
python train.py --dataset_dir ./datasets --checkpoint_dir ./checkpoints

# 2. Evaluate on test sets
python eval.py --checkpoint ./checkpoints/MetricWeightedLossBlenderNYU/best.pt \
               --dataset_dir ./datasets \
               --output_dir ./results

# 3. Generate visualizations
python visualize.py --pred_dir ./results \
                    --dataset_dir ./datasets \
                    --output_dir ./visualizations \
                    --colormap viridis
```

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
├── config.py         # Configuration and loss functions
├── data.py           # Dataset loader
├── model.py          # U-Net architecture
├── train.py          # Training script
├── eval.py           # Evaluation script
├── visualize.py      # Visualization and comparison tool
├── datasets/         # Your data (create this)
├── checkpoints/      # Saved models (auto-created)
├── results/          # Evaluation outputs (auto-created)
└── visualizations/   # Comparison figures (auto-created)
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