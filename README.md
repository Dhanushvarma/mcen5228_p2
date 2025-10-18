# MCEN5228 Project-2 (Depth Estimation with Coded Aperture)
Project 2 for MCEN 5228 (Advanced Computer Vision)

## Setup

### Requirements
```bash
uv pip install torch torchvision opencv-python numpy natsort matplotlib
uv pip install wandb  # Optional, for experiment tracking
```

### Quick Start with Bash Scripts

The easiest way to run the complete pipeline is using the provided bash scripts:

```bash
# 1. Setup (make scripts executable)
chmod +x setup.sh train_eval_aif.sh train_eval_coded.sh compare_models.sh
./setup.sh

# 2. Train and evaluate All-in-Focus RGB model
./train_eval_aif.sh

# 3. Train and evaluate Coded Aperture RGB model
./train_eval_coded.sh

# 4. Compare results from both models
./compare_models.sh
```

**Or run everything sequentially:**
```bash
./train_eval_aif.sh && ./train_eval_coded.sh && ./compare_models.sh
```

### What Each Script Does

**train_eval_aif.sh** - All-in-Focus RGB Model (25 pts)
- Trains on regular RGB images from `rgb/` directories
- Training datasets: NYUv2 + UMDCodedVO-LivingRoom
- Testing datasets: UMDCodedVO-Corridor + UMDCodedVO-DiningRoom
- Generates predictions and visualizations with viridis colormap

**train_eval_coded.sh** - Coded Aperture RGB Model (25 pts)
- Trains on coded aperture images from `Codedphasecam-27Linear/` directories
- Training datasets: NYUv2 + UMDCodedVO-LivingRoom
- Testing datasets: UMDCodedVO-Corridor + UMDCodedVO-DiningRoom
- Generates predictions and visualizations with plasma colormap

**compare_models.sh** - Model Comparison
- Compares Abs Rel and RMSE metrics from both models
- Shows side-by-side results for Corridor and DiningRoom datasets
- Provides paths to all visualization outputs

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

### Using Bash Scripts (Recommended)
The bash scripts handle the complete pipeline automatically.

**For All-in-Focus (AiF/Pinhole) RGB images:**
```bash
./train_eval_aif.sh
```
This trains on regular RGB images from the `rgb/` directories.

**For Coded Aperture RGB images:**
```bash
./train_eval_coded.sh
```
This trains on coded images from the `Codedphasecam-27Linear/` directories.

Both scripts will:
1. ✓ Train the model (80 epochs)
2. ✓ Evaluate on test datasets
3. ✓ Generate visualizations
4. ✓ Save metrics and comparisons

### Manual Training
Basic training with All-in-Focus RGB:
```bash
python train.py --dataset_dir ./datasets --checkpoint_dir ./checkpoints --image_type rgb
```

Training with Coded Aperture RGB:
```bash
python train.py --dataset_dir ./datasets --checkpoint_dir ./checkpoints --image_type coded
```

With Weights & Biases logging:
```bash
python train.py --dataset_dir ./datasets --checkpoint_dir ./checkpoints --image_type rgb --use_wandb --wandb_project your_project_name
```

Training will save:
- `best.pt` - Best model based on L1 error < 3m
- `epoch_N.pt` - Checkpoint every 10 epochs
- `final.pt` - Final model after all epochs

## Evaluation

### Using Bash Scripts
Evaluation is included in the training scripts, but you can also run manually:

**Evaluate AiF RGB model:**
```bash
python eval.py --checkpoint ./checkpoints/MetricWeightedLoss_RGB/best.pt \
               --dataset_dir ./datasets \
               --output_dir ./results_aif \
               --image_type rgb
```

**Evaluate Coded RGB model:**
```bash
python eval.py --checkpoint ./checkpoints/MetricWeightedLoss_CODED/best.pt \
               --dataset_dir ./datasets \
               --output_dir ./results_coded \
               --image_type coded
```

This will:
- Evaluate on DiningRoom and Corridor test sets
- Save predicted depth maps to `results_{type}/{dataset_name}/pred_depth/`
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

### Automated (Using Bash Scripts)
```bash
# Run complete pipeline for both models
./train_eval_aif.sh && ./train_eval_coded.sh && ./compare_models.sh
```

### Manual
```bash
# 1. Train AiF RGB model
python train.py --dataset_dir ./datasets --checkpoint_dir ./checkpoints --image_type rgb

# 2. Train Coded RGB model  
python train.py --dataset_dir ./datasets --checkpoint_dir ./checkpoints --image_type coded

# 3. Evaluate AiF model
python eval.py --checkpoint ./checkpoints/MetricWeightedLoss_RGB/best.pt \
               --dataset_dir ./datasets \
               --output_dir ./results_aif \
               --image_type rgb

# 4. Evaluate Coded model
python eval.py --checkpoint ./checkpoints/MetricWeightedLoss_CODED/best.pt \
               --dataset_dir ./datasets \
               --output_dir ./results_coded \
               --image_type coded

# 5. Visualize AiF results
python visualize.py --pred_dir ./results_aif \
                    --dataset_dir ./datasets \
                    --output_dir ./visualizations_aif \
                    --colormap viridis

# 6. Visualize Coded results
python visualize.py --pred_dir ./results_coded \
                    --dataset_dir ./datasets \
                    --output_dir ./visualizations_coded \
                    --colormap plasma
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
├── config.py                # Configuration and loss functions
├── data.py                  # Dataset loader
├── model.py                 # U-Net architecture
├── train.py                 # Training script
├── eval.py                  # Evaluation script
├── visualize.py             # Visualization and comparison tool
├── train_eval_aif.sh        # Complete pipeline for AiF RGB
├── train_eval_coded.sh      # Complete pipeline for Coded RGB
├── compare_models.sh        # Compare both models
├── setup.sh                 # Setup helper
├── datasets/                # Your data (create this)
├── checkpoints/             # Saved models (auto-created)
│   ├── MetricWeightedLoss_RGB/    # AiF model
│   └── MetricWeightedLoss_CODED/  # Coded model
├── results_aif/             # AiF predictions (auto-created)
├── results_coded/           # Coded predictions (auto-created)
├── visualizations_aif/      # AiF comparisons (auto-created)
└── visualizations_coded/    # Coded comparisons (auto-created)
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