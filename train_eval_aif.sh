#!/bin/bash

################################################################################
# Train and Evaluate Monocular Depth Estimation with All-in-Focus (AiF) RGB
#
# Training Data: NYUv2 + UMDCodedVO-LivingRoom
# Testing Data:  UMDCodedVO-Corridor + UMDCodedVO-DiningRoom
#
# Input: Regular RGB images (pinhole/all-in-focus)
# Output: Depth maps (0-6m metric)
################################################################################

set -e  # Exit on error

echo "=============================================================="
echo "  Monocular Depth Estimation - All-in-Focus RGB Model"
echo "=============================================================="
echo ""

# Configuration
DATASET_DIR="./datasets"
CHECKPOINT_DIR="./checkpoints"
RESULTS_DIR="./results_aif"
VIS_DIR="./visualizations_aif"
IMAGE_TYPE="rgb"  # Use regular RGB images (not coded)
COLORMAP="viridis"

# Optional: Use W&B for training logging
USE_WANDB=false  # Set to true to enable Weights & Biases
WANDB_PROJECT="depth_estimation_aif"

echo "Configuration:"
echo "  Dataset Directory:    $DATASET_DIR"
echo "  Checkpoint Directory: $CHECKPOINT_DIR"
echo "  Results Directory:    $RESULTS_DIR"
echo "  Visualization Dir:    $VIS_DIR"
echo "  Image Type:           All-in-Focus (AiF) RGB"
echo "  Colormap:             $COLORMAP"
echo "=============================================================="
echo ""

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found: $DATASET_DIR"
    echo "Please create the directory and organize your data as:"
    echo "  $DATASET_DIR/"
    echo "    ├── LivingRoom1/    (training)"
    echo "    │   ├── rgb/"
    echo "    │   └── depth/"
    echo "    ├── nyu_data/       (training)"
    echo "    │   ├── rgb/"
    echo "    │   └── depth/"
    echo "    ├── Corridor/       (testing)"
    echo "    │   ├── rgb/"
    echo "    │   └── depth/"
    echo "    └── DiningRoom/     (testing)"
    echo "        ├── rgb/"
    echo "        └── depth/"
    exit 1
fi

# Create output directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$VIS_DIR"

################################################################################
# Step 1: Training
################################################################################
echo ""
echo "=============================================================="
echo "  Step 1: Training Model on AiF RGB Images"
echo "=============================================================="
echo ""
echo "Training on:"
echo "  - NYUv2 dataset"
echo "  - UMDCodedVO-LivingRoom dataset"
echo ""

TRAIN_CMD="python train.py \
    --dataset_dir $DATASET_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --image_type $IMAGE_TYPE"

if [ "$USE_WANDB" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_wandb --wandb_project $WANDB_PROJECT"
fi

echo "Running: $TRAIN_CMD"
echo ""

eval $TRAIN_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
    echo "  Models saved to: $CHECKPOINT_DIR/MetricWeightedLoss_RGB/"
else
    echo ""
    echo "✗ Training failed!"
    exit 1
fi

################################################################################
# Step 2: Evaluation
################################################################################
echo ""
echo "=============================================================="
echo "  Step 2: Evaluating on Test Datasets"
echo "=============================================================="
echo ""
echo "Testing on:"
echo "  - UMDCodedVO-Corridor"
echo "  - UMDCodedVO-DiningRoom"
echo ""

CHECKPOINT_FILE="$CHECKPOINT_DIR/MetricWeightedLoss_RGB/best.pt"

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "ERROR: Checkpoint file not found: $CHECKPOINT_FILE"
    echo "Training may not have completed successfully."
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT_FILE"
echo ""

python eval.py \
    --checkpoint "$CHECKPOINT_FILE" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$RESULTS_DIR" \
    --image_type "$IMAGE_TYPE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation completed successfully!"
    echo "  Predictions saved to: $RESULTS_DIR/"
else
    echo ""
    echo "✗ Evaluation failed!"
    exit 1
fi

################################################################################
# Step 3: Visualization
################################################################################
echo ""
echo "=============================================================="
echo "  Step 3: Generating Visualizations"
echo "=============================================================="
echo ""
echo "Creating comparison images with $COLORMAP colormap..."
echo ""

python visualize.py \
    --pred_dir "$RESULTS_DIR" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$VIS_DIR" \
    --colormap "$COLORMAP"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Visualization completed successfully!"
    echo "  Visualizations saved to: $VIS_DIR/"
else
    echo ""
    echo "✗ Visualization failed!"
    exit 1
fi

################################################################################
# Step 4: Summary
################################################################################
echo ""
echo "=============================================================="
echo "  Pipeline Completed Successfully!"
echo "=============================================================="
echo ""
echo "Summary:"
echo "  Model Type:       All-in-Focus (AiF) RGB"
echo "  Training Data:    NYUv2 + UMDCodedVO-LivingRoom"
echo "  Testing Data:     UMDCodedVO-Corridor + UMDCodedVO-DiningRoom"
echo ""
echo "Output Locations:"
echo "  └── Checkpoints:    $CHECKPOINT_DIR/MetricWeightedLoss_RGB/"
echo "      ├── best.pt              (best model)"
echo "      ├── final.pt             (final model)"
echo "      └── epoch_*.pt           (periodic checkpoints)"
echo ""
echo "  └── Predictions:    $RESULTS_DIR/"
echo "      ├── Corridor/pred_depth/"
echo "      └── DiningRoom/pred_depth/"
echo ""
echo "  └── Visualizations: $VIS_DIR/"
echo "      ├── Corridor/"
echo "      │   ├── comparison_*.png (side-by-side comparisons)"
echo "      │   └── metrics.txt      (quantitative results)"
echo "      └── DiningRoom/"
echo "          ├── comparison_*.png"
echo "          └── metrics.txt"
echo ""
echo "Key Metrics Files:"
echo "  - $VIS_DIR/Corridor/metrics.txt"
echo "  - $VIS_DIR/DiningRoom/metrics.txt"
echo ""
echo "=============================================================="
echo "  All Done! 🎉"
echo "=============================================================="
