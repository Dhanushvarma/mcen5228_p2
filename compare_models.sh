#!/bin/bash

################################################################################
# Compare Results from AiF RGB vs Coded Aperture RGB Models
#
# Compares quantitative metrics (Abs Rel, RMSE) from both models
################################################################################

set -e

echo "=============================================================="
echo "  Model Comparison: AiF RGB vs Coded Aperture RGB"
echo "=============================================================="
echo ""

# Paths to metrics files
AIF_CORRIDOR="./visualizations_aif/Corridor/metrics.txt"
AIF_DINING="./visualizations_aif/DiningRoom/metrics.txt"
CODED_CORRIDOR="./visualizations_coded/Corridor/metrics.txt"
CODED_DINING="./visualizations_coded/DiningRoom/metrics.txt"

# Check if files exist
if [ ! -f "$AIF_CORRIDOR" ] || [ ! -f "$AIF_DINING" ]; then
    echo "ERROR: AiF RGB results not found!"
    echo "Please run ./train_eval_aif.sh first"
    exit 1
fi

if [ ! -f "$CODED_CORRIDOR" ] || [ ! -f "$CODED_DINING" ]; then
    echo "ERROR: Coded RGB results not found!"
    echo "Please run ./train_eval_coded.sh first"
    exit 1
fi

echo "Comparing metrics on test datasets:"
echo "  - UMDCodedVO-Corridor"
echo "  - UMDCodedVO-DiningRoom"
echo ""
echo "=============================================================="

# Function to extract metric value
extract_metric() {
    grep "$2" "$1" | awk '{print $NF}'
}

# Extract metrics
echo ""
echo "Corridor Dataset Results:"
echo "--------------------------------------------------------------"
printf "%-25s %-15s %-15s\n" "Metric" "AiF RGB" "Coded RGB"
echo "--------------------------------------------------------------"

AIF_ABS_REL=$(extract_metric "$AIF_CORRIDOR" "abs_rel")
CODED_ABS_REL=$(extract_metric "$CODED_CORRIDOR" "abs_rel")
printf "%-25s %-15s %-15s\n" "Abs Rel" "$AIF_ABS_REL" "$CODED_ABS_REL"

AIF_RMSE=$(extract_metric "$AIF_CORRIDOR" "rmse")
CODED_RMSE=$(extract_metric "$CODED_CORRIDOR" "rmse")
printf "%-25s %-15s %-15s\n" "RMSE (m)" "$AIF_RMSE" "$CODED_RMSE"

AIF_MAE=$(extract_metric "$AIF_CORRIDOR" "mae")
CODED_MAE=$(extract_metric "$CODED_CORRIDOR" "mae")
printf "%-25s %-15s %-15s\n" "MAE (m)" "$AIF_MAE" "$CODED_MAE"

AIF_SIGMA=$(extract_metric "$AIF_CORRIDOR" "sigma_1.25")
CODED_SIGMA=$(extract_metric "$CODED_CORRIDOR" "sigma_1.25")
printf "%-25s %-15s %-15s\n" "δ < 1.25" "$AIF_SIGMA" "$CODED_SIGMA"

echo ""
echo "DiningRoom Dataset Results:"
echo "--------------------------------------------------------------"
printf "%-25s %-15s %-15s\n" "Metric" "AiF RGB" "Coded RGB"
echo "--------------------------------------------------------------"

AIF_ABS_REL=$(extract_metric "$AIF_DINING" "abs_rel")
CODED_ABS_REL=$(extract_metric "$CODED_DINING" "abs_rel")
printf "%-25s %-15s %-15s\n" "Abs Rel" "$AIF_ABS_REL" "$CODED_ABS_REL"

AIF_RMSE=$(extract_metric "$AIF_DINING" "rmse")
CODED_RMSE=$(extract_metric "$CODED_DINING" "rmse")
printf "%-25s %-15s %-15s\n" "RMSE (m)" "$AIF_RMSE" "$CODED_RMSE"

AIF_MAE=$(extract_metric "$AIF_DINING" "mae")
CODED_MAE=$(extract_metric "$CODED_DINING" "mae")
printf "%-25s %-15s %-15s\n" "MAE (m)" "$AIF_MAE" "$CODED_MAE"

AIF_SIGMA=$(extract_metric "$AIF_DINING" "sigma_1.25")
CODED_SIGMA=$(extract_metric "$CODED_DINING" "sigma_1.25")
printf "%-25s %-15s %-15s\n" "δ < 1.25" "$AIF_SIGMA" "$CODED_SIGMA"

echo "--------------------------------------------------------------"
echo ""
echo "Visualizations:"
echo "  AiF RGB Model:       ./visualizations_aif/"
echo "  Coded RGB Model:     ./visualizations_coded/"
echo ""
echo "=============================================================="
echo ""
