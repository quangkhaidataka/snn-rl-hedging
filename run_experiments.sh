#!/bin/bash

# Script: run_experiments.sh
# Description: Sequentially runs Python experiments with caffeinate to prevent sleep.
#              Each command trains/evaluates a model with different settings (MJD_kappa variants with SNN_T3).
#              Assumes main.py is in the current directory and handles --settings flag dynamically.
# Author: [Your Name]
# Date: September 23, 2025
# Requirements: macOS (for caffeinate), Python environment with required libraries.
# Usage: ./run_experiments.sh
# Notes: 
#   - This script runs commands one after another. If one fails, it will stop (use 'set -e' for strict error handling).
#   - For scalability, you can parameterize the kappa values or read from config.cfg in future extensions.
#   - Output/logs from each run will be printed to stdout; redirect to files if needed (e.g., add ' > log_kappa1.txt' per command).

set -e  # Exit immediately if any command fails



echo "Starting experiment 1: Heston_kappa1_snn_T2_29000"
caffeinate python testing.py --test --model Heston_kappa1_snn_T2_29000


echo "Starting experiment 2: Heston_kappa2_snn_T2_29000"
caffeinate python testing.py --test --model Heston_kappa2_snn_T2_29000


echo "Starting experiment 3: Heston_kappa3_snn_T2_29000"
caffeinate python testing.py --test --model Heston_kappa3_snn_T2_29000


echo "Starting experiment 8: SABR_kappa1_snn_T2_34000"
caffeinate python testing.py --test --model SABR_kappa1_snn_T2_34000

echo "Starting experiment 9: SABR_kappa2_snn_T2_38000"
caffeinate python testing.py --test --model SABR_kappa2_snn_T2_38000

echo "Starting experiment 10: SABR_kappa3_snn_T2_29000"
caffeinate python testing.py --test --model SABR_kappa3_snn_T2_29000

echo "All experiments completed successfully."