#!/bin/bash
#SBATCH --job-name=FID_calibration
#SBATCH --output=slurm-%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1



: "${HF_TOKEN:?Set HF_TOKEN env var: export HF_TOKEN=hf_xxx}"

# Computes FID for the calibration baselines produced by
# experiments/fid_calibration_experiment.py. Run that script first.

#python -m pytorch_fid results_fid_calibration/images/baseline_A results_fid_calibration/images/baseline_B

# CFG variation
python -m pytorch_fid results_fid_calibration/images/baseline_A results_fid_calibration/images/cfg_6.5
python -m pytorch_fid results_fid_calibration/images/baseline_A results_fid_calibration/images/cfg_8.5

# Steps variation
python -m pytorch_fid results_fid_calibration/images/baseline_A results_fid_calibration/images/steps_15
python -m pytorch_fid results_fid_calibration/images/baseline_A results_fid_calibration/images/steps_10

# Scheduler variation
python -m pytorch_fid results_fid_calibration/images/baseline_A results_fid_calibration/images/euler

