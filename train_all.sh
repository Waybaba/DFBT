#!/bin/bash
# DFBT Complete Training Script
# Simple sequential execution: belief model training -> SAC training

set -e  # Exit on error

echo "=========================================="
echo "DFBT Complete Training Pipeline"
echo "=========================================="
echo ""

# Stage 1: Train belief models
echo "Stage 1: Training Belief Models"
echo "----------------------------------------"

echo "Training halfcheetah belief model..."
python3 scripts/train_dfbt.py --dataset_name halfcheetah --delay 128 --seed 2025

echo "Training hopper belief model..."
python3 scripts/train_dfbt.py --dataset_name hopper --delay 128 --seed 2025

echo "Training walker2d belief model..."
python3 scripts/train_dfbt.py --dataset_name walker2d --delay 128 --seed 2025

echo "Training ant belief model..."
python3 scripts/train_dfbt.py --dataset_name ant --delay 128 --seed 2025

echo ""
echo "Stage 1 completed!"
echo ""

# Stage 2: Train SAC agents
echo "Stage 2: Training SAC Agents"
echo "----------------------------------------"

echo "Training HalfCheetah-v2 SAC..."
python3 scripts/train_dfbt_sac.py --env_name HalfCheetah-v2 --delay 128 --total_step 5000000 --seed 2025

echo "Training Hopper-v2 SAC..."
python3 scripts/train_dfbt_sac.py --env_name Hopper-v2 --delay 128 --total_step 5000000 --seed 2025

echo "Training Walker2d-v2 SAC..."
python3 scripts/train_dfbt_sac.py --env_name Walker2d-v2 --delay 128 --total_step 5000000 --seed 2025

echo "Training Ant-v2 SAC..."
python3 scripts/train_dfbt_sac.py --env_name Ant-v2 --delay 128 --total_step 5000000 --seed 2025

echo ""
echo "=========================================="
echo "All training completed!"
echo "=========================================="
