# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Author : Code_charon
# @Time : 2024/11/15 20:40
# ------------------------------------------------------------------------------

import os
import torch

class Config:
    # Path configurations
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'outputs')
    CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, 'checkpoints')
    LOG_DIR = os.path.join(OUTPUT_ROOT, 'logs')
    PREDICTION_DIR = os.path.join(OUTPUT_ROOT, 'predictions')
    LOSS_CURVE_DIR = os.path.join(OUTPUT_ROOT, 'loss_curves')

    # Create directories if they don't exist
    for directory in [CHECKPOINT_DIR, LOG_DIR, PREDICTION_DIR, LOSS_CURVE_DIR]:
        os.makedirs(directory, exist_ok=True)

    # Dataset configurations
    CURRENT_DATASET = 'data1'
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2

    # Time format
    TIME_FORMAT = '%d/%m/%Y %H:%M:%S'
    OUTPUT_TIME_FORMAT = '%Y/%m/%d %H:%M'
    TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

    # Vehicle types
    VEHICLE_TYPES = ['B1', 'B2', 'B3', 'T1', 'T2', 'T3']

    # Prediction horizons in minutes
    PREDICTION_HORIZONS = [5, 10, 15, 30, 60]

    # Data preprocessing
    NORMALIZATION = 'z-score'
    TIME_GRANULARITY = 5

    # Model hyperparameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GCN parameters
    GCN_HIDDEN_CHANNELS = [64, 32]
    GCN_DROPOUT = 0.2
    NUM_RELATIONS = 3
    RELATION_AGGREGATION = 'concat'
    DISTANCE_THRESHOLD = 10.0
    SIMILARITY_THRESHOLD = 0.1

    # TCN parameters
    TCN_KERNEL_SIZE = 3
    TCN_CHANNELS = [32, 64, 128]
    TCN_DROPOUT = 0.2
    SEQUENCE_LENGTH = 12
    HORIZON = len(PREDICTION_HORIZONS)

    # Bi-TSENet parameters
    BIDIRECTIONAL = True
    FINAL_FC_HIDDEN = 64

    # Training parameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.005
    EPOCHS = 100
    PATIENCE = 30
    SCHEDULER_STEP = 10
    SCHEDULER_GAMMA = 0.7

    # Visualization
    MAX_NODES_TO_PLOT = 10

    # Log file
    LOG_FILE = os.path.join(LOG_DIR, "bi_tsenet.log")