# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Author : Code_charon
# @Time : 2024/11/15 20:40
# ------------------------------------------------------------------------------

import os
import argparse
import logging
import time
from configs import Config
from preprocess import TrafficDataProcessor
from train import train_model
from test import test_model
from visualization import visualize_results


def setup_logger(config):
    """Set up single logger for the whole project"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("bi_tsenet")


def main():
    parser = argparse.ArgumentParser(description='Bi-TSENet: Traffic State Estimation Project')
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'test', 'visualize', 'all'],
                        help='Mode to run (default: all)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.001)')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional TCN')
    parser.add_argument('--relation_aggregation', type=str, default='concat',
                        choices=['weighted_sum', 'attention', 'concat'],
                        help='Relation aggregation method (default: concat)')
    args = parser.parse_args()

    # Initialize config
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.BIDIRECTIONAL = args.bidirectional
    config.RELATION_AGGREGATION = args.relation_aggregation

    # Setup logger
    logger = setup_logger(config)

    # Log configuration
    logger.info(f"Running Bi-TSENet on {config.CURRENT_DATASET} dataset")
    logger.info(f"Configuration: batch_size={config.BATCH_SIZE}, epochs={config.EPOCHS}, "
                f"lr={config.LEARNING_RATE}, bidirectional={config.BIDIRECTIONAL}, "
                f"relation_aggregation={config.RELATION_AGGREGATION}")

    # Create necessary directories
    for directory in [config.OUTPUT_ROOT, config.CHECKPOINT_DIR, config.LOG_DIR,
                      config.PREDICTION_DIR, config.LOSS_CURVE_DIR]:
        os.makedirs(directory, exist_ok=True)

    start_time = time.time()

    # Run requested mode
    if args.mode in ['train', 'all']:
        logger.info("Starting training phase")
        train_model(config, logger)

    if args.mode in ['test', 'all']:
        logger.info("Starting testing phase")
        test_model(config, logger)

    if args.mode in ['visualize', 'all']:
        logger.info("Starting visualization phase")
        visualize_results(config)

    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info("Bi-TSENet execution completed successfully")


if __name__ == "__main__":
    main()