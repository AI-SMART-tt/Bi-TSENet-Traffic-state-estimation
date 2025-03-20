import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from models.bi_tsenet import BiTSENet
from preprocess import TrafficDataProcessor
from metrics import calculate_grouped_metrics


def test_model(config, logger):
    # Initialize data processor and load data first to set NUM_NODES
    data_processor = TrafficDataProcessor(config)
    _, _, test_loader, adj_matrices = data_processor.generate_datasets()

    # Get raw test data and timestamps
    test_data_raw = data_processor.test_data_raw
    test_timestamps_raw = data_processor.test_timestamps

    # Initialize the model after NUM_NODES is set
    model = BiTSENet(config).to(config.DEVICE)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{config.CURRENT_DATASET}_best_model.pth")

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from epoch {checkpoint['epoch'] + 1}")

    # Testing
    model.eval()
    all_predictions = []
    all_targets = []
    all_timestamps = []

    logger.info(f"Starting testing on {config.CURRENT_DATASET} dataset with {config.NUM_NODES} nodes")

    with torch.no_grad():
        for data, target, batch_times in test_loader:
            output = model(data, adj_matrices)

            # Collect predictions and targets
            pred = output.transpose(1, 2).cpu().numpy()
            true = target.cpu().numpy()

            all_predictions.append(pred)
            all_targets.append(true)
            all_timestamps.extend(batch_times)

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Denormalize predictions and targets
    all_predictions_denorm = np.array([
        data_processor.denormalize_data(all_predictions[:, i, :])
        for i in range(config.HORIZON)
    ])
    all_targets_denorm = np.array([
        data_processor.denormalize_data(all_targets[:, i, :])
        for i in range(config.HORIZON)
    ])

    # Reshape to [horizon, samples, nodes]
    all_predictions_denorm = all_predictions_denorm.transpose(0, 1, 2)
    all_targets_denorm = all_targets_denorm.transpose(0, 1, 2)

    # Ensure non-negative values
    all_predictions_denorm = np.maximum(all_predictions_denorm, 0.0)

    logger.info("Applied non-negative constraint to predictions")

    # Get vehicle types
    vehicle_types = data_processor.vehicle_types

    # Create predictions directory if it doesn't exist
    real_flow_dir = os.path.join(config.PREDICTION_DIR, 'real_flow')
    pred_flow_dir = os.path.join(config.PREDICTION_DIR, 'pred_flow')
    os.makedirs(real_flow_dir, exist_ok=True)
    os.makedirs(pred_flow_dir, exist_ok=True)

    # Calculate metrics for each horizon
    all_metrics = []
    # Define B-type and T-type vehicle indices
    b_indices = [i for i, vtype in enumerate(vehicle_types) if vtype.startswith('B')]
    t_indices = [i for i, vtype in enumerate(vehicle_types) if vtype.startswith('T')]

    for h in range(config.HORIZON):
        logger.info(f"Calculating metrics for horizon {h + 1}")

        # Current horizon predictions and targets
        horizon_pred = all_predictions_denorm[h]
        horizon_true = all_targets_denorm[h]

        # Calculate metrics for individual nodes and groups
        node_metrics, b_metrics, t_metrics, overall_metrics = calculate_grouped_metrics(
            horizon_true, horizon_pred, b_indices=b_indices, t_indices=t_indices
        )

        logger.info(f"Horizon {h + 1} Overall Metrics: MAE={overall_metrics['MAE']:.4f}, RMSE={overall_metrics['RMSE']:.4f}, R2={overall_metrics['R2']:.4f}")
        logger.info(f"Horizon {h + 1} B Group Metrics: MAE={b_metrics['MAE']:.4f}, RMSE={b_metrics['RMSE']:.4f}, R2={b_metrics['R2']:.4f}")
        logger.info(f"Horizon {h + 1} T Group Metrics: MAE={t_metrics['MAE']:.4f}, RMSE={t_metrics['RMSE']:.4f}, R2={t_metrics['R2']:.4f}")

        all_metrics.append((node_metrics, b_metrics, t_metrics, overall_metrics))

    # Convert string timestamps back to datetime for processing
    timestamps = [datetime.strptime(ts, config.TIMESTAMP_FORMAT) for ts in all_timestamps]

    # Process time alignment issue
    seq_offset = config.SEQUENCE_LENGTH - 1

    # Convert test timestamps to datetime objects
    test_timestamps_datetime = []
    for ts in test_timestamps_raw:
        if isinstance(ts, str):
            test_timestamps_datetime.append(datetime.strptime(ts, config.TIMESTAMP_FORMAT))
        else:
            test_timestamps_datetime.append(ts)

    logger.info(f"Original timestamps - prediction starts: {timestamps[0].strftime('%Y/%m/%d %H:%M:%S')}, " +
                f"real data starts: {test_timestamps_datetime[0].strftime('%Y/%m/%d %H:%M:%S')}")

    if seq_offset < len(test_timestamps_datetime):
        # Get aligned test data and timestamps
        aligned_test_data = test_data_raw[seq_offset:seq_offset + len(timestamps)]
        aligned_test_timestamps = test_timestamps_datetime[seq_offset:seq_offset + len(timestamps)]

        # Ensure consistent lengths
        min_length = min(len(timestamps), len(aligned_test_data))
        timestamps = timestamps[:min_length]
        aligned_test_data = aligned_test_data[:min_length]
        aligned_test_timestamps = aligned_test_timestamps[:min_length]

        logger.info(f"Aligned timestamps - prediction starts: {timestamps[0].strftime('%Y/%m/%d %H:%M:%S')}, " +
                    f"real data starts: {aligned_test_timestamps[0].strftime('%Y/%m/%d %H:%M:%S')}")
    else:
        logger.warning("Sequence offset exceeds test data length, using original test data")
        aligned_test_data = test_data_raw
        aligned_test_timestamps = test_timestamps_datetime
        min_length = min(len(timestamps), len(aligned_test_data))

    # Create files for each node (vehicle type)
    for node_idx, vehicle_type in enumerate(vehicle_types):
        node_number = node_idx + 1

        # 1. Create prediction CSV file for this node
        prediction_rows = []

        for t_idx, timestamp in enumerate(timestamps[:min_length]):
            base_time_str = timestamp.strftime(config.OUTPUT_TIME_FORMAT)

            for h_idx, minutes in enumerate(config.PREDICTION_HORIZONS):
                # Skip if beyond our prediction horizon
                if h_idx >= config.HORIZON:
                    continue

                pred_time = timestamp + timedelta(minutes=minutes)
                pred_time_str = pred_time.strftime(config.OUTPUT_TIME_FORMAT)

                row = {
                    'time': base_time_str,
                    'horizon': minutes,
                    'pred_time': pred_time_str,
                    'B1': int(round(all_predictions_denorm[h_idx, t_idx, 0])),
                    'B2': int(round(all_predictions_denorm[h_idx, t_idx, 1])),
                    'B3': int(round(all_predictions_denorm[h_idx, t_idx, 2])),
                    'T1': int(round(all_predictions_denorm[h_idx, t_idx, 3])),
                    'T2': int(round(all_predictions_denorm[h_idx, t_idx, 4])),
                    'T3': int(round(all_predictions_denorm[h_idx, t_idx, 5]))
                }

                prediction_rows.append(row)

        # Create prediction DataFrame
        prediction_df = pd.DataFrame(prediction_rows)

        # Save prediction file
        prediction_path = os.path.join(pred_flow_dir, f"prediction_G{node_number:03d}.csv")
        prediction_df.to_csv(prediction_path, index=False)
        logger.info(f"Saved prediction file for node {node_number}: {prediction_path}")

        # 2. Create real values DataFrame - using raw test data
        real_rows = []

        # Use same timestamps as predictions
        for i in range(min_length):
            real_row = {
                'Time': aligned_test_timestamps[i].strftime(config.OUTPUT_TIME_FORMAT),
                'B1': int(round(aligned_test_data[i, 0])),
                'B2': int(round(aligned_test_data[i, 1])),
                'B3': int(round(aligned_test_data[i, 2])),
                'T1': int(round(aligned_test_data[i, 3])),
                'T2': int(round(aligned_test_data[i, 4])),
                'T3': int(round(aligned_test_data[i, 5]))
            }
            real_rows.append(real_row)

        real_df = pd.DataFrame(real_rows)

        # Save real values file
        real_path = os.path.join(real_flow_dir, f"real_G{node_number:03d}.csv")
        real_df.to_csv(real_path, index=False)
        logger.info(f"Saved real values file for node {node_number}: {real_path}")

    # Check time alignment
    if min_length > 0:
        logger.info(f"Time alignment check - First timestamps:")
        logger.info(f"Prediction file starts at: {timestamps[0].strftime('%Y/%m/%d %H:%M:%S')}")
        logger.info(f"Real data file starts at: {aligned_test_timestamps[0].strftime('%Y/%m/%d %H:%M:%S')}")

        if min_length > 1:
            logger.info(f"Last timestamps:")
            logger.info(f"Prediction file ends at: {timestamps[min_length - 1].strftime('%Y/%m/%d %H:%M:%S')}")
            logger.info(
                f"Real data file ends at: {aligned_test_timestamps[min_length - 1].strftime('%Y/%m/%d %H:%M:%S')}")

    logger.info("Testing completed successfully")
    return all_metrics