# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Author : Code_charon
# @Time : 2025/3/5 23:40
# ------------------------------------------------------------------------------

import os
import sys
import argparse
import time
import logging
import pandas as pd
from typing import Dict

# Add the project root to the path so Python can find our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the validator, but we'll patch its methods before using it
from models.traffic_physical_estimation.blind_segment_estimation import AdvancedSegmentValidator


# Replacement for the _load_traffic_flow_data method
def patched_load_traffic_flow_data(self) -> Dict:
    """Patched method to load historical traffic flow data with column name flexibility"""
    if not self.traffic_flow_dir:
        raise ValueError("Historical traffic flow data directory not provided")

    self.logger.info(f"Loading historical traffic flow data: {self.traffic_flow_dir}")
    traffic_flow_data = {}

    for gantry_id in self.all_gantry_ids:
        flow_file = os.path.join(self.traffic_flow_dir, f"trafficflow_{gantry_id}.csv")
        if os.path.exists(flow_file):
            try:
                flow_data = pd.read_csv(flow_file)
                self.logger.info(f"Loaded flow file: {flow_file} with columns: {flow_data.columns.tolist()}")

                # Check which time column exists in the data
                time_column = None
                for col_name in ['Time', 'time', 'TIME', 'timestamp', 'TIMESTAMP', 'Timestamp']:
                    if col_name in flow_data.columns:
                        time_column = col_name
                        break

                if time_column is None:
                    self.logger.warning(
                        f"No time column found in {flow_file}. Available columns: {flow_data.columns.tolist()}")
                    continue

                # Convert time format
                flow_data['time'] = pd.to_datetime(flow_data[time_column])
                traffic_flow_data[gantry_id] = flow_data

            except Exception as e:
                self.logger.error(f"Error loading flow file {flow_file}: {e}")
                continue

    self.logger.info(f"Loaded historical traffic flow data for {len(traffic_flow_data)} gantries")
    if len(traffic_flow_data) == 0:
        self.logger.warning("No traffic flow data was loaded. Check the data directory and file formats.")

    return traffic_flow_data


# Replacement for the _load_prediction_flow_data method
def patched_load_prediction_flow_data(self) -> Dict:
    """Patched method to load predicted traffic flow data with column name flexibility"""
    if not self.prediction_flow_dir:
        self.logger.info("Prediction traffic flow directory not provided, will use only historical data")
        return {}

    self.logger.info(f"Loading prediction traffic flow data: {self.prediction_flow_dir}")
    prediction_flow_data = {}

    for gantry_id in self.all_gantry_ids:
        prediction_file = os.path.join(self.prediction_flow_dir, f"prediction_{gantry_id}.csv")
        if os.path.exists(prediction_file):
            try:
                prediction_data = pd.read_csv(prediction_file)
                self.logger.info(
                    f"Loaded prediction file: {prediction_file} with columns: {prediction_data.columns.tolist()}")

                # Check which time columns exist in the data
                time_column = None
                pred_time_column = None

                for col_name in ['time', 'Time', 'TIME', 'timestamp', 'Timestamp']:
                    if col_name in prediction_data.columns:
                        time_column = col_name
                        break

                for col_name in ['pred_time', 'Pred_time', 'PRED_TIME', 'prediction_time', 'future_time']:
                    if col_name in prediction_data.columns:
                        pred_time_column = col_name
                        break

                if time_column is None:
                    self.logger.warning(
                        f"Missing base time column in {prediction_file}. Available columns: {prediction_data.columns.tolist()}")
                    continue

                prediction_data['time'] = pd.to_datetime(prediction_data[time_column])

                # If pred_time is missing, we'll try to create it from time + horizon
                if pred_time_column is None:
                    if 'horizon' in prediction_data.columns:
                        self.logger.info(f"Creating pred_time from time + horizon in {prediction_file}")
                        # Convert horizon (minutes) to timedelta and add to base time
                        prediction_data['pred_time'] = prediction_data.apply(
                            lambda row: row['time'] + pd.Timedelta(minutes=row['horizon']), axis=1)
                    else:
                        self.logger.warning(f"Missing both pred_time and horizon columns in {prediction_file}")
                        continue
                else:
                    prediction_data['pred_time'] = pd.to_datetime(prediction_data[pred_time_column])

                prediction_flow_data[gantry_id] = prediction_data

            except Exception as e:
                self.logger.error(f"Error loading prediction file {prediction_file}: {e}")
                continue

    self.logger.info(f"Loaded prediction traffic flow data for {len(prediction_flow_data)} gantries")
    if len(prediction_flow_data) == 0:
        self.logger.warning("No prediction flow data was loaded. Check the data directory and file formats.")

    return prediction_flow_data


def main():
    """Main entry point for Traffic Physical Estimation module"""
    parser = argparse.ArgumentParser(description="Advanced Blind Segment Traffic Flow Estimation and Validation")
    parser.add_argument("--road_data", type=str,
                        default="./data/data2/ETC_data_example/roadETC.csv",
                        help="Road segment data file path")
    parser.add_argument("--etc_data", type=str,
                        default="./data/data2/ETC_data_example/raw_data_all.csv",
                        help="ETC data file path")
    parser.add_argument("--flow_dir", type=str,
                        default="./data/data2/ETC_data_example/flow",
                        help="Historical traffic flow data directory")
    parser.add_argument("--pred_dir", type=str,
                        default="./outputs/predictions/pred_flow",
                        help="Prediction traffic flow data directory")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/physical_estimation_results",
                        help="Output directory")
    parser.add_argument("--parameter_dir", type=str,
                        default="./parameter_results",
                        help="Parameter saving directory")
    parser.add_argument("--time_window", type=int, default=5,
                        help="Time window (minutes)")
    parser.add_argument("--add_noise", action="store_true",
                        help="Add random noise to simulations")
    parser.add_argument("--noise_level", type=float, default=0.05,
                        help="Noise level (0-1)")
    parser.add_argument("--demand_times", type=int, nargs="+",
                        default=[5, 15, 30, 60],
                        help="Demand time options (minutes)")
    parser.add_argument("--position_weight", type=float, default=0.5,
                        help="Ramp position impact weight (0-1)")
    parser.add_argument("--state_threshold", type=float, default=1.0,
                        help="Traffic state threshold adjustment factor")
    parser.add_argument("--vehicle_factor", type=float, default=0.4,
                        help="Vehicle type impact factor")
    parser.add_argument("--time_factor", type=float, default=0.5,
                        help="Time period impact factor")
    parser.add_argument("--ramp_flow_damping", type=float, default=0.7,
                        help="Ramp flow damping factor")
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="Batch size for ETC data processing")
    parser.add_argument("--force_recalculate", action="store_true",
                        help="Force recalculation of parameters, ignore existing files")
    parser.add_argument("--log_file", type=str,
                        default="./outputs/logs/physical_estimation.log",
                        help="Log file path")

    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.parameter_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    # Set up basic logging until validator is initialized
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('main')
    logger.info("Starting Traffic Physical Estimation Module")
    logger.info(f"Using data from: {args.etc_data}")
    logger.info(f"Using prediction data from: {args.pred_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Patch the problematic methods
    AdvancedSegmentValidator._load_traffic_flow_data = patched_load_traffic_flow_data
    AdvancedSegmentValidator._load_prediction_flow_data = patched_load_prediction_flow_data

    # Run timing
    start_time = time.time()

    # Initialize and run validator
    try:
        validator = AdvancedSegmentValidator(
            road_data_path=args.road_data,
            etc_data_path=args.etc_data,
            traffic_flow_dir=args.flow_dir,
            prediction_flow_dir=args.pred_dir,
            output_dir=args.output_dir,
            parameter_dir=args.parameter_dir,
            time_window=args.time_window,
            add_noise=args.add_noise,
            noise_level=args.noise_level,
            demand_time_options=args.demand_times,
            position_weight=args.position_weight,
            state_threshold_factor=args.state_threshold,
            vehicle_factor=args.vehicle_factor,
            time_factor=args.time_factor,
            ramp_flow_damping=args.ramp_flow_damping,
            chunk_size=args.chunk_size,
            force_recalculate=args.force_recalculate,
            log_file=args.log_file
        )

        # Run validation
        validation_results = validator.run()

        # Report completion
        end_time = time.time()
        total_time = end_time - start_time
        validator.logger.info(f"Total execution time: {total_time:.2f} seconds")
        validator.logger.info(
            "Physical estimation completed successfully. Please check the results directory for detailed reports.")

    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)