import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Set, Tuple, Union, Optional
from collections import defaultdict
import os
import datetime
import logging
from datetime import timedelta
from tqdm import tqdm
import time
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class SceneType(Enum):
    """Scene type enumeration"""
    SCENE1 = 1  # No ramps
    SCENE2 = 2  # Upstream with entry ramp
    SCENE3 = 3  # Upstream with exit ramp
    SCENE4 = 4  # Upstream with both entry and exit ramps
    SCENE5 = 5  # Special segment (tunnel, bridge, curved road)


class VehicleType(Enum):
    """Vehicle type enumeration"""
    B1 = "B1"  # Passenger vehicle type 1
    B2 = "B2"  # Passenger vehicle type 2
    B3 = "B3"  # Passenger vehicle type 3
    T1 = "T1"  # Truck type 1
    T2 = "T2"  # Truck type 2
    T3 = "T3"  # Truck type 3

    @classmethod
    def from_string(cls, value):
        try:
            return cls(value)
        except ValueError:
            # Return default B1 if no match
            return cls.B1


def _save_travel_times_to_csv(travel_times, output_dir="./outputs/physical_estimation_results"):
    """Save vehicle travel times to CSV file"""
    logger = logging.getLogger('validator')
    logger.info("Saving vehicle travel times to CSV file...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create data list for saving
    data_rows = []

    # Flatten nested dictionary
    for date_time_key, segments in travel_times.items():
        # Split date and time period
        date_str, time_period = date_time_key.split('_')

        for segment_id, vehicle_types in segments.items():
            for vehicle_type, travel_time in vehicle_types.items():
                data_rows.append({
                    'date': date_str,
                    'time_period': time_period,
                    'segment_id': segment_id,
                    'vehicle_type': vehicle_type,
                    'travel_time': travel_time
                })

    # Convert to DataFrame
    df = pd.DataFrame(data_rows)

    # Save to CSV
    travel_times_path = os.path.join(output_dir, "travel_times.csv")
    df.to_csv(travel_times_path, index=False)
    logger.info(f"Successfully saved to {travel_times_path}, total of {len(df)} records")


def _save_diversion_coefficients_to_csv(diversion_coefficients, output_dir="./outputs/physical_estimation_results"):
    """Save ramp diversion coefficients to CSV file"""
    logger = logging.getLogger('validator')
    logger.info("Saving ramp diversion coefficients to CSV file...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create data list for saving
    data_rows = []

    # Flatten nested dictionary
    for date_time_key, segments in diversion_coefficients.items():
        # Split date and time period
        date_str, time_period = date_time_key.split('_')

        for segment_id, vehicle_types in segments.items():
            for vehicle_type, coefficients in vehicle_types.items():
                data_rows.append({
                    'date': date_str,
                    'time_period': time_period,
                    'segment_id': segment_id,
                    'vehicle_type': vehicle_type,
                    'on_ramp_coefficient': coefficients['on_ramp'],
                    'off_ramp_coefficient': coefficients['off_ramp']
                })

    # Convert to DataFrame
    df = pd.DataFrame(data_rows)

    # Save to CSV
    coef_path = os.path.join(output_dir, "diversion_coefficients.csv")
    df.to_csv(coef_path, index=False)
    logger.info(f"Successfully saved to {coef_path}, total of {len(df)} records")


def _load_travel_times_from_csv(file_path):
    """Load vehicle travel times from CSV file"""
    logger = logging.getLogger('validator')
    logger.info(f"Loading vehicle travel times from CSV: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        logger.warning(f"File does not exist: {file_path}")
        return None

    # Load CSV data
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} travel time records")

        # Convert to nested dictionary structure
        travel_times = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        for _, row in df.iterrows():
            key = f"{row['date']}_{row['time_period']}"
            travel_times[key][row['segment_id']][row['vehicle_type']] = row['travel_time']

        return travel_times
    except Exception as e:
        logger.error(f"Error loading travel time file: {e}")
        return None


def _load_diversion_coefficients_from_csv(file_path):
    """Load ramp diversion coefficients from CSV file"""
    logger = logging.getLogger('validator')
    logger.info(f"Loading ramp diversion coefficients from CSV: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        logger.warning(f"File does not exist: {file_path}")
        return None

    # Load CSV data
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} diversion coefficient records")

        # Convert to nested dictionary structure
        diversion_coefficients = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for _, row in df.iterrows():
            key = f"{row['date']}_{row['time_period']}"
            diversion_coefficients[key][row['segment_id']][row['vehicle_type']] = {
                'on_ramp': row['on_ramp_coefficient'],
                'off_ramp': row['off_ramp_coefficient']
            }

        return diversion_coefficients
    except Exception as e:
        logger.error(f"Error loading diversion coefficient file: {e}")
        return None


class AdvancedSegmentValidator:
    def __init__(self,
                 road_data_path: str,
                 etc_data_path: str = None,
                 traffic_flow_dir: str = None,
                 prediction_flow_dir: str = None,
                 output_dir: str = "./outputs/physical_estimation_results",
                 parameter_dir: str = "./outputs/physical_estimation_results",
                 time_window: int = 5,
                 add_noise: bool = True,
                 noise_level: float = 0.05,  # 5% random noise
                 demand_time_options: List[int] = [5, 15, 30, 60],  # minutes
                 position_weight: float = 0.5,  # ramp position impact weight
                 state_threshold_factor: float = 1.0,  # traffic state threshold adjustment factor
                 vehicle_factor: float = 0.4,  # vehicle type impact factor
                 time_factor: float = 0.5,  # time period impact factor
                 ramp_flow_damping: float = 0.7,  # ramp flow damping factor
                 chunk_size: int = 100000,
                 force_recalculate: bool = False,  # force recalculation of parameters
                 log_file: str = None):  # log file path
        """
        Initialize advanced segment validator

        Parameters:
        road_data_path: Road segment data file path
        etc_data_path: Raw ETC data file path
        traffic_flow_dir: Historical traffic flow data directory
        prediction_flow_dir: Predicted traffic flow data directory
        output_dir: Output directory
        parameter_dir: Parameter saving directory
        time_window: Time window size (minutes)
        add_noise: Whether to add random noise to simulate real conditions
        noise_level: Random noise level (percentage)
        demand_time_options: Demand time options (minutes)
        position_weight: Ramp position impact weight (0-1)
        state_threshold_factor: Traffic state threshold adjustment factor
        vehicle_factor: Vehicle type impact factor
        time_factor: Time period impact factor
        ramp_flow_damping: Ramp flow damping factor
        chunk_size: Batch size for ETC data processing
        force_recalculate: Force recalculation of parameters
        log_file: Log file path
        """
        self.road_data_path = road_data_path
        self.etc_data_path = etc_data_path
        self.traffic_flow_dir = traffic_flow_dir
        self.prediction_flow_dir = prediction_flow_dir
        self.output_dir = output_dir
        self.parameter_dir = parameter_dir
        self.time_window = time_window
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.demand_time_options = demand_time_options
        self.position_weight = position_weight
        self.state_threshold_factor = state_threshold_factor
        self.vehicle_factor = vehicle_factor
        self.time_factor = time_factor
        self.ramp_flow_damping = ramp_flow_damping
        self.chunk_size = chunk_size
        self.force_recalculate = force_recalculate

        # Ensure output directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(parameter_dir, exist_ok=True)

        # Setup logging
        self._setup_logging(log_file)

        self.logger.info("====== Advanced Traffic Flow Estimation and Validation System ======")
        self.logger.info(f"Road data path: {road_data_path}")
        self.logger.info(f"ETC data path: {etc_data_path}")
        self.logger.info(f"Historical traffic flow directory: {traffic_flow_dir}")
        self.logger.info(f"Prediction traffic flow directory: {prediction_flow_dir}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Parameter directory: {parameter_dir}")
        self.logger.info(f"Time window: {time_window} minutes")
        self.logger.info(f"Ramp position impact weight: {position_weight}")
        self.logger.info(f"Ramp flow damping factor: {ramp_flow_damping}")
        self.logger.info(f"Force recalculation: {force_recalculate}")
        self.logger.info("==========================================")

        # Load data
        self.road_data = self._load_road_data()
        self.scene_units = self._create_scene_units()
        self.all_gantry_ids = self._get_all_gantry_ids()

        # Load ETC data and traffic flow data
        self.etc_data = self._load_etc_data()
        self.traffic_flow_data = self._load_traffic_flow_data()
        self.prediction_flow_data = self._load_prediction_flow_data()

        # Store validation results
        self.validation_results = []

        # Calculate or load travel times and diversion coefficients
        travel_times_path = os.path.join(self.parameter_dir, "travel_times.csv")
        diversion_coef_path = os.path.join(self.parameter_dir, "diversion_coefficients.csv")

        if self.force_recalculate or not os.path.exists(travel_times_path):
            self.logger.info("Calculating travel times...")
            self.travel_times = self._calculate_travel_times()
        else:
            self.logger.info("Loading travel times from existing file...")
            self.travel_times = _load_travel_times_from_csv(travel_times_path)
            if self.travel_times is None:
                self.logger.warning("Failed to load travel times, recalculating")
                self.travel_times = self._calculate_travel_times()

        if self.force_recalculate or not os.path.exists(diversion_coef_path):
            self.logger.info("Calculating diversion coefficients...")
            self.diversion_coefficients = self._calculate_dynamic_diversion_coefficients()
        else:
            self.logger.info("Loading diversion coefficients from existing file...")
            self.diversion_coefficients = _load_diversion_coefficients_from_csv(diversion_coef_path)
            if self.diversion_coefficients is None:
                self.logger.warning("Failed to load diversion coefficients, recalculating")
                self.diversion_coefficients = self._calculate_dynamic_diversion_coefficients()

        # Calculate traffic states from historical data
        self.traffic_states = self._calculate_traffic_states()

        # Log ramp position distribution for debugging
        self._log_ramp_positions()

    def _setup_logging(self, log_file):
        """Setup logging"""
        if log_file is None:
            log_file = os.path.join(self.output_dir, "validation.log")
        self.log_file = log_file

        # Configure logger
        self.logger = logging.getLogger('validator')
        self.logger.setLevel(logging.INFO)

        # Reset handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers = []

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def _log_ramp_positions(self):
        """Log ramp position distribution for debugging"""
        ramp_positions = {2: [], 3: [], 4: []}
        entry_positions = []
        exit_positions = []

        for segment_id, segment in self.scene_units.items():
            scene_type = segment['type'].value

            if scene_type == 2 and 'ramp_position' in segment:
                ramp_positions[2].append(segment['ramp_position'])
            elif scene_type == 3 and 'ramp_position' in segment:
                ramp_positions[3].append(segment['ramp_position'])
            elif scene_type == 4:
                if 'entry_ramp_position' in segment:
                    entry_positions.append(segment['entry_ramp_position'])
                if 'exit_ramp_position' in segment:
                    exit_positions.append(segment['exit_ramp_position'])

        self.logger.info("Ramp position distribution:")
        if ramp_positions[2]:
            self.logger.info(
                f"Entry ramp positions: min={min(ramp_positions[2]):.2f}, max={max(ramp_positions[2]):.2f}, avg={sum(ramp_positions[2]) / len(ramp_positions[2]):.2f}")
        if ramp_positions[3]:
            self.logger.info(
                f"Exit ramp positions: min={min(ramp_positions[3]):.2f}, max={max(ramp_positions[3]):.2f}, avg={sum(ramp_positions[3]) / len(ramp_positions[3]):.2f}")
        if entry_positions:
            self.logger.info(
                f"Combined entry ramp positions: min={min(entry_positions):.2f}, max={max(entry_positions):.2f}, avg={sum(entry_positions) / len(entry_positions):.2f}")
        if exit_positions:
            self.logger.info(
                f"Combined exit ramp positions: min={min(exit_positions):.2f}, max={max(exit_positions):.2f}, avg={sum(exit_positions) / len(exit_positions):.2f}")

    def _load_road_data(self) -> pd.DataFrame:
        """Load road segment data"""
        self.logger.info(f"Loading road data: {self.road_data_path}")
        road_data = pd.read_csv(self.road_data_path)
        self.logger.info(f"Loaded {len(road_data)} road segment records")
        return road_data

    def _create_scene_units(self) -> Dict:
        """Create scene units from road data, preserving ramp position information"""
        scene_units = {}
        for _, row in self.road_data.iterrows():
            scene_unit = {
                'type': SceneType(row['type']),
                'type_name': row.get('type_name', ''),
                'length': row['length'],
                'up_node': str(row['up_node']),
                'down_node': str(row['down_node']),
                'speed_limit': row.get('speed_limit', 100),
                'lanes': row.get('lanes', 3)  # Default 3 lanes
            }

            # Add complexity and flow impact factors
            if 'complexity_factor' in row:
                scene_unit['complexity_factor'] = row['complexity_factor']
            if 'flow_impact_factor' in row:
                scene_unit['flow_impact_factor'] = row['flow_impact_factor']

            # Save ramp properties
            if row['type'] in [2, 3, 4]:  # Ramp segments
                # Add ramp length and speed limit
                if 'ramp_length' in row:
                    scene_unit['ramp_length'] = row['ramp_length']
                if 'ramp_speed_limit' in row:
                    scene_unit['ramp_speed_limit'] = row['ramp_speed_limit']
                if 'ramp_lanes' in row:
                    scene_unit['ramp_lanes'] = row['ramp_lanes']

                # Add ramp positions
                if row['type'] == 2:  # Entry ramp
                    if 'ramp_position' in row:
                        scene_unit['ramp_position'] = row['ramp_position']
                    else:
                        scene_unit['ramp_position'] = 0.2  # Default at 20% of segment

                elif row['type'] == 3:  # Exit ramp
                    if 'ramp_position' in row:
                        scene_unit['ramp_position'] = row['ramp_position']
                    else:
                        scene_unit['ramp_position'] = 0.8  # Default at 80% of segment

                elif row['type'] == 4:  # Entry and exit ramps
                    if 'entry_ramp_position' in row:
                        scene_unit['entry_ramp_position'] = row['entry_ramp_position']
                    else:
                        scene_unit['entry_ramp_position'] = 0.2

                    if 'exit_ramp_position' in row:
                        scene_unit['exit_ramp_position'] = row['exit_ramp_position']
                    else:
                        scene_unit['exit_ramp_position'] = 0.8

            # Add special segment features
            if row['type'] == 5:  # Special segment
                if 'special_feature' in row:
                    scene_unit['special_feature'] = row['special_feature']
                if 'gradient' in row:
                    scene_unit['gradient'] = row['gradient']

            scene_units[row['id']] = scene_unit

        return scene_units

    def _get_all_gantry_ids(self) -> Set[str]:
        """Get all ETC gantry IDs"""
        gantry_ids = set()
        for unit in self.scene_units.values():
            gantry_ids.add(unit['up_node'])
            gantry_ids.add(unit['down_node'])
        return gantry_ids

    def _load_etc_data(self) -> pd.DataFrame:
        """Load ETC data"""
        if not self.etc_data_path:
            raise ValueError("ETC data path not provided")

        self.logger.info(f"Loading ETC data: {self.etc_data_path}")
        etc_data = pd.read_csv(self.etc_data_path)

        # Process ETC data
        etc_data['TRANSTIME'] = pd.to_datetime(etc_data['TRANSTIME'], format="%d/%m/%Y %H:%M:%S")
        etc_data = etc_data[['GANTRYID', 'VEHICLEPLATE', 'VEHICLETYPE', 'TRANSTIME']]

        # Parse vehicle plate numbers
        etc_data['VEHICLEPLATE'] = etc_data['VEHICLEPLATE'].apply(self._parse_vehicle_plate)

        # Filter to keep only gantries of interest
        etc_data = etc_data[etc_data['GANTRYID'].isin(self.all_gantry_ids)]

        # Add time period column
        etc_data['time_period'] = etc_data['TRANSTIME'].apply(self._calculate_time_period)

        # Add date column
        etc_data['date'] = etc_data['TRANSTIME'].apply(self._get_date_string)

        self.logger.info(f"Processed {len(etc_data)} ETC records")
        return etc_data

    def _parse_vehicle_plate(self, plate_str: str) -> str:
        """Parse vehicle plate (remove suffix like _0)"""
        if isinstance(plate_str, str) and "_" in plate_str:
            return plate_str.split("_")[0]
        return plate_str

    def _calculate_time_period(self, timestamp) -> int:
        """Calculate time period index within day based on time_window"""
        # Handle numpy.datetime64 type
        if hasattr(timestamp, 'dtype') and np.issubdtype(timestamp.dtype, np.datetime64):
            timestamp = pd.Timestamp(timestamp)

        # Convert to minutes of day
        minutes_of_day = timestamp.hour * 60 + timestamp.minute
        # Calculate time period
        return minutes_of_day // self.time_window

    def _get_date_string(self, timestamp) -> str:
        """Get date string (for grouping)"""
        # Handle numpy.datetime64 type
        if hasattr(timestamp, 'dtype') and np.issubdtype(timestamp.dtype, np.datetime64):
            timestamp = pd.Timestamp(timestamp)

        return timestamp.strftime("%Y-%m-%d")

    def _load_traffic_flow_data(self) -> Dict:
        """Load historical traffic flow data"""
        if not self.traffic_flow_dir:
            raise ValueError("Historical traffic flow data directory not provided")

        self.logger.info(f"Loading historical traffic flow data: {self.traffic_flow_dir}")
        traffic_flow_data = {}

        for gantry_id in self.all_gantry_ids:
            flow_file = os.path.join(self.traffic_flow_dir, f"trafficflow_{gantry_id}.csv")
            if os.path.exists(flow_file):
                flow_data = pd.read_csv(flow_file)
                # Convert time format
                flow_data['time'] = pd.to_datetime(flow_data['Time'])
                traffic_flow_data[gantry_id] = flow_data

        self.logger.info(f"Loaded historical traffic flow data for {len(traffic_flow_data)} gantries")
        return traffic_flow_data

    def _load_prediction_flow_data(self) -> Dict:
        """Load predicted traffic flow data"""
        if not self.prediction_flow_dir:
            self.logger.info("Prediction traffic flow directory not provided, will use only historical data")
            return {}

        self.logger.info(f"Loading prediction traffic flow data: {self.prediction_flow_dir}")
        prediction_flow_data = {}

        for gantry_id in self.all_gantry_ids:
            prediction_file = os.path.join(self.prediction_flow_dir, f"prediction_{gantry_id}.csv")
            if os.path.exists(prediction_file):
                prediction_data = pd.read_csv(prediction_file)
                # Convert time format
                prediction_data['time'] = pd.to_datetime(prediction_data['time'])
                prediction_data['pred_time'] = pd.to_datetime(prediction_data['pred_time'])
                prediction_flow_data[gantry_id] = prediction_data

        self.logger.info(f"Loaded prediction traffic flow data for {len(prediction_flow_data)} gantries")
        return prediction_flow_data

    def _calculate_travel_times(self) -> Dict:
        """Calculate vehicle travel times (minutes) for all segments"""
        self.logger.info("Calculating vehicle travel times...")
        travel_times = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        # Process by date
        for date_str, date_group in tqdm(self.etc_data.groupby('date'), desc="Processing dates"):
            # Calculate for each segment
            for segment_id, segment in self.scene_units.items():
                up_node = segment['up_node']
                down_node = segment['down_node']

                # Filter upstream and downstream data
                up_data = date_group[date_group['GANTRYID'] == up_node]
                down_data = date_group[date_group['GANTRYID'] == down_node]

                if up_data.empty or down_data.empty:
                    continue

                # Calculate by time period and vehicle type
                for time_period in range(int(24 * 60 / self.time_window)):  # Number of time periods in a day
                    # Convert to hour
                    hour = (time_period * self.time_window) // 60

                    # Determine time period characteristics
                    is_peak = (7 <= hour <= 9) or (17 <= hour <= 19)
                    is_weekend = pd.to_datetime(date_str).weekday() >= 5

                    for veh_type in VehicleType:
                        vehicle_type = veh_type.value

                        # Filter by time period and vehicle type
                        up_period_data = up_data[(up_data['time_period'] == time_period) &
                                                 (up_data['VEHICLETYPE'] == vehicle_type)]
                        down_period_data = down_data[(down_data['time_period'] == time_period) &
                                                     (down_data['VEHICLETYPE'] == vehicle_type)]

                        # Find vehicles that passed both upstream and downstream gantries
                        common_vehicles = set(up_period_data['VEHICLEPLATE']).intersection(
                            set(down_period_data['VEHICLEPLATE']))

                        if common_vehicles:
                            # Calculate travel times
                            travel_time_values = []
                            for vehicle in common_vehicles:
                                up_time = up_period_data[up_period_data['VEHICLEPLATE'] == vehicle]['TRANSTIME'].min()
                                down_time = down_period_data[down_period_data['VEHICLEPLATE'] == vehicle][
                                    'TRANSTIME'].min()

                                if down_time > up_time:
                                    # Calculate travel time (minutes)
                                    travel_time = (down_time - up_time).total_seconds() / 60

                                    # Filter outliers
                                    speed = segment['length'] / (travel_time / 60)  # km/h
                                    speed_limit = segment.get('speed_limit', 150)
                                    if 10 <= speed <= speed_limit * 1.2:
                                        travel_time_values.append(travel_time)

                            if travel_time_values:
                                # Special processing based on scene type
                                scene_type = segment['type']

                                # Adjust travel time for ramp scenarios
                                if scene_type in [SceneType.SCENE2, SceneType.SCENE3, SceneType.SCENE4]:
                                    # Adjust based on complexity
                                    complexity = segment.get('complexity_factor', 0.0)

                                    # Adjust based on vehicle size
                                    if vehicle_type in ['B3', 'T2', 'T3']:  # Larger vehicles
                                        size_factor = 1.0 + 0.1 * self.vehicle_factor  # Larger vehicles slightly slower
                                    else:
                                        size_factor = 1.0

                                    # Adjust based on peak/off-peak
                                    if is_peak and not is_weekend:
                                        peak_factor = 1.0 + 0.2 * self.time_factor  # Weekday peak is slower
                                    elif is_peak and is_weekend:
                                        peak_factor = 1.0 + 0.1 * self.time_factor  # Weekend peak is slightly slower
                                    else:
                                        peak_factor = 1.0

                                    # Apply adjustment factors
                                    adjustment_factor = 1.0 + (complexity * 0.2 * size_factor * peak_factor)
                                else:
                                    adjustment_factor = 1.0

                                # Store average travel time
                                key = f"{date_str}_{time_period}"
                                avg_travel_time = sum(travel_time_values) / len(travel_time_values)

                                # Apply ramp position and scene adjustments
                                avg_travel_time *= adjustment_factor

                                # Add random noise to simulate real conditions
                                if self.add_noise:
                                    noise = np.random.normal(0, self.noise_level * avg_travel_time)
                                    avg_travel_time = max(0.1, avg_travel_time + noise)  # Ensure positive travel time

                                travel_times[key][segment_id][vehicle_type] = avg_travel_time

        # Save travel_times to CSV file
        _save_travel_times_to_csv(travel_times, self.parameter_dir)

        return travel_times

    def _calculate_dynamic_diversion_coefficients(self) -> Dict:
        """Calculate dynamic diversion coefficients considering ramp positions"""
        self.logger.info("Calculating dynamic ramp diversion coefficients...")
        diversion_coefficients = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        # Process by date
        for date_str, date_group in tqdm(self.etc_data.groupby('date'), desc="Processing dates"):
            # Calculate for each segment
            for segment_id, segment in self.scene_units.items():
                scene_type = segment['type']

                # Only process segments with ramps
                if scene_type not in [SceneType.SCENE2, SceneType.SCENE3, SceneType.SCENE4]:
                    continue

                up_node = segment['up_node']
                down_node = segment['down_node']

                # Filter upstream and downstream data
                up_data = date_group[date_group['GANTRYID'] == up_node]
                down_data = date_group[date_group['GANTRYID'] == down_node]

                if up_data.empty or down_data.empty:
                    continue

                # Calculate by time period and vehicle type
                for time_period in range(int(24 * 60 / self.time_window)):
                    # Convert to hour
                    hour = (time_period * self.time_window) // 60

                    # Determine time period characteristics
                    is_peak = (7 <= hour <= 9) or (17 <= hour <= 19)
                    is_weekend = pd.to_datetime(date_str).weekday() >= 5

                    for veh_type in VehicleType:
                        vehicle_type = veh_type.value

                        # Filter by time period and vehicle type
                        up_period_data = up_data[(up_data['time_period'] == time_period) &
                                                 (up_data['VEHICLETYPE'] == vehicle_type)]
                        down_period_data = down_data[(down_data['time_period'] == time_period) &
                                                     (down_data['VEHICLETYPE'] == vehicle_type)]

                        # Get upstream and downstream vehicle sets
                        up_vehicles = set(up_period_data['VEHICLEPLATE'])
                        down_vehicles = set(down_period_data['VEHICLEPLATE'])

                        # Initialize diversion coefficients
                        on_ramp_coef = 0.0
                        off_ramp_coef = 0.0

                        # Calculate diversion coefficients based on scene type
                        if scene_type == SceneType.SCENE2:  # Upstream with entry ramp
                            # Get ramp position
                            ramp_position = segment.get('ramp_position', 0.2)

                            total_downstream = len(down_vehicles)
                            if total_downstream > 0:
                                new_vehicles = len(down_vehicles - up_vehicles)  # Vehicles entering from ramp

                                # Limit max coefficient to avoid extreme cases
                                max_new_ratio = 1  # Set based on actual conditions

                                base_on_ramp_coef = min(new_vehicles / total_downstream, max_new_ratio)

                                # Simplified ramp position adjustment logic
                                position_factor = 1.0 - (ramp_position - 0.1) * self.position_weight
                                position_factor = max(0.5, min(1.2, position_factor))  # Limit to reasonable range

                                # Apply position factor and time characteristics
                                if is_peak and not is_weekend:
                                    # Weekday peak has higher ramp usage
                                    time_adjust = 1.0 + 0.1 * self.time_factor
                                elif is_peak and is_weekend:
                                    # Weekend peak has medium ramp usage
                                    time_adjust = 1.0 + 0.05 * self.time_factor
                                else:
                                    # Off-peak has normal ramp usage
                                    time_adjust = 1.0

                                on_ramp_coef = base_on_ramp_coef * position_factor * time_adjust

                                # Vehicle type adjustment
                                if vehicle_type in ['B1', 'B2']:  # Small passenger vehicles
                                    veh_adjust = 1.0 + 0.05 * self.vehicle_factor
                                elif vehicle_type in ['T2', 'T3']:  # Large trucks
                                    veh_adjust = 1.0 - 0.05 * self.vehicle_factor
                                else:
                                    veh_adjust = 1.0

                                on_ramp_coef *= veh_adjust

                        elif scene_type == SceneType.SCENE3:  # Upstream with exit ramp
                            # Get ramp position
                            ramp_position = segment.get('ramp_position', 0.8)

                            total_upstream = len(up_vehicles)
                            if total_upstream > 0:
                                lost_vehicles = len(up_vehicles - down_vehicles)  # Vehicles exiting via ramp

                                # Limit max coefficient
                                max_lost_ratio = 0.7  # Max 70% exit rate

                                base_off_ramp_coef = min(lost_vehicles / total_upstream, max_lost_ratio)

                                # Simplified position adjustment logic
                                position_factor = 1.0 + (0.9 - ramp_position) * self.position_weight
                                position_factor = max(0.5, min(1.2, position_factor))

                                # Apply position factor and time characteristics
                                if hour >= 7 and hour <= 9:  # Morning peak
                                    # Higher exit ramp usage in morning peak
                                    time_adjust = 1.0 + 0.15 * self.time_factor
                                elif hour >= 17 and hour <= 19:  # Evening peak
                                    # Even higher exit ramp usage in evening peak
                                    time_adjust = 1.0 + 0.2 * self.time_factor
                                else:
                                    # Normal exit ramp usage in off-peak
                                    time_adjust = 1.0

                                off_ramp_coef = base_off_ramp_coef * position_factor * time_adjust

                                # Weekend adjustment
                                if is_weekend:
                                    if 10 <= hour <= 16:  # Weekend shopping/leisure hours
                                        weekend_adjust = 1.0 + 0.1 * self.time_factor
                                    else:
                                        weekend_adjust = 1.0 - 0.05 * self.time_factor
                                    off_ramp_coef *= weekend_adjust

                                # Vehicle type adjustment
                                if vehicle_type in ['B1', 'B2']:  # Small passenger vehicles
                                    veh_adjust = 1.0 + 0.05 * self.vehicle_factor
                                elif vehicle_type in ['T2', 'T3']:  # Large trucks
                                    veh_adjust = 1.0 - 0.05 * self.vehicle_factor
                                else:
                                    veh_adjust = 1.0

                                off_ramp_coef *= veh_adjust

                        elif scene_type == SceneType.SCENE4:  # Upstream with both entry and exit ramps
                            # Get ramp positions
                            entry_position = segment.get('entry_ramp_position', 0.2)
                            exit_position = segment.get('exit_ramp_position', 0.8)

                            # Calculate exit ramp coefficient
                            total_upstream = len(up_vehicles)
                            if total_upstream > 0:
                                lost_vehicles = len(up_vehicles - down_vehicles)
                                # Limit max exit rate
                                max_lost_ratio = 0.7
                                base_off_ramp_coef = min(lost_vehicles / total_upstream, max_lost_ratio)

                                # Apply position and time factors
                                exit_position_factor = 1.0 + (0.9 - exit_position) * self.position_weight
                                exit_position_factor = max(0.5, min(1.2, exit_position_factor))

                                if is_peak:
                                    time_adjust = 1.0 + 0.15 * self.time_factor
                                else:
                                    time_adjust = 1.0

                                off_ramp_coef = base_off_ramp_coef * exit_position_factor * time_adjust

                                # Vehicle type adjustment
                                if vehicle_type in ['B1', 'B2']:
                                    veh_adjust = 1.0 + 0.05 * self.vehicle_factor
                                elif vehicle_type in ['T2', 'T3']:
                                    veh_adjust = 1.0 - 0.05 * self.vehicle_factor
                                else:
                                    veh_adjust = 1.0

                                off_ramp_coef *= veh_adjust

                            # Calculate entry ramp coefficient
                            new_vehicles = len(down_vehicles - up_vehicles)
                            remaining_vehicles = len(up_vehicles.intersection(down_vehicles))
                            total_with_new = remaining_vehicles + new_vehicles

                            if total_with_new > 0:
                                # Limit max entry ratio
                                max_new_ratio = 0.6
                                base_on_ramp_coef = min(new_vehicles / total_with_new, max_new_ratio)

                                # Simplified position impact calculation
                                entry_position_factor = 1.0 - (entry_position - 0.1) * self.position_weight
                                entry_position_factor = max(0.5, min(1.2, entry_position_factor))

                                # Time period adjustment
                                if is_peak:
                                    time_adjust = 1.0 + 0.1 * self.time_factor
                                else:
                                    time_adjust = 1.0

                                on_ramp_coef = base_on_ramp_coef * entry_position_factor * time_adjust

                                # Vehicle type adjustment
                                if vehicle_type in ['B1', 'B2']:
                                    veh_adjust = 1.0 + 0.05 * self.vehicle_factor
                                elif vehicle_type in ['T2', 'T3']:
                                    veh_adjust = 1.0 - 0.05 * self.vehicle_factor
                                else:
                                    veh_adjust = 1.0

                                on_ramp_coef *= veh_adjust

                                # Consider distance between ramps
                                ramp_distance = exit_position - entry_position
                                if ramp_distance < 0.3:  # Small distance between ramps
                                    # Short weaving section, higher interaction
                                    interaction_factor = 0.8
                                    on_ramp_coef *= interaction_factor
                                    off_ramp_coef *= interaction_factor

                        # Apply damping factor to control coefficients
                        on_ramp_coef *= self.ramp_flow_damping
                        off_ramp_coef *= self.ramp_flow_damping

                        # Add random noise to simulate real conditions
                        if self.add_noise:
                            on_ramp_coef += np.random.normal(0, self.noise_level * max(0.01, on_ramp_coef))
                            off_ramp_coef += np.random.normal(0, self.noise_level * max(0.01, off_ramp_coef))

                        # Ensure coefficients are within valid range
                        on_ramp_coef = max(0, min(0.8, on_ramp_coef))  # Stricter upper limit
                        off_ramp_coef = max(0, min(0.8, off_ramp_coef))  # Stricter upper limit

                        # Store results
                        key = f"{date_str}_{time_period}"
                        diversion_coefficients[key][segment_id][vehicle_type] = {
                            'on_ramp': on_ramp_coef,
                            'off_ramp': off_ramp_coef
                        }

        # Save diversion_coefficients to CSV file
        _save_diversion_coefficients_to_csv(diversion_coefficients, self.parameter_dir)

        return diversion_coefficients

    def _calculate_traffic_states(self) -> Dict:
        """Calculate traffic states for each segment and time period"""
        self.logger.info("Calculating traffic states...")
        traffic_states = defaultdict(lambda: defaultdict(str))

        # Process all traffic flow data
        for gantry_id, flow_data in self.traffic_flow_data.items():
            # Find upstream segments
            upstream_segments = [seg_id for seg_id, seg in self.scene_units.items()
                                 if seg['down_node'] == gantry_id]

            if not upstream_segments:
                continue

            # Calculate traffic state for each time point
            for _, row in flow_data.iterrows():
                time_point = row['time']
                date_str = self._get_date_string(time_point)
                time_period = self._calculate_time_period(time_point)
                key = f"{date_str}_{time_period}"

                # Calculate total flow
                total_flow = 0
                for vt in VehicleType:
                    vehicle_type = vt.value
                    if vehicle_type in row:
                        total_flow += row[vehicle_type]

                # Determine traffic state for each upstream segment
                for segment_id in upstream_segments:
                    segment = self.scene_units[segment_id]

                    # Calculate theoretical capacity
                    speed_limit = segment.get('speed_limit', 100)
                    lanes = segment.get('lanes', 3)  # Use actual segment lanes
                    lane_capacity = 2000 if speed_limit >= 100 else 1800
                    total_capacity = lane_capacity * lanes

                    # Calculate flow/capacity ratio
                    hour = time_point.hour
                    is_weekend = time_point.weekday() >= 5
                    hourly_flow = total_flow * (60 / self.time_window)
                    v_c_ratio = hourly_flow / total_capacity

                    # Apply threshold adjustment factor
                    threshold_factor = self.state_threshold_factor

                    # Determine traffic state based on flow/capacity ratio
                    if 7 <= hour <= 9 or 17 <= hour <= 19:  # Peak hours
                        if not is_weekend:  # Weekday peak
                            if v_c_ratio > 0.75 * threshold_factor:
                                state = 'congested'
                            elif v_c_ratio > 0.6 * threshold_factor:
                                state = 'transition'
                            else:
                                state = 'free_flow'
                        else:  # Weekend peak
                            if v_c_ratio > 0.8 * threshold_factor:
                                state = 'congested'
                            elif v_c_ratio > 0.65 * threshold_factor:
                                state = 'transition'
                            else:
                                state = 'free_flow'
                    else:  # Off-peak
                        if v_c_ratio > 0.85 * threshold_factor:
                            state = 'congested'
                        elif v_c_ratio > 0.7 * threshold_factor:
                            state = 'transition'
                        else:
                            state = 'free_flow'

                    # Store traffic state
                    traffic_states[key][segment_id] = state

        return traffic_states

    def _get_historical_flow(self, gantry_id: str, time_point) -> Dict[str, float]:
        """Get historical flow for a gantry at a specific time point"""
        flow_data = self.traffic_flow_data.get(gantry_id)
        if flow_data is None:
            return {}

        # Convert time point to pandas Timestamp
        if isinstance(time_point, str):
            time_point = pd.to_datetime(time_point)

        # Find closest time point data
        closest_data = flow_data[flow_data['time'] == time_point]

        if closest_data.empty:
            return {}

        # Return flow values for each vehicle type
        flow_values = {}
        for veh_type in VehicleType:
            vehicle_type = veh_type.value
            if vehicle_type in closest_data.columns:
                flow_values[vehicle_type] = closest_data[vehicle_type].iloc[0]

        return flow_values

    def _get_predicted_flow(self, gantry_id: str, base_time, pred_time_horizon: int) -> Dict[str, float]:
        """
        Get predicted flow for a gantry at a future time point

        Parameters:
        gantry_id: Gantry ID
        base_time: Base time
        pred_time_horizon: Prediction time horizon (minutes)
        """
        if not self.prediction_flow_data:
            return {}  # Return empty dict if no prediction data available

        prediction_data = self.prediction_flow_data.get(gantry_id)
        if prediction_data is None:
            return {}

        # Convert time point to pandas Timestamp
        if isinstance(base_time, str):
            base_time = pd.to_datetime(base_time)

        # Find predictions for base time
        base_predictions = prediction_data[prediction_data['time'] == base_time]

        if base_predictions.empty:
            return {}

        # Find closest prediction horizon
        closest_prediction = base_predictions.iloc[
            (base_predictions['horizon'] - pred_time_horizon).abs().argsort()[:1]]

        if closest_prediction.empty:
            return {}

        # Return predicted flow for each vehicle type
        pred_flow_values = {}
        for veh_type in VehicleType:
            vehicle_type = veh_type.value
            if vehicle_type in closest_prediction.columns:
                pred_flow_values[vehicle_type] = closest_prediction[vehicle_type].iloc[0]

        return pred_flow_values

    def _get_position_impact_factor(self, segment, time_point, traffic_state):
        """Calculate ramp position impact factor"""
        scene_type = segment['type']

        # For segments without ramps, return 1.0
        if scene_type not in [SceneType.SCENE2, SceneType.SCENE3, SceneType.SCENE4]:
            return 1.0

        # Extract ramp position information
        if scene_type == SceneType.SCENE2:  # Entry ramp
            ramp_position = segment.get('ramp_position', 0.2)  # Default at 20% of segment

            # Simplified linear model
            if traffic_state == 'free_flow':
                # Free flow: ramp position has less impact, linear increase
                return 0.9 + 0.2 * ramp_position
            elif traffic_state == 'congested':
                # Congested: ramp position has more impact, linear increase
                return 0.7 + 0.6 * ramp_position
            else:  # Transition state
                # Linear interpolation of middle state
                return 0.8 + 0.4 * ramp_position

        elif scene_type == SceneType.SCENE3:  # Exit ramp
            ramp_position = segment.get('ramp_position', 0.8)  # Default at 80% of segment

            # Simplified linear model
            if traffic_state == 'free_flow':
                # Free flow: ramp position has less impact, linear decrease
                return 1.0 - 0.1 * (1 - ramp_position)
            elif traffic_state == 'congested':
                # Congested: ramp position has more impact, linear decrease
                return 1.0 - 0.3 * (1 - ramp_position)
            else:  # Transition state
                # Linear interpolation of middle state
                return 1.0 - 0.2 * (1 - ramp_position)

        elif scene_type == SceneType.SCENE4:  # Entry and exit ramps
            entry_position = segment.get('entry_ramp_position', 0.2)
            exit_position = segment.get('exit_ramp_position', 0.8)

            # Simplified weaving section impact
            if traffic_state == 'free_flow':
                # Free flow state
                entry_factor = 0.9 + 0.2 * entry_position
                exit_factor = 1.0 - 0.1 * (1 - exit_position)
                return (entry_factor + exit_factor) / 2  # Average impact
            elif traffic_state == 'congested':
                # Congested state, consider weaving length
                weaving_length = exit_position - entry_position
                weaving_factor = 0.5 + 0.5 * min(1.0, weaving_length / 0.4)  # Normalize weaving length
                return weaving_factor
            else:  # Transition state
                # Linear interpolation of middle state
                entry_factor = 0.8 + 0.3 * entry_position
                exit_factor = 1.0 - 0.2 * (1 - exit_position)
                weaving_length = exit_position - entry_position
                weaving_factor = 0.7 + 0.3 * min(1.0, weaving_length / 0.4)
                return (entry_factor + exit_factor) * weaving_factor / 2

        return 1.0  # Default return 1.0 (no impact)

    def _estimate_flow_with_position(self, upstream_gantry: str, time_point,
                                     segment_id: int, demand_time: int) -> Dict[str, float]:
        """
        Flow estimation formula considering ramp position

        Parameters:
        upstream_gantry: Upstream gantry ID
        time_point: Base time point
        segment_id: Segment ID
        demand_time: Demand time (minutes)
        """
        # Get scene type and segment info
        segment = self.scene_units[segment_id]
        scene_type = segment['type']
        length = segment['length']
        downstream_gantry = segment['down_node']

        # Convert time point to key format
        time_point_ts = pd.Timestamp(time_point) if not isinstance(time_point, pd.Timestamp) else time_point
        date_str = self._get_date_string(time_point_ts)
        time_period = self._calculate_time_period(time_point_ts)
        key = f"{date_str}_{time_period}"

        # Get traffic state
        traffic_state = self.traffic_states.get(key, {}).get(segment_id, 'free_flow')

        # Estimate flow (for downstream gantry)
        estimated_flow = {}

        for veh_type in VehicleType:
            vehicle_type = veh_type.value

            # Get travel time
            travel_time = self.travel_times.get(key, {}).get(segment_id, {}).get(vehicle_type, None)

            if travel_time is None:
                # If no record, use default travel time
                speed_limit = segment.get('speed_limit', 60)  # km/h
                travel_time = (length / speed_limit) * 60  # minutes

            # Get diversion coefficients
            diversion_coef = self.diversion_coefficients.get(key, {}).get(segment_id, {}).get(vehicle_type, {})
            on_ramp_coef = diversion_coef.get('on_ramp', 0.0)
            off_ramp_coef = diversion_coef.get('off_ramp', 0.0)

            # Get ramp position impact factor
            position_factor = self._get_position_impact_factor(segment, time_point, traffic_state)

            # Decide whether to use historical or prediction data
            if demand_time <= travel_time:
                # Demand time <= travel time, use historical data
                upstream_flow = self._get_historical_flow(upstream_gantry, time_point).get(vehicle_type, 0)

                if scene_type == SceneType.SCENE1 or scene_type == SceneType.SCENE5:
                    # Case 1/5: No ramps
                    flow_ratio = demand_time / travel_time
                    estimated_flow[vehicle_type] = upstream_flow * flow_ratio

                elif scene_type == SceneType.SCENE2:
                    # Case 2: Upstream with entry ramp
                    # Get ramp position
                    ramp_position = segment.get('ramp_position', 0.2)

                    # Calculate relationship between demand time and ramp position
                    ramp_time = travel_time * ramp_position

                    if demand_time <= ramp_time:
                        # Demand time hasn't reached ramp position, flow unaffected by ramp
                        flow_ratio = demand_time / travel_time
                        estimated_flow[vehicle_type] = upstream_flow * flow_ratio
                    else:
                        # Demand time has passed ramp position, consider ramp flow
                        # Pre-ramp flow ratio
                        pre_ramp_ratio = ramp_time / travel_time
                        # Post-ramp flow ratio
                        post_ramp_ratio = (demand_time - ramp_time) / travel_time

                        # Pre-ramp flow
                        pre_ramp_flow = upstream_flow * pre_ramp_ratio

                        # Post-ramp flow, considering vehicles joining from ramp
                        adjusted_on_coef = on_ramp_coef * position_factor
                        if adjusted_on_coef < 0.5:  # Use stricter threshold
                            # Use more robust calculation formula
                            additional_flow_ratio = adjusted_on_coef / (1 - adjusted_on_coef)
                            additional_flow_ratio = min(additional_flow_ratio, 1.0)  # Limit max value
                            additional_flow = upstream_flow * post_ramp_ratio * additional_flow_ratio
                        else:
                            # Use safe upper limit in extreme cases
                            additional_flow = upstream_flow * post_ramp_ratio * 1.0

                        # Total flow = pre-ramp flow + post-ramp mainline flow + ramp flow
                        post_ramp_main_flow = upstream_flow * post_ramp_ratio
                        ramp_flow = additional_flow

                        estimated_flow[vehicle_type] = pre_ramp_flow + post_ramp_main_flow + ramp_flow

                elif scene_type == SceneType.SCENE3:
                    # Case 3: Upstream with exit ramp
                    # Get ramp position
                    ramp_position = segment.get('ramp_position', 0.8)

                    # Calculate relationship between demand time and ramp position
                    ramp_time = travel_time * ramp_position

                    if demand_time <= ramp_time:
                        # Demand time hasn't reached ramp position, flow unaffected by ramp
                        flow_ratio = demand_time / travel_time
                        estimated_flow[vehicle_type] = upstream_flow * flow_ratio
                    else:
                        # Demand time has passed ramp position, consider ramp diversion
                        # Pre-ramp flow ratio
                        pre_ramp_ratio = ramp_time / travel_time
                        # Post-ramp flow ratio
                        post_ramp_ratio = (demand_time - ramp_time) / travel_time

                        # Pre-ramp flow
                        pre_ramp_flow = upstream_flow * pre_ramp_ratio

                        # Post-ramp flow, considering ramp diversion
                        adjusted_off_coef = off_ramp_coef * position_factor
                        adjusted_off_coef = min(adjusted_off_coef, 0.9)  # Limit max diversion coefficient

                        remaining_ratio = 1.0 - adjusted_off_coef
                        post_ramp_flow = upstream_flow * post_ramp_ratio * remaining_ratio

                        # Total flow = pre-ramp flow + post-ramp flow
                        estimated_flow[vehicle_type] = pre_ramp_flow + post_ramp_flow

                elif scene_type == SceneType.SCENE4:
                    # Case 4: Upstream with entry and exit ramps
                    # Get ramp positions
                    entry_position = segment.get('entry_ramp_position', 0.2)
                    exit_position = segment.get('exit_ramp_position', 0.8)

                    # Calculate relationship between demand time and ramp positions
                    entry_time = travel_time * entry_position
                    exit_time = travel_time * exit_position

                    if demand_time <= entry_time:
                        # Demand time hasn't reached entry ramp, flow unaffected by ramps
                        flow_ratio = demand_time / travel_time
                        estimated_flow[vehicle_type] = upstream_flow * flow_ratio

                    elif demand_time <= exit_time:
                        # Demand time has passed entry ramp but not exit ramp
                        # Entry ramp pre-flow ratio
                        pre_entry_ratio = entry_time / travel_time
                        # Entry ramp to demand time flow ratio
                        post_entry_ratio = (demand_time - entry_time) / travel_time

                        # Entry ramp pre-flow
                        pre_entry_flow = upstream_flow * pre_entry_ratio

                        # Entry ramp post-flow, considering vehicles joining from ramp
                        adjusted_on_coef = on_ramp_coef * position_factor

                        if adjusted_on_coef < 0.5:
                            # Use more robust calculation formula
                            additional_flow_ratio = adjusted_on_coef / (1 - adjusted_on_coef)
                            additional_flow_ratio = min(additional_flow_ratio, 1.0)
                            additional_flow = upstream_flow * post_entry_ratio * additional_flow_ratio
                        else:
                            # Use safe upper limit in extreme cases
                            additional_flow = upstream_flow * post_entry_ratio * 1.0

                        post_entry_main_flow = upstream_flow * post_entry_ratio
                        entry_ramp_flow = additional_flow

                        # Total flow = entry ramp pre-flow + entry ramp post mainline flow + entry ramp flow
                        estimated_flow[vehicle_type] = pre_entry_flow + post_entry_main_flow + entry_ramp_flow

                    else:
                        # Demand time has passed exit ramp
                        # Flow ratios for each section
                        pre_entry_ratio = entry_time / travel_time
                        mid_section_ratio = (exit_time - entry_time) / travel_time
                        post_exit_ratio = (demand_time - exit_time) / travel_time

                        # Entry ramp pre-flow
                        pre_entry_flow = upstream_flow * pre_entry_ratio

                        # Flow between entry and exit ramps
                        adjusted_on_coef = on_ramp_coef * position_factor

                        if adjusted_on_coef < 0.5:
                            additional_flow_ratio = adjusted_on_coef / (1 - adjusted_on_coef)
                            additional_flow_ratio = min(additional_flow_ratio, 1.0)
                            additional_flow = upstream_flow * mid_section_ratio * additional_flow_ratio
                        else:
                            additional_flow = upstream_flow * mid_section_ratio * 1.0

                        mid_main_flow = upstream_flow * mid_section_ratio
                        entry_ramp_flow = additional_flow

                        mid_total_flow = mid_main_flow + entry_ramp_flow

                        # Post-exit ramp flow
                        adjusted_off_coef = off_ramp_coef * position_factor
                        adjusted_off_coef = min(adjusted_off_coef, 0.9)

                        remaining_ratio = 1.0 - adjusted_off_coef
                        post_exit_flow = (upstream_flow + entry_ramp_flow) * post_exit_ratio * remaining_ratio

                        # Total flow = entry ramp pre-flow + weaving section flow + exit ramp post-flow
                        estimated_flow[vehicle_type] = pre_entry_flow + mid_total_flow * mid_section_ratio + post_exit_flow

            else:
                # Demand time > travel time, use prediction data
                # Calculate prediction horizon needed
                prediction_horizon = demand_time - travel_time

                # Get predicted flow
                predicted_flow = self._get_predicted_flow(
                    upstream_gantry, time_point, prediction_horizon).get(vehicle_type, 0)

                # Apply same ramp position logic to predicted flow
                if scene_type == SceneType.SCENE1 or scene_type == SceneType.SCENE5:
                    # Case 1/5: No ramps
                    estimated_flow[vehicle_type] = predicted_flow

                elif scene_type == SceneType.SCENE2:
                    # Case 2: Upstream with entry ramp
                    adjusted_on_coef = on_ramp_coef * position_factor

                    if adjusted_on_coef < 0.5:
                        additional_flow_ratio = adjusted_on_coef / (1 - adjusted_on_coef)
                        additional_flow_ratio = min(additional_flow_ratio, 1.0)
                        additional_flow = predicted_flow * additional_flow_ratio
                    else:
                        additional_flow = predicted_flow * 1.0

                    estimated_flow[vehicle_type] = predicted_flow + additional_flow

                elif scene_type == SceneType.SCENE3:
                    # Case 3: Upstream with exit ramp
                    adjusted_off_coef = off_ramp_coef * position_factor
                    adjusted_off_coef = min(adjusted_off_coef, 0.9)

                    remaining_ratio = 1.0 - adjusted_off_coef
                    estimated_flow[vehicle_type] = predicted_flow * remaining_ratio

                elif scene_type == SceneType.SCENE4:
                    # Case 4: Upstream with entry and exit ramps
                    adjusted_on_coef = on_ramp_coef * position_factor
                    adjusted_off_coef = off_ramp_coef * position_factor

                    # Limit coefficients
                    adjusted_on_coef = min(adjusted_on_coef, 0.5)
                    adjusted_off_coef = min(adjusted_off_coef, 0.9)

                    # First consider entry ramp
                    additional_flow_ratio = adjusted_on_coef / (1 - adjusted_on_coef)
                    additional_flow_ratio = min(additional_flow_ratio, 1.0)
                    entry_flow = predicted_flow * additional_flow_ratio

                    total_flow = predicted_flow + entry_flow

                    # Then consider exit ramp
                    remaining_ratio = 1.0 - adjusted_off_coef
                    estimated_flow[vehicle_type] = total_flow * remaining_ratio

            # Ensure flow is non-negative
            if vehicle_type in estimated_flow:
                estimated_flow[vehicle_type] = max(0, estimated_flow[vehicle_type])

        return estimated_flow

    def _estimate_flow_using_formula(self, upstream_gantry: str, time_point,
                                     segment_id: int, demand_time: int) -> Dict[str, float]:
        """
        Standard flow estimation formula (without ramp position consideration)
        """
        # Get scene type and segment info
        segment = self.scene_units[segment_id]
        scene_type = segment['type']
        length = segment['length']
        downstream_gantry = segment['down_node']

        # Convert time point to key format
        time_point_ts = pd.Timestamp(time_point) if not isinstance(time_point, pd.Timestamp) else time_point
        date_str = self._get_date_string(time_point_ts)
        time_period = self._calculate_time_period(time_point_ts)
        key = f"{date_str}_{time_period}"

        # Estimate flow (for downstream gantry)
        estimated_flow = {}

        for veh_type in VehicleType:
            vehicle_type = veh_type.value

            # Get travel time
            travel_time = self.travel_times.get(key, {}).get(segment_id, {}).get(vehicle_type, None)

            if travel_time is None:
                # If no record, use default travel time
                speed_limit = segment.get('speed_limit', 60)  # km/h
                travel_time = (length / speed_limit) * 60  # minutes

            # Get diversion coefficients
            diversion_coef = self.diversion_coefficients.get(key, {}).get(segment_id, {}).get(vehicle_type, {})
            on_ramp_coef = diversion_coef.get('on_ramp', 0.0)
            off_ramp_coef = diversion_coef.get('off_ramp', 0.0)

            # Decide whether to use historical or prediction data
            if demand_time <= travel_time:
                # Demand time <= travel time, use historical data
                upstream_flow = self._get_historical_flow(upstream_gantry, time_point).get(vehicle_type, 0)

                if scene_type == SceneType.SCENE1 or scene_type == SceneType.SCENE5:
                    # Case 1/5: No ramps
                    flow_ratio = demand_time / travel_time
                    estimated_flow[vehicle_type] = upstream_flow * flow_ratio

                elif scene_type == SceneType.SCENE2:
                    # Case 2: Upstream with entry ramp
                    # Downstream gantry is after ramp, consider entry ramp flow
                    flow_ratio = demand_time / travel_time
                    mainline_flow = upstream_flow * flow_ratio

                    if on_ramp_coef < 0.5:  # Use safer threshold
                        ramp_flow = mainline_flow * on_ramp_coef / (1 - on_ramp_coef)
                    else:
                        ramp_flow = mainline_flow * 1.0

                    estimated_flow[vehicle_type] = mainline_flow + ramp_flow

                elif scene_type == SceneType.SCENE3:
                    # Case 3: Upstream with exit ramp
                    # Downstream gantry is after ramp, consider exit diversion
                    flow_ratio = demand_time / travel_time
                    off_ramp_coef = min(off_ramp_coef, 0.9)  # Limit max diversion coefficient
                    estimated_flow[vehicle_type] = upstream_flow * (1 - off_ramp_coef) * flow_ratio

                elif scene_type == SceneType.SCENE4:
                    # Case 4: Upstream with entry and exit ramps
                    # Downstream gantry is after ramps, consider both entry and exit ramps
                    flow_ratio = demand_time / travel_time
                    # First calculate flow considering exit ramp
                    off_ramp_coef = min(off_ramp_coef, 0.9)
                    mainline_flow = upstream_flow * (1 - off_ramp_coef) * flow_ratio

                    # Then consider entry ramp
                    if on_ramp_coef < 0.5:
                        ramp_flow = mainline_flow * on_ramp_coef / (1 - on_ramp_coef)
                    else:
                        ramp_flow = mainline_flow * 1.0

                    estimated_flow[vehicle_type] = mainline_flow + ramp_flow

            else:
                # Demand time > travel time, use prediction data
                # Calculate prediction horizon needed
                prediction_horizon = demand_time - travel_time

                # Get predicted flow
                predicted_flow = self._get_predicted_flow(
                    upstream_gantry, time_point, prediction_horizon).get(vehicle_type, 0)

                if scene_type == SceneType.SCENE1 or scene_type == SceneType.SCENE5:
                    # Case 1/5: No ramps
                    estimated_flow[vehicle_type] = predicted_flow

                elif scene_type == SceneType.SCENE2:
                    # Case 2: Upstream with entry ramp
                    # Downstream gantry is after ramp, consider entry ramp flow
                    mainline_flow = predicted_flow

                    if on_ramp_coef < 0.5:
                        ramp_flow = mainline_flow * on_ramp_coef / (1 - on_ramp_coef)
                    else:
                        ramp_flow = mainline_flow * 1.0

                    estimated_flow[vehicle_type] = mainline_flow + ramp_flow

                elif scene_type == SceneType.SCENE3:
                    # Case 3: Upstream with exit ramp
                    # Downstream gantry is after ramp, consider exit diversion
                    off_ramp_coef = min(off_ramp_coef, 0.9)
                    estimated_flow[vehicle_type] = predicted_flow * (1 - off_ramp_coef)

                elif scene_type == SceneType.SCENE4:
                    # Case 4: Upstream with entry and exit ramps
                    # Downstream gantry is after ramps, consider both entry and exit ramps
                    # First calculate flow considering exit ramp
                    off_ramp_coef = min(off_ramp_coef, 0.9)
                    mainline_flow = predicted_flow * (1 - off_ramp_coef)

                    # Then consider entry ramp
                    if on_ramp_coef < 0.5:
                        ramp_flow = mainline_flow * on_ramp_coef / (1 - on_ramp_coef)
                    else:
                        ramp_flow = mainline_flow * 1.0

                    estimated_flow[vehicle_type] = mainline_flow + ramp_flow

            # Ensure flow is non-negative
            if vehicle_type in estimated_flow:
                estimated_flow[vehicle_type] = max(0, estimated_flow[vehicle_type])

        return estimated_flow

    def _find_consecutive_segments(self) -> List[Tuple[str, str, int]]:
        """
        Find all pairs of consecutive gantries, each pair representing an "upstream ETC → downstream blind segment" relationship to validate

        Returns:
        List of gantry pairs, each item as (upstream_gantry_id, downstream_gantry_id, segment_id)
        """
        consecutive_segments = []

        # Iterate through all segments
        for segment_id, segment in self.scene_units.items():
            up_node = segment['up_node']
            down_node = segment['down_node']

            # Check if both upstream and downstream gantries have data
            if up_node in self.traffic_flow_data and down_node in self.traffic_flow_data:
                consecutive_segments.append((up_node, down_node, segment_id))

        return consecutive_segments

    def validate_blind_segments(self):
        """Perform blind segment validation"""
        self.logger.info("Starting blind segment validation...")
        all_results = []

        # Find all consecutive gantry pairs
        consecutive_segments = self._find_consecutive_segments()
        self.logger.info(f"Found {len(consecutive_segments)} consecutive gantry pairs")

        if not consecutive_segments:
            self.logger.warning("No consecutive gantry pairs found, cannot perform validation")
            return pd.DataFrame()

        # Run validation for each demand time option
        for demand_time in self.demand_time_options:
            self.logger.info(f"\nValidating for demand time: {demand_time} minutes")

            # Validate each pair of consecutive gantries
            for upstream_gantry, downstream_gantry, segment_id in tqdm(consecutive_segments, desc="Validating segments"):
                # Get scene type
                scene_type = self.scene_units[segment_id]['type'].value
                relationship_id = f"{upstream_gantry}->{downstream_gantry}"

                # For ramp segments, use improved position-aware model
                if scene_type in [2, 3, 4]:  # Ramp segments
                    validation_method = "position_aware"
                    results = self._validate_segment_with_position(
                        upstream_gantry, downstream_gantry, segment_id, demand_time)
                else:  # Non-ramp segments
                    validation_method = "standard"
                    results = self._validate_standard_segment(
                        upstream_gantry, downstream_gantry, segment_id, demand_time)

                # Add validation method information and relationship ID
                for result in results:
                    result['validation_scheme'] = validation_method
                    result['demand_time'] = demand_time
                    result['relationship_id'] = relationship_id

                all_results.extend(results)

        # Convert to DataFrame
        validation_df = pd.DataFrame(all_results)

        # Save validation results
        if not validation_df.empty:
            validation_path = os.path.join(self.output_dir, "validation_results.csv")
            validation_df.to_csv(validation_path, index=False)
            self.logger.info(f"Validation results saved to: {validation_path}")

            # Calculate and save evaluation metrics
            self._calculate_and_save_metrics(validation_df)

        return validation_df

    def _validate_standard_segment(self, upstream_gantry, downstream_gantry, segment_id, demand_time):
        """Validate non-ramp segment using standard method"""
        validation_results = []

        # Get all time points for upstream gantry
        time_points = []
        if upstream_gantry in self.traffic_flow_data:
            time_points = self.traffic_flow_data[upstream_gantry]['time'].unique()

        if len(time_points) == 0:
            return []

        # Randomly select time points for validation (avoid too much data)
        if len(time_points) > 100:
            time_points = np.random.choice(time_points, 100, replace=False)

        # Validate for each time point
        for time_point in time_points:
            # Estimate downstream gantry flow using standard formula
            estimated_flow = self._estimate_flow_using_formula(
                upstream_gantry, time_point, segment_id, demand_time)

            # Get actual downstream gantry flow
            actual_flow = self._get_historical_flow(downstream_gantry, time_point)

            # Record validation results
            if estimated_flow and actual_flow:
                for veh_type in VehicleType:
                    vehicle_type = veh_type.value
                    if vehicle_type in estimated_flow and vehicle_type in actual_flow:
                        validation_results.append({
                            'segment_id': segment_id,
                            'upstream_gantry': upstream_gantry,
                            'downstream_gantry': downstream_gantry,
                            'time_point': time_point,
                            'vehicle_type': vehicle_type,
                            'estimated_flow': estimated_flow[vehicle_type],
                            'actual_flow': actual_flow[vehicle_type],
                            'scene_type': self.scene_units[segment_id]['type'].value
                        })

        return validation_results

    def _validate_segment_with_position(self, upstream_gantry, downstream_gantry, segment_id, demand_time):
        """Validate ramp segment using position-aware method"""
        validation_results = []

        # Get all time points for upstream gantry
        time_points = []
        if upstream_gantry in self.traffic_flow_data:
            time_points = self.traffic_flow_data[upstream_gantry]['time'].unique()

        if len(time_points) == 0:
            return []

        # Randomly select time points for validation (avoid too much data)
        if len(time_points) > 100:
            time_points = np.random.choice(time_points, 100, replace=False)

        # Validate for each time point
        for time_point in time_points:
            # Estimate downstream gantry flow using position-aware formula
            estimated_flow = self._estimate_flow_with_position(
                upstream_gantry, time_point, segment_id, demand_time)

            # Get actual downstream gantry flow
            actual_flow = self._get_historical_flow(downstream_gantry, time_point)

            # Record validation results
            if estimated_flow and actual_flow:
                for veh_type in VehicleType:
                    vehicle_type = veh_type.value
                    if vehicle_type in estimated_flow and vehicle_type in actual_flow:
                        validation_results.append({
                            'segment_id': segment_id,
                            'upstream_gantry': upstream_gantry,
                            'downstream_gantry': downstream_gantry,
                            'time_point': time_point,
                            'vehicle_type': vehicle_type,
                            'estimated_flow': estimated_flow[vehicle_type],
                            'actual_flow': actual_flow[vehicle_type],
                            'scene_type': self.scene_units[segment_id]['type'].value
                        })

        return validation_results

    def _calculate_metrics(self, validation_df: pd.DataFrame) -> Dict:
        """Calculate validation metrics"""
        metrics = {}

        # Ensure dataframe is not empty
        if validation_df.empty:
            self.logger.warning("Warning: No validation data, cannot calculate metrics")
            return metrics

        # Extract ground truth and predictions
        y_true = validation_df['actual_flow'].values
        y_pred = validation_df['estimated_flow'].values

        # Calculate MAE
        mae = np.mean(np.abs(y_pred - y_true))
        metrics['MAE'] = mae

        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        metrics['RMSE'] = rmse

        # Calculate R²
        metrics['R2'] = r2_score(y_true, y_pred)

        return metrics

    def _calculate_and_save_metrics(self, validation_df: pd.DataFrame):
        """Calculate and save various evaluation metrics"""
        # 1. Overall metrics
        overall_metrics = self._calculate_metrics(validation_df)
        self.logger.info("\nOverall validation metrics:")
        for metric, value in overall_metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")

        # 2. Metrics by validation scheme
        self.logger.info("\nMetrics by validation scheme:")
        for scheme in validation_df['validation_scheme'].unique():
            scheme_df = validation_df[validation_df['validation_scheme'] == scheme]
            scheme_metrics = self._calculate_metrics(scheme_df)
            self.logger.info(f"\n{scheme} scheme metrics:")
            for metric, value in scheme_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")

        # 3. Metrics by scene type
        self.logger.info("\nMetrics by scene type:")
        for scene_type in sorted(validation_df['scene_type'].unique()):
            scene_df = validation_df[validation_df['scene_type'] == scene_type]
            if len(scene_df) > 0:
                scene_metrics = self._calculate_metrics(scene_df)
                self.logger.info(f"\nScene type {scene_type} metrics:")
                for metric, value in scene_metrics.items():
                    self.logger.info(f"{metric}: {value:.4f}")

        # Save detailed metrics
        self._save_detailed_metrics(validation_df)

    def _save_detailed_metrics(self, validation_df):
        """Save detailed evaluation metrics to CSV files"""
        # Create metrics directory
        metrics_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # 1. Overall metrics
        overall_metrics = self._calculate_metrics(validation_df)
        pd.DataFrame([overall_metrics]).to_csv(
            os.path.join(metrics_dir, "overall_metrics.csv"), index=False)

        # 2. Metrics by relationship
        relationship_metrics = []
        for rel_id in validation_df['relationship_id'].unique():
            rel_df = validation_df[validation_df['relationship_id'] == rel_id]
            metrics = self._calculate_metrics(rel_df)
            relationship_metrics.append({
                'relationship_id': rel_id,
                'sample_count': len(rel_df),
                **metrics
            })
        pd.DataFrame(relationship_metrics).to_csv(
            os.path.join(metrics_dir, "relationship_metrics.csv"), index=False)

        # 3. Metrics by validation scheme
        scheme_metrics = []
        for scheme in validation_df['validation_scheme'].unique():
            scheme_df = validation_df[validation_df['validation_scheme'] == scheme]
            metrics = self._calculate_metrics(scheme_df)
            scheme_metrics.append({
                'validation_scheme': scheme,
                'sample_count': len(scheme_df),
                **metrics
            })
        pd.DataFrame(scheme_metrics).to_csv(
            os.path.join(metrics_dir, "validation_scheme_metrics.csv"), index=False)

        # 4. Metrics by demand time
        demand_metrics = []
        for demand_time in sorted(validation_df['demand_time'].unique()):
            demand_df = validation_df[validation_df['demand_time'] == demand_time]
            metrics = self._calculate_metrics(demand_df)
            demand_metrics.append({
                'demand_time': demand_time,
                'sample_count': len(demand_df),
                **metrics
            })
        pd.DataFrame(demand_metrics).to_csv(
            os.path.join(metrics_dir, "demand_time_metrics.csv"), index=False)

        # 5. Metrics by scene type
        scene_metrics = []
        for scene_type in sorted(validation_df['scene_type'].unique()):
            scene_df = validation_df[validation_df['scene_type'] == scene_type]
            if len(scene_df) > 0:
                metrics = self._calculate_metrics(scene_df)
                scene_metrics.append({
                    'scene_type': scene_type,
                    'sample_count': len(scene_df),
                    **metrics
                })
        pd.DataFrame(scene_metrics).to_csv(
            os.path.join(metrics_dir, "scene_type_metrics.csv"), index=False)

        # 6. Metrics by vehicle type
        vehicle_metrics = []
        for vehicle_type in [vt.value for vt in VehicleType]:
            veh_df = validation_df[validation_df['vehicle_type'] == vehicle_type]
            if not veh_df.empty:
                metrics = self._calculate_metrics(veh_df)
                vehicle_metrics.append({
                    'vehicle_type': vehicle_type,
                    'sample_count': len(veh_df),
                    **metrics
                })
        pd.DataFrame(vehicle_metrics).to_csv(
            os.path.join(metrics_dir, "vehicle_type_metrics.csv"), index=False)

        # 7. Metrics by scheme and vehicle type combination
        scheme_vehicle_metrics = []
        for scheme in validation_df['validation_scheme'].unique():
            for vehicle_type in [vt.value for vt in VehicleType]:
                combo_df = validation_df[(validation_df['validation_scheme'] == scheme) &
                                         (validation_df['vehicle_type'] == vehicle_type)]
                if not combo_df.empty:
                    metrics = self._calculate_metrics(combo_df)
                    scheme_vehicle_metrics.append({
                        'validation_scheme': scheme,
                        'vehicle_type': vehicle_type,
                        'sample_count': len(combo_df),
                        **metrics
                    })
        pd.DataFrame(scheme_vehicle_metrics).to_csv(
            os.path.join(metrics_dir, "scheme_vehicle_metrics.csv"), index=False)

        # 8. Metrics by scheme and scene type combination
        scheme_scene_metrics = []
        for scheme in validation_df['validation_scheme'].unique():
            for scene_type in sorted(validation_df['scene_type'].unique()):
                combo_df = validation_df[(validation_df['validation_scheme'] == scheme) &
                                         (validation_df['scene_type'] == scene_type)]
                if len(combo_df) > 0:
                    metrics = self._calculate_metrics(combo_df)
                    scheme_scene_metrics.append({
                        'validation_scheme': scheme,
                        'scene_type': scene_type,
                        'sample_count': len(combo_df),
                        **metrics
                    })
        pd.DataFrame(scheme_scene_metrics).to_csv(
            os.path.join(metrics_dir, "scheme_scene_metrics.csv"), index=False)

        # 9. Metrics by demand time and scene type combination
        demand_scene_metrics = []
        for demand_time in sorted(validation_df['demand_time'].unique()):
            for scene_type in sorted(validation_df['scene_type'].unique()):
                combo_df = validation_df[(validation_df['demand_time'] == demand_time) &
                                         (validation_df['scene_type'] == scene_type)]
                if len(combo_df) > 0:
                    metrics = self._calculate_metrics(combo_df)
                    demand_scene_metrics.append({
                        'demand_time': demand_time,
                        'scene_type': scene_type,
                        'sample_count': len(combo_df),
                        **metrics
                    })
        pd.DataFrame(demand_scene_metrics).to_csv(
            os.path.join(metrics_dir, "demand_scene_metrics.csv"), index=False)

    def run(self):
        """Run complete validation workflow"""
        start_time = time.time()
        self.logger.info("Starting blind segment validation...")

        # Perform validation
        validation_df = self.validate_blind_segments()

        end_time = time.time()
        self.logger.info(f"Blind segment validation completed! Time taken: {end_time - start_time:.2f} seconds")

        return validation_df