# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Author : Code_charon
# @Time : 2024/11/15 20:40
# ------------------------------------------------------------------------------

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from configs import Config
from models.stgcn.gcn import process_graph_matrices


class TrafficDataset(Dataset):
    def __init__(self, data, time_stamps, seq_length, horizon, device):
        self.data = data
        self.time_stamps = time_stamps
        self.seq_length = seq_length
        self.horizon = horizon
        self.device = device

    def __len__(self):
        return len(self.data) - self.seq_length - self.horizon + 1

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
        y = self.data[index + self.seq_length:index + self.seq_length + self.horizon]

        # Get the timestamp at the end of input sequence
        t = self.time_stamps[index + self.seq_length - 1]

        # Convert timestamp to string if it's a pandas Timestamp
        if hasattr(t, 'strftime'):
            t = t.strftime('%Y-%m-%d %H:%M:%S')

        return torch.FloatTensor(x).to(self.device), torch.FloatTensor(y).to(self.device), t


class TrafficDataProcessor:
    def __init__(self, config):
        self.config = config
        self.dataset_path = os.path.join(config.DATA_ROOT, config.CURRENT_DATASET)
        self.traffic_folder = os.path.join(self.dataset_path, 'traffic')
        self.scaler = None
        self.adj_matrices = None
        self.time_stamps = None
        self.vehicle_types = None

    def load_data(self):
        # Load graph data
        adj_path = os.path.join(self.dataset_path, f"{self.config.CURRENT_DATASET}_adj.csv")
        distance_path = os.path.join(self.dataset_path, f"{self.config.CURRENT_DATASET}_distance.csv")
        similarity_path = os.path.join(self.dataset_path, f"{self.config.CURRENT_DATASET}_similarity.csv")

        # Load graph matrices
        adj_matrix = pd.read_csv(adj_path, header=None).values
        distance_matrix = pd.read_csv(distance_path, header=None).values
        similarity_matrix = pd.read_csv(similarity_path, header=None).values

        # Convert to tensor
        adj_matrix = torch.FloatTensor(adj_matrix).to(self.config.DEVICE)
        distance_matrix = torch.FloatTensor(distance_matrix).to(self.config.DEVICE)
        similarity_matrix = torch.FloatTensor(similarity_matrix).to(self.config.DEVICE)

        # Process graph matrices
        self.adj_matrices = process_graph_matrices(
            adj_matrix,
            distance_matrix,
            similarity_matrix,
            self.config.DISTANCE_THRESHOLD,
            self.config.SIMILARITY_THRESHOLD
        )

        # Load traffic flow data from multiple files
        traffic_files = glob.glob(os.path.join(self.traffic_folder, 'trafficflow_G*.csv'))
        if not traffic_files:
            raise FileNotFoundError(f"No traffic flow files found in {self.traffic_folder}")

        # Sort files to ensure consistent order
        traffic_files.sort()

        all_data = []
        all_timestamps = []

        # Process each traffic file
        for file_path in traffic_files:
            df = pd.read_csv(file_path)

            # Convert Time column to datetime
            df['Time'] = pd.to_datetime(df['Time'], format=self.config.TIME_FORMAT)

            # Store time stamps as strings (to avoid collate issues)
            timestamps = df['Time'].dt.strftime(self.config.TIMESTAMP_FORMAT).tolist()
            all_timestamps.extend(timestamps)

            # Extract vehicle type columns
            if self.vehicle_types is None:
                # Identify vehicle type columns (B1, B2, B3, T1, T2, T3)
                self.vehicle_types = [col for col in df.columns if col in self.config.VEHICLE_TYPES]
                self.config.NUM_NODES = len(self.vehicle_types)
                print(f"Detected {self.config.NUM_NODES} vehicle types: {self.vehicle_types}")

            # Extract flow data for all vehicle types
            flow_data = df[self.vehicle_types].values
            all_data.append(flow_data)

        # Combine all data
        if len(all_data) > 1:
            traffic_flow = np.vstack(all_data)
            self.time_stamps = all_timestamps
        else:
            traffic_flow = all_data[0]
            self.time_stamps = all_timestamps

        print(f"Loaded traffic flow data with shape: {traffic_flow.shape}")

        return traffic_flow

    def normalize_data(self, data):
        if self.config.NORMALIZATION == 'min-max':
            # Min-max normalization to [0, 1]
            self.scaler = {'min': data.min(axis=0), 'max': data.max(axis=0)}
            denominator = self.scaler['max'] - self.scaler['min']
            denominator[denominator == 0] = 1  # Avoid division by zero
            normalized_data = (data - self.scaler['min']) / denominator
        else:  # z-score normalization
            self.scaler = {'mean': data.mean(axis=0), 'std': data.std(axis=0)}
            self.scaler['std'][self.scaler['std'] == 0] = 1  # Avoid division by zero
            normalized_data = (data - self.scaler['mean']) / self.scaler['std']

        return normalized_data

    def denormalize_data(self, normalized_data):
        """
        将归一化的数据转换回原始尺度
        """
        if self.config.NORMALIZATION == 'min-max':
            denominator = self.scaler['max'] - self.scaler['min']
            denominator[denominator == 0] = 1  # Avoid division by zero
            denormalized = normalized_data * denominator + self.scaler['min']
        else:  # z-score normalization
            denormalized = normalized_data * self.scaler['std'] + self.scaler['mean']

        return denormalized

    def generate_datasets(self):
        # Load and normalize data
        traffic_flow = self.load_data()
        normalized_data = self.normalize_data(traffic_flow)

        # Split data into train, validation, and test sets
        n_samples = len(normalized_data)
        train_end = int(n_samples * self.config.TRAIN_RATIO)
        val_end = train_end + int(n_samples * self.config.VAL_RATIO)

        train_data = normalized_data[:train_end]
        val_data = normalized_data[train_end:val_end]
        test_data = normalized_data[val_end:]

        # Split timestamps accordingly
        train_timestamps = self.time_stamps[:train_end]
        val_timestamps = self.time_stamps[train_end:val_end]
        test_timestamps = self.time_stamps[val_end:]

        # Create datasets
        train_dataset = TrafficDataset(train_data, train_timestamps, self.config.SEQUENCE_LENGTH,
                                       self.config.HORIZON, self.config.DEVICE)
        val_dataset = TrafficDataset(val_data, val_timestamps, self.config.SEQUENCE_LENGTH,
                                     self.config.HORIZON, self.config.DEVICE)
        test_dataset = TrafficDataset(test_data, test_timestamps, self.config.SEQUENCE_LENGTH,
                                      self.config.HORIZON, self.config.DEVICE)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)

        # Save raw test data for evaluation
        self.test_data_raw = traffic_flow[val_end:]
        self.test_timestamps = test_timestamps

        return train_loader, val_loader, test_loader, self.adj_matrices