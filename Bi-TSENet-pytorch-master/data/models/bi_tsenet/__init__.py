# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Author : Code_charon
# @Time : 2024/11/15 20:40
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stgcn.gcn import GCNBlock, process_graph_matrices
from models.stgcn.tcn import TemporalConvNet, BiDirectionalTCN


class BiTSENet(nn.Module):
    def __init__(self, config):
        super(BiTSENet, self).__init__()
        self.config = config

        # Calculate feature multiplier based on aggregation method
        feature_multiplier = config.NUM_RELATIONS if config.RELATION_AGGREGATION == 'concat' else 1

        # GCN for spatial feature extraction
        self.gcn = GCNBlock(
            in_channels=1,  # Start with single feature per node (traffic flow)
            hidden_channels=config.GCN_HIDDEN_CHANNELS,
            out_channels=config.GCN_HIDDEN_CHANNELS[-1],
            num_relations=config.NUM_RELATIONS,
            dropout=config.GCN_DROPOUT,
            aggregation=config.RELATION_AGGREGATION
        )

        # Use NUM_NODES from config for proper TCN input sizing
        # Consider feature_multiplier for concat aggregation
        gcn_output_dim = config.GCN_HIDDEN_CHANNELS[-1] * feature_multiplier
        tcn_input_size = config.NUM_NODES * gcn_output_dim

        self.gcn_output_dim = gcn_output_dim  # Save for later use

        # TCN for temporal feature extraction
        if config.BIDIRECTIONAL:
            self.tcn = BiDirectionalTCN(
                num_inputs=tcn_input_size,
                num_channels=config.TCN_CHANNELS,
                kernel_size=config.TCN_KERNEL_SIZE,
                dropout=config.TCN_DROPOUT
            )
        else:
            self.tcn = TemporalConvNet(
                num_inputs=tcn_input_size,
                num_channels=config.TCN_CHANNELS,
                kernel_size=config.TCN_KERNEL_SIZE,
                dropout=config.TCN_DROPOUT
            )

        # 投影层 - 使用更高效的实现
        self.projection = nn.Sequential(
            nn.Linear(config.TCN_CHANNELS[-1], config.NUM_NODES * config.FINAL_FC_HIDDEN),
            nn.ReLU()  # 移除BatchNorm以提高速度
        )

        # Final prediction layer
        self.predictor = nn.Linear(config.FINAL_FC_HIDDEN, config.HORIZON)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """
        使用Xavier初始化权重
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, adj_matrices):
        """
        x: Traffic flow data [batch_size, sequence_length, num_nodes]
        adj_matrices: List of adjacency matrices for each relation type

        Returns:
        out: Predicted traffic flow [batch_size, num_nodes, horizon]
        """
        batch_size, seq_len, num_nodes = x.size()

        # Process each time step with GCN for spatial features
        spatial_features = []
        for t in range(seq_len):
            # [batch_size, num_nodes] -> [batch_size, num_nodes, 1]
            node_features = x[:, t, :].unsqueeze(-1)

            # Apply GCN to extract spatial features
            gcn_out = self.gcn(node_features, adj_matrices)
            spatial_features.append(gcn_out)

        # Stack spatial features for all time steps
        # [batch_size, seq_len, num_nodes, gcn_output_dim]
        spatial_features = torch.stack(spatial_features, dim=1)

        # Reshape for TCN: [batch_size, seq_len, num_nodes * gcn_output_dim]
        tcn_in = spatial_features.reshape(batch_size, seq_len, num_nodes * self.gcn_output_dim)

        # TCN sequence modeling
        temporal_features = self.tcn(tcn_in)

        # Use the last time step's features for prediction
        last_features = temporal_features[:, -1, :]  # [batch_size, tcn_output_size]

        # Project to get node features - 使用Sequential
        projected = self.projection(last_features)  # [batch_size, num_nodes * final_fc_hidden]
        node_features = projected.reshape(batch_size, num_nodes, self.config.FINAL_FC_HIDDEN)

        # Apply final prediction layer to each node
        predictions = self.predictor(node_features)  # [batch_size, num_nodes, horizon]

        return predictions