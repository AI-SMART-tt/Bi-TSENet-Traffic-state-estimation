# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Author : Code_charon
# @Time : 2024/11/15 20:40
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiRelationGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations=3, aggregation='weighted_sum'):
        super(MultiRelationGCNLayer, self).__init__()
        self.num_relations = num_relations
        self.aggregation = aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear transformations for each relation type
        self.linear_layers = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(num_relations)
        ])

        # Relation weights for aggregation
        self.relation_weights = nn.Parameter(torch.ones(num_relations))

        # Attention mechanism for relation aggregation if needed
        if aggregation == 'attention':
            self.attention_layer = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.Tanh(),
                nn.Linear(out_channels, 1, bias=False)
            )

    def forward(self, x, adj_matrices):
        """
        x: Node features [batch_size, num_nodes, in_channels]
        adj_matrices: List of normalized adjacency matrices [num_nodes, num_nodes] for each relation
        """
        batch_size, num_nodes, _ = x.size()
        supports = []

        # Process each relation type
        for r in range(self.num_relations):
            # Apply linear transformation to each node's features
            transformed_x = self.linear_layers[r](x)  # [batch_size, num_nodes, out_channels]

            # Graph convolution: multiply with adjacency matrix
            # Expand adjacency matrix for batch processing
            adj = adj_matrices[r].unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nodes, num_nodes]

            # Apply graph convolution
            support = torch.bmm(adj, transformed_x)  # [batch_size, num_nodes, out_channels]
            supports.append(support)

        # Aggregate the multi-relation outputs
        if self.aggregation == 'concat':
            # Concatenate the outputs along the feature dimension
            output = torch.cat(supports, dim=-1)  # [batch_size, num_nodes, out_channels*num_relations]

        elif self.aggregation == 'attention':
            # Stack outputs for attention mechanism
            stacked = torch.stack(supports, dim=1)  # [batch_size, num_relations, num_nodes, out_channels]

            # Reshape for attention calculation
            flat_stacked = stacked.view(-1, self.out_channels)  # [batch_size*num_relations*num_nodes, out_channels]
            scores = self.attention_layer(flat_stacked)  # [batch_size*num_relations*num_nodes, 1]
            scores = scores.view(batch_size, self.num_relations, num_nodes)  # [batch_size, num_relations, num_nodes]

            # Apply softmax on relation dimension
            attention = F.softmax(scores, dim=1)  # [batch_size, num_relations, num_nodes]

            # Apply attention weights and sum
            attention = attention.unsqueeze(-1)  # [batch_size, num_relations, num_nodes, 1]
            weighted_sum = (stacked * attention).sum(dim=1)  # [batch_size, num_nodes, out_channels]
            output = weighted_sum

        else:  # Default: weighted_sum
            # Apply learnable weights to each relation output
            normalized_weights = F.softmax(self.relation_weights, dim=0)

            # Weighted sum of relation outputs
            output = 0
            for i, support in enumerate(supports):
                output = output + support * normalized_weights[i]

        return output


class GCNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=3,
                 dropout=0.3, aggregation='weighted_sum'):
        super(GCNBlock, self).__init__()

        # Stack of GCN layers
        self.gcn_layers = nn.ModuleList()
        current_channels = in_channels

        # Add hidden layers
        for hidden_dim in hidden_channels:
            self.gcn_layers.append(MultiRelationGCNLayer(
                current_channels, hidden_dim, num_relations, aggregation
            ))
            current_channels = hidden_dim if aggregation != 'concat' else hidden_dim * num_relations

        # Add output layer
        if aggregation == 'concat':
            final_in_channels = current_channels
        else:
            final_in_channels = current_channels

        self.gcn_layers.append(MultiRelationGCNLayer(
            final_in_channels, out_channels, num_relations, aggregation
        ))

        # ReLU and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_matrices):
        """
        x: Node features [batch_size, num_nodes, in_channels]
        adj_matrices: List of adjacency matrices for each relation
        """
        # Apply layers sequentially with activations and dropout
        for i, layer in enumerate(self.gcn_layers[:-1]):
            x = layer(x, adj_matrices)
            x = self.relu(x)
            x = self.dropout(x)

        # Apply final layer without activation
        x = self.gcn_layers[-1](x, adj_matrices)
        return x


def process_graph_matrices(adj_matrix, distance_matrix, similarity_matrix,
                           distance_threshold=10.0, similarity_threshold=0.1):
    """
    Process and normalize different graph matrices for multi-relation GCN

    Returns:
        List of processed adjacency matrices for each relation type
    """
    # Process adjacency matrix
    adj = adj_matrix.clone()
    adj = adj + torch.eye(adj.size(0), device=adj.device)  # Add self-loops
    degree = adj.sum(1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
    adj_normalized = degree_inv_sqrt.unsqueeze(-1) * adj * degree_inv_sqrt.unsqueeze(0)

    # Process distance matrix - inverse transform and thresholding
    dist = distance_matrix.clone()
    dist = 1.0 / (dist + 1e-6)  # Inverse transformation
    dist[dist < 1.0 / distance_threshold] = 0  # Thresholding
    degree_dist = dist.sum(1)
    degree_dist_inv_sqrt = torch.pow(degree_dist, -0.5)
    degree_dist_inv_sqrt[degree_dist_inv_sqrt == float('inf')] = 0
    dist_normalized = degree_dist_inv_sqrt.unsqueeze(-1) * dist * degree_dist_inv_sqrt.unsqueeze(0)

    # Process similarity matrix - thresholding
    sim = similarity_matrix.clone()
    sim[sim < similarity_threshold] = 0  # Thresholding
    degree_sim = sim.sum(1)
    degree_sim_inv_sqrt = torch.pow(degree_sim, -0.5)
    degree_sim_inv_sqrt[degree_sim_inv_sqrt == float('inf')] = 0
    sim_normalized = degree_sim_inv_sqrt.unsqueeze(-1) * sim * degree_sim_inv_sqrt.unsqueeze(0)

    return [adj_normalized, dist_normalized, sim_normalized]