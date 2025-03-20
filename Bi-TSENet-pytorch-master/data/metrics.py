# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Author : Code_charon
# @Time : 2024/11/15 20:40
# ------------------------------------------------------------------------------

import numpy as np
import torch


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_pred - y_true))


def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error"""
    return np.mean((y_pred - y_true) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r_squared(y_true, y_pred):
    """Calculate R-squared (coefficient of determination)"""
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    if ss_total == 0:
        return 0  # Avoid division by zero
    return 1 - (ss_residual / ss_total)


def calculate_metrics(y_true, y_pred):
    """Calculate all metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }


def calculate_node_metrics(y_true, y_pred):
    """Calculate metrics for each node and overall average"""
    n_nodes = y_true.shape[1]
    node_metrics = []

    # Calculate metrics for each node
    for i in range(n_nodes):
        node_true = y_true[:, i]
        node_pred = y_pred[:, i]
        metrics = calculate_metrics(node_true, node_pred)
        node_metrics.append(metrics)

    # Calculate overall metrics - using flattened arrays
    overall_metrics = calculate_metrics(y_true.flatten(), y_pred.flatten())

    return node_metrics, overall_metrics


def calculate_grouped_metrics(y_true, y_pred, b_indices=[0, 1, 2], t_indices=[3, 4, 5]):
    """Calculate metrics for groups of nodes and overall"""
    n_samples = y_true.shape[0]
    node_metrics = []

    # Calculate metrics for each individual node
    for i in range(y_true.shape[1]):
        node_true = y_true[:, i]
        node_pred = y_pred[:, i]
        metrics = calculate_metrics(node_true, node_pred)
        node_metrics.append(metrics)

    # Calculate B group metrics (B1+B2+B3)
    b_true = np.sum(y_true[:, b_indices], axis=1)
    b_pred = np.sum(y_pred[:, b_indices], axis=1)
    b_metrics = calculate_metrics(b_true, b_pred)

    # Calculate T group metrics (T1+T2+T3)
    t_true = np.sum(y_true[:, t_indices], axis=1)
    t_pred = np.sum(y_pred[:, t_indices], axis=1)
    t_metrics = calculate_metrics(t_true, t_pred)

    # Calculate overall metrics (all combined)
    overall_true = np.sum(y_true, axis=1)
    overall_pred = np.sum(y_pred, axis=1)
    overall_metrics = calculate_metrics(overall_true, overall_pred)

    return node_metrics, b_metrics, t_metrics, overall_metrics