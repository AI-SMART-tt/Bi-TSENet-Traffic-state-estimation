# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Author : Code_charon
# @Time : 2024/11/15 20:40
# ------------------------------------------------------------------------------

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models.bi_tsenet import BiTSENet
from preprocess import TrafficDataProcessor


def train_model(config, logger):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Initialize data processor
    data_processor = TrafficDataProcessor(config)

    # Load the data to set config.NUM_NODES
    train_loader, val_loader, _, adj_matrices = data_processor.generate_datasets()

    # Initialize the model after NUM_NODES is set
    model = BiTSENet(config).to(config.DEVICE)
    logger.info(
        f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=config.SCHEDULER_STEP, gamma=config.SCHEDULER_GAMMA)

    # Training variables
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    # Start training
    logger.info(f"Starting training on {config.CURRENT_DATASET} dataset with {config.NUM_NODES} nodes")
    start_time = time.time()

    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        epoch_train_loss = 0
        for batch_idx, (data, target, batch_times) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            output = model(data, adj_matrices)
            loss = criterion(output, target.transpose(1, 2))

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{config.EPOCHS} | Batch {batch_idx}/{len(train_loader)}")

        # Calculate average epoch loss
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0

        with torch.no_grad():
            for data, target, batch_times in val_loader:
                output = model(data, adj_matrices)
                loss = criterion(output, target.transpose(1, 2))
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

        # Learning rate scheduler step
        scheduler.step()

        # Log epoch results
        logger.info(
            f"Epoch {epoch + 1}/{config.EPOCHS} completed")

        # Check for early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0

            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(config.CHECKPOINT_DIR, f"{config.CURRENT_DATASET}_best_model.pth"))

            logger.info(f"Saved new best model at epoch {epoch + 1}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Training completed
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    return train_losses, val_losses