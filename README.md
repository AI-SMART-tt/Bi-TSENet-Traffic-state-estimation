# Bi-TSENet: The Bi-layer Traffic State Estimation Network framework that combines traffic flow theory with graph-based deep learning.

## Project Overview

Bi-TSENet is a deep learning model specifically designed for traffic flow prediction, combining the advantages of Graph Convolutional Networks (GCN) and Temporal Convolutional Networks (TCN) to simultaneously capture spatial and temporal dependencies in traffic data. The model innovatively introduces multi-relation graph convolution and bidirectional temporal processing mechanisms to effectively handle complex traffic network dynamics.

## Model Architecture

The Bi-TSENet architecture consists of three main components:

### 1. Multi-Relation Graph Convolutional Network

Processes spatial dependencies in traffic networks by considering three different types of graph relationships:
- **Adjacency Relation**: Describes direct connections in the road network
- **Distance Relation**: Weighted connections based on geographic distance
- **Similarity Relation**: Node similarity based on historical traffic patterns

The Multi-Relation GCN layer supports three aggregation methods:
- `weighted_sum`: Weights different relations through learnable parameters
- `attention`: Adaptively focuses on important relations through attention mechanisms
- `concat`: Directly concatenates feature representations from different relations

### 2. Bidirectional Temporal Convolutional Network

Processes time series data to capture temporal patterns in traffic flow:
- Uses causal convolutions to handle temporal data
- Employs dilated convolutions to expand the receptive field, capturing long-term dependencies
- Implements bidirectional processing to consider both forward and backward temporal information
- Utilizes residual connections to ensure effective training of deep networks

### 3. Prediction Layer

Transforms extracted spatiotemporal features into future traffic flow predictions:
- Projection layer converts TCN output into node features
- Prediction layer generates multi-step predictions for each node

## Environment Requirements

- Python 3.6+
- PyTorch 1.7+
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Project Structure

```
project_root/
├── configs.py               # Configuration management
├── data/                    # Data storage
│   ├── data1/               # Dataset 1
│   │   ├── data1_adj.csv    # Graph adjacency matrix: M*M
│   │   ├── data1_distance.csv  # Graph distance matrix: M*M
│   │   ├── data1_similarity.csv  # Graph similarity matrix: M*M
│   │   └── data1_trafficflow.csv  # Traffic flow data: N*M
├── models/                  # Model definitions
│   ├── stgcn/               # STGCN-related modules
│   │   ├── tcn.py           # Temporal feature extraction
│   │   └── gcn.py           # Graph convolutional layers
│   ├── bi_tsenet.py         # Bidirectional time-space encoding network model
├── preprocess.py            # Data preprocessing
├── train.py                 # Training entry script
├── test.py                  # Testing entry script
├── metrics.py               # Evaluation metrics
├── visualization.py         # Visualization module
├── outputs/                 # Results output
│   ├── checkpoints/         # Model weights storage
│   ├── logs/                # Training logs
│   ├── loss_curves/         # Loss curves
│   └── predictions/         # Prediction results
├── main.py                  # Main program entry
```

## Data Format

This model requires the following input data:

1. **Traffic Flow Data** (`data*_trafficflow.csv`): CSV file with N rows and M columns, where:
   - N: Number of time steps
   - M: Number of nodes (monitoring stations)
   - Each cell represents the traffic flow at a specific time step for a specific node

2. **Graph Relationship Data**:
   - **Adjacency Matrix** (`data*_adj.csv`): M×M matrix representing connections between nodes
   - **Distance Matrix** (`data*_distance.csv`): M×M matrix representing physical distances between nodes
   - **Similarity Matrix** (`data*_similarity.csv`): M×M matrix representing similarity between nodes

## Usage Instructions

### Training the Model

```bash
python run_estimation.py --dataset data1 --mode train --batch_size 64 --epochs 100 --lr 0.0005 --bidirectional
```

### Testing the Model

```bash
python run_estimation.py --dataset data1 --mode test
```

### Visualizing Results

```bash
python run_estimation.py --dataset data1 --mode visualize
```

### Complete Process (Training, Testing, and Visualization)

```bash
python run_estimation.py --dataset data1 --mode all
```

## Parameter Configuration

The following key parameters can be configured in `configs.py`:

### Data Parameters
- `DATASETS`: List of available datasets
- `CURRENT_DATASET`: Currently used dataset
- `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`: Dataset split ratios
- `NORMALIZATION`: Data normalization method ('min-max' or 'z-score')

### Model Parameters
- `GCN_HIDDEN_CHANNELS`: GCN hidden layer channel numbers
- `GCN_DROPOUT`: GCN dropout rate
- `NUM_RELATIONS`: Number of graph relation types
- `RELATION_AGGREGATION`: Relation aggregation method
- `TCN_KERNEL_SIZE`: TCN convolution kernel size
- `TCN_CHANNELS`: TCN channel configuration
- `TCN_DROPOUT`: TCN dropout rate
- `SEQUENCE_LENGTH`: Input sequence length
- `HORIZON`: Prediction horizon
- `BIDIRECTIONAL`: Whether to use bidirectional TCN
- `FINAL_FC_HIDDEN`: Final fully-connected layer hidden units

### Training Parameters
- `BATCH_SIZE`: Batch size
- `LEARNING_RATE`: Learning rate
- `WEIGHT_DECAY`: Weight decay coefficient
- `EPOCHS`: Number of training epochs
- `PATIENCE`: Early stopping patience
- `SCHEDULER_STEP`: Learning rate scheduler step size
- `SCHEDULER_GAMMA`: Learning rate decay factor

## Evaluation Metrics

The model's performance is evaluated using the following metrics:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)

Evaluation results are saved in CSV format and visualized through charts, including:
- Scatter plots of predicted vs. actual values
- Time series comparison plots
- Overall performance metric comparison charts
- Error distribution plots

## Model Advantages

1. **Multi-Relation Spatial Modeling**: Simultaneously considers adjacency, distance, and similarity relations
2. **Bidirectional Temporal Processing**: Captures richer temporal context information through forward and backward processing
3. **Dilated Convolutions**: Efficiently captures long-term temporal dependencies
4. **End-to-End Learning**: Learns directly from raw traffic flow data without manual feature engineering

## Important Notes

- Traffic flow data should be non-negative
- GPU usage is recommended for better training performance
- The `SEQUENCE_LENGTH` and `HORIZON` parameters should be adjusted according to dataset characteristics

---

## Citation

If you use the Bi-TSENet model in your research, please cite the following paper:

```
To be added
```

For any questions, please contact us at [ttshi3514@163.com].
