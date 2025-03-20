
## Project Overview

BiTSENet highway traffic state estimation system is a comprehensive traffic flow prediction and estimation framework consisting of two core modules:

1. **Prediction Module**: A deep learning model based on Bidirectional Time-Space Encoding Network (Bi-TSENet), used for high-precision prediction of traffic states in areas with sensor coverage
2. **Traffic Physical Estimation Module**: A physical model based on traffic flow theory that uses outputs from the prediction module to estimate traffic states in road segments without sensor coverage (sparse sensors)

This system is particularly suitable for highway traffic monitoring systems with insufficient sensor deployment, capable of providing traffic state estimation for the entire road network and supporting traffic management decisions with data.

**Recommendation**: Download the complete Bi-TSENet-pytorch-master.rar package [download link: https://github.com/AIcharon-stt/Bi-TSENet-Traffic-state-estimation/blob/main/Bi-TSENet-pytorch-master.rar] to access the full codebase and examples.


![The framework of the proposed Bi-TSENet_00](https://github.com/user-attachments/assets/7d973595-e66b-4dfb-8fa0-72c4e2941f6a)




## Model Details

### Bi-TSENet Prediction Module

The Bi-TSENet (Bidirectional Time-Space Encoding Network) model combines Graph Convolutional Networks (GCN) and Temporal Convolutional Networks (TCN) to simultaneously capture spatial dependencies and temporal dynamics of traffic networks:

- **Multi-Relation GCN**: Processes multiple spatial relationships including adjacency, distance, and similarity
- **Bidirectional TCN**: Captures long-term and short-term temporal dependencies through forward and backward time series analysis
- **Feature Fusion**: Organically combines spatial and temporal features to generate predictions for multiple time ranges

### Traffic Physical Estimation Module

The physical estimation module is based on traffic flow theory, considering road segment geometry and dynamic traffic states to estimate traffic conditions in "blind segments" without sensor coverage:

- **Scene Classification**: Classifies road segments into five scene types based on ramp configurations
- **Ramp Position Modeling**: Considers the precise position of ramps within segments and their impact on traffic flow
- **Dynamic Diversion Coefficients**: Calculates vehicle entry and exit ratios at ramps based on ETC data
- **Traffic State Awareness**: Identifies free flow, transition, and congestion states based on flow/capacity ratios

## Project Structure

```
bi-tsenet/
├── configs.py                           # Configuration management
├── data/                                # Data storage
│   ├── data1/                           # Model training dataset
│   │   ├── data1_adj.csv                # Graph adjacency matrix
│   │   ├── data1_distance.csv           # Graph distance matrix
│   │   ├── data1_similarity.csv         # Graph similarity matrix
│   │   └── data1_trafficflow.csv        # Traffic flow data
│   └── data2/                           # Test data
│       └── ETC_data_example/            # Auto-generated ETC test data
│           ├── roadETC.csv              # Road segment data
│           ├── raw_data_all.csv         # ETC transaction records
│           ├── flow/                    # Historical traffic flow data
│           └── prediction/              # Prediction traffic flow data
├── generate_test_example_data.py        # Test data generation tool
├── models/                              # Model definitions
│   ├── stgcn/                           # STGCN-related modules
│   │   ├── tcn.py                       # Temporal feature extraction
│   │   └── gcn.py                       # Graph convolutional layers
│   ├── bi_tsenet.py                     # Bidirectional time-space encoding network model
│   └── traffic_physical_estimation/     # Traffic physical estimation module
│       └── blind_segment_estimation.py  # Blind segment estimation algorithm
├── preprocess.py                        # Data preprocessing
├── train.py                             # Training entry script
├── test.py                              # Testing entry script
├── metrics.py                           # Evaluation metrics
├── visualization.py                     # Visualization module
├── run_estimation.py                    # Physical estimation module entry
├── outputs/                             # Results output
│   ├── checkpoints/                     # Model weights storage
│   ├── logs/                            # Training logs
│   ├── loss_curves/                     # Loss curves
│   ├── physical_estimation_results/     # Physical estimation outputs
│   └── predictions/                     # Prediction results
│       ├── pred_flow/                   # Predicted flow
│       └── real_flow/                   # Actual flow
├── parameter_results/                   # Physical model parameter results
│   ├── travel_times.csv                 # Vehicle travel times
│   └── diversion_coefficients.csv       # Ramp diversion coefficients
├── main.py                              # Main program entry
├── requirements.txt                     # Project dependencies
└── README.md                            # Project documentation
```

## Environment Requirements

### Installing Dependencies

```bash
pip install -r requirements.txt
```

Or manually install the following dependencies:
```bash
pip install torch pandas numpy matplotlib scipy scikit-learn tqdm
```

## Usage Instructions

### 1. Prediction Module (Bi-TSENet)

#### Data Preparation

1. Place data in the `data/data1/` directory, including:
   - `data1_adj.csv`: Adjacency matrix
   - `data1_distance.csv`: Distance matrix
   - `data1_similarity.csv`: Similarity matrix
   - `data1_trafficflow.csv`: Traffic flow data

2. Traffic flow data format should include the following columns:
   - `Time`: Timestamp
   - `B1`, `B2`, `B3`, `T1`, `T2`, `T3`: Flow values for different vehicle types

#### Running the Bi-TSENet Model

Use the `main.py` script to run the prediction module:

```bash
# Complete process (training, testing, visualization)
python main.py --mode all --batch_size 64 --epochs 100 --lr 0.0001 --bidirectional

# Training only
python main.py --mode train --epochs 100 --batch_size 64

# Testing only (for trained models)
python main.py --mode test

# Visualization only
python main.py --mode visualize
```

Parameter descriptions:
- `--mode`: Running mode, options include `train`, `test`, `visualize`, or `all`
- `--batch_size`: Batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--bidirectional`: Whether to use bidirectional TCN
- `--relation_aggregation`: Relation aggregation method, options include `weighted_sum`, `attention`, or `concat`

### 2. Traffic Physical Estimation Module

After running the prediction module, the Bi-TSENet model will generate predicted flow data in the `outputs/predictions/pred_flow/` directory. Next, run the physical estimation module to process road segments without sensor coverage:

```bash
python models/traffic_physical_estimation/blind_segment_estimation.py \
    --road_data ./ETC_data_example/roadETC.csv \
    --etc_data ./ETC_data_example/raw_data_all.csv \
    --flow_dir ./ETC_data_example/flow \
    --pred_dir ./outputs/predictions/pred_flow \
    --output_dir ./validation_results \
    --parameter_dir ./parameter_results \
    --time_window 5 \
    --position_weight 0.5 \
    --add_noise
```

Parameter descriptions:
- `--road_data`: Road segment data file path
- `--etc_data`: ETC data file path
- `--flow_dir`: Historical traffic flow data directory
- `--pred_dir`: Predicted traffic flow data directory (Bi-TSENet output)
- `--output_dir`: Output directory
- `--parameter_dir`: Parameter storage directory
- `--time_window`: Time window (minutes)
- `--position_weight`: Ramp position impact weight (0-1)
- `--add_noise`: Add random noise to simulate real conditions
- `--demand_times`: Demand time options, default [5,15,30,60] minutes
- `--force_recalculate`: Force recalculation of parameters, ignore existing files

## Detailed Model Architecture

### Bi-TSENet Deep Architecture

Bi-TSENet consists of three main components:

1. **Multi-Relation Graph Convolutional Network (GCN)**:
   - `MultiRelationGCNLayer`: Simultaneously processes three relationship types: adjacency, distance, and similarity
   - `GCNBlock`: Multiple stacked GCN layers for spatial feature extraction
   - Three relation aggregation modes: weighted sum, attention mechanism, feature concatenation

2. **Bidirectional Temporal Convolutional Network (Bi-TCN)**:
   - `TemporalBlock`: Basic unit using dilated convolutions
   - `BiDirectionalTCN`: Simultaneously analyzes forward and backward time series
   - Residual connections enhance gradient propagation

3. **Prediction Layer**:
   - Feature projection layer transforms extracted spatiotemporal features
   - Multi-step prediction output layer generates predictions for multiple future time points

### Traffic Physical Estimation Core Components

The physical estimation module includes several key computational units:

1. **Scene Classification**:
   - Scene 1: Segments without ramps
   - Scene 2: Segments with upstream entry ramp
   - Scene 3: Segments with upstream exit ramp
   - Scene 4: Segments with upstream entry and exit ramps
   - Scene 5: Special segments (tunnels, bridges, etc.)

2. **Vehicle Travel Time Calculation**:
   - Extracts vehicle transit time based on ETC data
   - Dynamically adjusts for vehicle type and time period characteristics

3. **Ramp Diversion Coefficient Calculation**:
   - Analyzes ETC vehicle entry and exit patterns at ramps
   - Dynamically adjusts ramp flow impact factors

4. **Position-Aware Flow Estimation**:
   - Precisely models the impact of ramp positions on traffic flow
   - Selects historical or prediction data based on demand time and travel time
   - Uses differentiated estimation algorithms for different scene types

5. **Traffic State Determination**:
   - Determines traffic state based on flow/capacity ratio
   - Three states: free flow, transition, congestion
   - Dynamically adjusts thresholds to adapt to different time period characteristics

## Output Results

### Prediction Module Outputs

1. **Model Checkpoints**:
   - `outputs/checkpoints/data1_best_model.pth`: Saved best model weights

2. **Predicted Flow**:
   - `outputs/predictions/pred_flow/prediction_G*.csv`: Predicted flow for each gantry
   - `outputs/predictions/real_flow/real_G*.csv`: Actual flow data

3. **Visualization Results**:
   - `outputs/loss_curves/data1_loss_curves.pdf`: Training and validation loss curves
   - `outputs/predictions/data1_h*_error_distribution.pdf`: Prediction error distribution plots

### Physical Estimation Module Outputs

1. **Parameter Results**:
   - `parameter_results/travel_times.csv`: Vehicle travel time data
   - `parameter_results/diversion_coefficients.csv`: Ramp diversion coefficients

2. **Validation Results**:
   - `validation_results/validation_results.csv`: Validation results summary
   - `validation_results/metrics/`: Multiple CSV files containing detailed evaluation metrics
   - `validation_results/validation.log`: Detailed validation log

## Baseline Models Reference

1. Graph Neural Network (GNN) Based Baseline Models
  * **Reference Link:** [https://github.com/AIcharon-stt/Traffic-prediction-models-GNN](https://github.com/AIcharon-stt/Traffic-prediction-models-GNN)

2. Classic Deep Learning/Machine Learning Based Baseline Models
  * **Reference Link:** [https://github.com/AIcharon-stt/ClassicML-DL-Models](https://github.com/AIcharon-stt/ClassicML-DL-Models)


## Important Notes

1. Ensure correct data format, with timestamp format as `%d/%m/%Y %H:%M:%S`
2. The physical estimation module should only be run after the prediction module has completed training
3. The physical estimation module requires ETC data and road topology data
4. Weaving sections (where entry and exit ramps are close together) may have larger estimation errors
5. It is recommended to use the `--force_recalculate` parameter to recalculate all parameters for the first run

## Research Applications

This system is suitable for the following scenarios:

1. Highway traffic management and monitoring
2. Traffic flow prediction and congestion warning
3. Intelligent transportation system development
4. Traffic planning and design evaluation
5. Traffic management for large events or special circumstances

---

## Citation

If you use the Bi-TSENet model in your research, please cite the following paper:

```
To be added
```

For any questions, please contact us at [ttshi3514@163.com] or [1765309248@qq.com].
