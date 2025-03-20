# Bi-TSENet Traffic State Estimation Project Test Data Generator

## I. Overview

This tool generates simulated highway traffic flow data to provide test datasets for the Bi-TSENet traffic state estimation model. Since the original ETC data exceeds 1GB and cannot be uploaded to GitHub, this tool can generate complete datasets that conform to real traffic characteristics.
If you need the complete ETC data, please contact us [ttshi3514@163.com].

## II. Generated Data Structure

The tool generates the following files in the specified directory:

### 1. Road Network Data
- **roadETC.csv**: Contains basic road segment information
  - Segment ID, type, length, speed limit
  - Upstream and downstream gantry IDs
  - Ramp characteristics (position, length, speed limit)
  - Special segment features (tunnels, bridges, curves, gradients)

### 2. ETC Transaction Records
- **sample_etc.csv**: Vehicle passage records
  - Gantry ID
  - License plate number
  - Vehicle type (B1-B3 passenger vehicles, T1-T3 trucks)
  - Passage time

### 3. Historical Traffic Flow Data
- **flow/trafficflow_G*.csv**: Historical flow for each gantry
  - Timestamp (aggregated by specified time window)
  - Flow volume by vehicle type

### 4. Traffic Prediction Data
- **prediction/prediction_G*.csv**: Simulated prediction results
  - Base timestamp
  - Prediction horizon
  - Target prediction time
  - Predicted flow by vehicle type

### 5. Visualization Charts
- Visualization charts for different segment types and vehicle flows (PNG format)

## III. Advanced Traffic Simulation Features

### 1. Segment Diversity
- **Basic Segments**: Standard segments without ramps
- **Entry Ramp Segments**: Simulates traffic merging dynamics
- **Exit Ramp Segments**: Simulates traffic diverging dynamics
- **Compound Ramp Segments**: Simulates complex weaving section dynamics
- **Special Segments**: Tunnels, bridges, curves, gradients

### 2. Vehicle Type Differences
- **Passenger Vehicles (B1-B3)**: Better acceleration, more inclined to use ramps
- **Trucks (T1-T3)**: Poorer acceleration, highly affected by gradients, less likely to use ramps
- **Vehicle Type Distribution**: Conforms to real traffic distributions

### 3. Traffic State Simulation
- **Free Flow**: Speed is 75%-100% of the speed limit
- **Transition State**: Speed is 55%-75% of the speed limit
- **Congested State**: Speed is 30%-55% of the speed limit
- **Time-Based State Changes**: Higher probability of congestion during peak hours

### 4. Spatiotemporal Distribution
- **Daytime Pattern**: Weekday dual peaks (7-9 AM, 5-7 PM)
- **Weekend Pattern**: Smoother peaks, more uniform distribution
- **Nighttime Pattern**: Significantly reduced flow, primarily free flow

### 5. Ramp Impact Factors
- **Position Sensitive**: Entry ramps closer to segment start have greater impact
- **Weaving Section Length**: Shorter weaving sections have more significant impact
- **Vehicle Decision**: Different vehicle types have different ramp usage preferences

### 6. Special Segment Effects
- **Tunnels**: Speed reduction of 15%-30%
- **Curves**: Speed reduction of 20%-35%
- **Bridges**: Speed reduction of 10%-20%
- **Gradients**: Speed adjusted based on gradient and vehicle weight

### 7. Prediction Data Characteristics
- **Error Increases with Prediction Horizon**: Longer-term predictions have larger errors
- **Segment Complexity Impact**: Complex segments have larger prediction errors
- **Realistic Prediction Patterns**: Simulates over/under-estimation patterns of real prediction systems

## IV. Usage Instructions

### Basic Usage

```bash
python data/generate_test_example_data.py
```

### Custom Parameters

```bash
python data/generate_test_example_data.py --output_dir ./my_test_data --num_segments 15 --num_vehicles 2000 --days 5 --time_window 10 --no-visualize
```

### Parameter Description

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| --output_dir | Output directory | ./test_data |
| --num_segments | Number of road segments | 10 |
| --num_vehicles | Number of simulated vehicles | 1000 |
| --days | Number of simulation days | 3 |
| --time_window | Time window (minutes) | 5 |
| --no-visualize | Do not generate visualization results | False |
| --seed | Random seed | 42 |

## V. Visualization Features

After data generation, the tool automatically generates various visualization charts:

1. **Basic Segment Passenger Vehicle Flow**: Shows B1 vehicle flow on segments without ramps
2. **Entry Ramp Truck Flow**: Shows the impact of entry ramps on T1 vehicle flow
3. **Special Segment Large Vehicle Flow**: Shows B3 vehicle flow on special segments
4. **Weekend vs. Weekday Comparison**: Shows flow differences between weekdays and weekends

Charts display actual flow compared with predictions for different time horizons.

### Manual Visualization

You can also manually generate visualizations using the provided function:

```python
from generate_test_example_data import visualize_traffic_data

visualize_traffic_data(
    output_dir="./test_data",
    gantry_id="G001",
    vehicle_type="B1",
    date="2022-06-02",
    show_prediction=True
)
```

## VI. Integration with Bi-TSENet Model

The generated data can be directly used for testing the Bi-TSENet model:

1. **Prediction Module Testing**: Use historical traffic flow data to train and test the Bi-TSENet model
2. **Physical Estimation Module Testing**: Use ETC data and flow data to test the blind_segment_estimation module
3. **End-to-End Testing**: Validate the complete traffic state estimation system

## VII. Data Statistics and Performance

After execution, the tool displays the following statistics:

- Number of generated road segments
- Number of generated gantries
- Number of generated ETC records
- Number of simulation days
- Number of time points per day
- Data generation time

## VIII. Obtaining Real Data

The generated test data is sufficient for model development and testing, but researchers who need the complete real dataset (>1GB) should contact the author: ttshi3514@163.com

## IX. Important Notes

- Generating large-scale data may require significant time and memory
- Increasing the number of vehicles improves data realism but increases generation time
- Generated data is for testing only and should not be used for actual traffic management decisions
- All timestamps are in the format "%Y-%m-%d %H:%M:%S"

---