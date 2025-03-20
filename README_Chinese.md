# BiTSENet交通状态估计

## 项目概述

Bi-TSENet高速公路交通状态估计系统是一个综合性的交通流量预测与估计框架，由两个核心模块组成：

1. **预测模块**：基于双向时空编码网络(Bi-TSENet)的深度学习模型，用于对有传感器覆盖区域的交通状态进行高精度预测
2. **交通物理估计模块**：基于交通流理论的物理模型，利用预测模块的输出结果，估计无传感器覆盖（稀疏传感器）路段的交通状态

该系统特别适用于高速公路交通监控系统中传感器部署不足的情况，能够提供全路网的交通状态估计，为交通管理决策提供数据支持。

## 模型细节

### Bi-TSENet预测模块

Bi-TSENet（双向时空编码网络）模型结合了图卷积网络(GCN)和时间卷积网络(TCN)，能够同时捕获交通网络的空间依赖性与时间动态特性：

- **多关系GCN**：处理包括邻接、距离、相似性在内的多种空间关系
- **双向TCN**：通过前向和后向时间序列分析捕获长期和短期时间依赖关系
- **特征融合**：将空间和时间特征有机结合，生成多时间范围的预测结果

### 交通物理估计模块

物理估计模块基于交通流理论，考虑路段的几何特性和动态交通状态，对无传感器覆盖的"盲区路段"进行交通状态估计：

- **场景分类**：根据匝道配置将路段分为五种场景类型
- **匝道位置建模**：考虑匝道在路段中的精确位置对流量的影响
- **动态分流系数**：基于ETC数据计算车辆在匝道的进出比例
- **交通状态感知**：根据流量/容量比识别自由流、过渡和拥堵状态

## 项目结构

```
bi-tsenet/
├── configs.py                           # 配置管理
├── data/                                # 数据存储
│   ├── data1/                           # 模型训练数据集
│   │   ├── data1_adj.csv                # 图数据邻接矩阵
│   │   ├── data1_distance.csv           # 图数据距离矩阵
│   │   ├── data1_similarity.csv         # 图数据相似性矩阵
│   │   └── data1_trafficflow.csv        # 交通流数据
│   └── data2/                           # 测试数据
│       └── ETC_data_example/            # 自动生成的ETC测试数据
│           ├── roadETC.csv              # 路段数据
│           ├── raw_data_all.csv         # ETC交易记录
│           ├── flow/                    # 历史交通流数据
│           └── prediction/              # 预测交通流数据
├── generate_test_example_data.py        # 测试数据生成工具
├── models/                              # 模型定义
│   ├── stgcn/                           # STGCN相关模块
│   │   ├── tcn.py                       # 时间特征提取
│   │   └── gcn.py                       # 图卷积层
│   ├── bi_tsenet.py                     # 双向时空编码网络模型
│   └── traffic_physical_estimation/     # 交通物理估计模块
│       └── blind_segment_estimation.py  # 盲区路段估计算法
├── preprocess.py                        # 数据预处理
├── train.py                             # 训练入口脚本
├── test.py                              # 测试入口脚本
├── metrics.py                           # 评估指标
├── visualization.py                     # 可视化模块
├── run_estimation.py                    # 物理估计模块运行入口
├── outputs/                             # 结果输出
│   ├── checkpoints/                     # 模型权重保存
│   ├── logs/                            # 训练日志
│   ├── loss_curves/                     # 损失曲线
│   ├── physical_estimation_results/     # 物理估计输出
│   └── predictions/                     # 预测结果
│       ├── pred_flow/                   # 预测流量
│       └── real_flow/                   # 真实流量
├── parameter_results/                   # 物理模型参数结果
│   ├── travel_times.csv                 # 车辆行驶时间
│   └── diversion_coefficients.csv       # 匝道分流系数
├── main.py                              # 主程序入口
├── requirements.txt                     # 项目依赖
└── README.md                            # 项目说明
```

## 环境要求

### 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装以下依赖：
```bash
pip install torch pandas numpy matplotlib scipy scikit-learn tqdm
```

## 使用说明

### 1. 预测模块 (Bi-TSENet)

#### 数据准备

1. 将数据放在 `data/data1/` 目录下，包括：
   - `data1_adj.csv`: 邻接矩阵
   - `data1_distance.csv`: 距离矩阵 
   - `data1_similarity.csv`: 相似性矩阵
   - `data1_trafficflow.csv`: 交通流量数据

2. 交通流量数据格式应包含以下列：
   - `Time`: 时间戳
   - `B1`, `B2`, `B3`, `T1`, `T2`, `T3`: 不同车辆类型的流量

#### 运行Bi-TSENet模型

使用`main.py`脚本运行预测模块：

```bash
# 完整流程（训练、测试、可视化）
python main.py --mode all --batch_size 64 --epochs 100 --lr 0.0001 --bidirectional

# 仅训练模型
python main.py --mode train --epochs 100 --batch_size 64

# 仅测试已训练模型
python main.py --mode test

# 仅生成可视化结果
python main.py --mode visualize
```

参数说明：
- `--mode`: 运行模式，可选 `train`、`test`、`visualize` 或 `all`
- `--batch_size`: 批处理大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--bidirectional`: 是否使用双向TCN
- `--relation_aggregation`: 关系聚合方法，可选 `weighted_sum`、`attention` 或 `concat`

### 2. 交通物理估计模块

在完成预测模块运行后，Bi-TSENet模型会在`outputs/predictions/pred_flow/`目录生成预测流量数据。接下来，运行物理估计模块处理无传感器覆盖的路段：

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

参数说明：
- `--road_data`: 路段数据文件路径
- `--etc_data`: ETC数据文件路径 
- `--flow_dir`: 历史交通流数据目录
- `--pred_dir`: 预测交通流数据目录（Bi-TSENet输出）
- `--output_dir`: 输出目录
- `--parameter_dir`: 参数保存目录
- `--time_window`: 时间窗口(分钟)
- `--position_weight`: 匝道位置影响权重(0-1)
- `--add_noise`: 添加随机噪声模拟真实情况
- `--demand_times`: 需求时间选项，默认[5,15,30,60]分钟
- `--force_recalculate`: 强制重新计算参数，不读取已有文件

## 模型架构详解

### Bi-TSENet深度架构

Bi-TSENet由三个主要组件构成：

1. **多关系图卷积网络(GCN)**：
   - `MultiRelationGCNLayer`: 同时处理邻接、距离、相似性三种关系
   - `GCNBlock`: 多层GCN堆叠，提取空间特征
   - 三种关系聚合模式：加权求和、注意力机制、特征拼接

2. **双向时间卷积网络(Bi-TCN)**：
   - `TemporalBlock`: 使用空洞卷积的基本单元
   - `BiDirectionalTCN`: 同时分析正向和反向时间序列
   - 残差连接增强梯度传播

3. **预测层**：
   - 特征投影层转换抽取的时空特征
   - 多步预测输出层生成多个未来时间点的预测

### 交通物理估计核心组件

物理估计模块包含多个关键计算单元：

1. **场景分类**：
   - 场景1: 无匝道路段
   - 场景2: 上游有入口匝道
   - 场景3: 上游有出口匝道
   - 场景4: 上游有入口和出口匝道
   - 场景5: 特殊路段(隧道、桥梁等)

2. **车辆行驶时间计算**：
   - 基于ETC数据提取车辆穿行时间
   - 考虑车型、时段特性动态调整

3. **匝道分流系数计算**：
   - 分析ETC车辆在匝道的进出情况
   - 动态调整匝道流量影响因子

4. **位置感知流量估计**：
   - 精确建模匝道位置对流量的影响
   - 根据需求时间、行驶时间选择历史或预测数据
   - 针对不同场景类型使用差异化估计算法

5. **交通状态判定**：
   - 基于流量/容量比确定交通状态
   - 三种状态：自由流、过渡、拥堵
   - 动态调整阈值适应不同时段特性

## 输出结果

### 预测模块输出

1. **模型检查点**：
   - `outputs/checkpoints/data1_best_model.pth`: 保存的最佳模型权重

2. **预测流量**：
   - `outputs/predictions/pred_flow/prediction_G*.csv`: 各门架的预测流量
   - `outputs/predictions/real_flow/real_G*.csv`: 真实流量数据

3. **可视化结果**：
   - `outputs/loss_curves/data1_loss_curves.pdf`: 训练和验证损失曲线图
   - `outputs/predictions/data1_h*_error_distribution.pdf`: 预测误差分布图

### 物理估计模块输出

1. **参数结果**：
   - `parameter_results/travel_times.csv`: 车辆行驶时间数据
   - `parameter_results/diversion_coefficients.csv`: 匝道分流系数

2. **验证结果**：
   - `validation_results/validation_results.csv`: 验证结果总表
   - `validation_results/metrics/`: 包含详细评估指标的多个CSV文件
   - `validation_results/validation.log`: 详细验证日志


## 注意事项

1. 确保数据格式正确，时间戳格式为 `%d/%m/%Y %H:%M:%S`
2. 预测模块训练完成后才能运行物理估计模块
3. 物理估计模块需要ETC数据和路段拓扑数据
4. 交织区（入口与出口匝道距离较近）路段的估计误差可能较大
5. 建议首次运行时使用`--force_recalculate`参数重新计算所有参数

## 研究应用

该系统适用于以下场景：

1. 高速公路交通管理与监控
2. 交通流量预测与拥堵预警
3. 智能交通系统开发
4. 交通规划与设计评估
5. 大型活动或特殊事件的交通管理

---

---

## 引用

如果您在研究中使用了Bi-TSENet模型，请引用以下论文：

```
待添加
```



如有任何问题，请通过 [ttshi3514@163.com] 或 [1765309248@qq.com]联系我们。
