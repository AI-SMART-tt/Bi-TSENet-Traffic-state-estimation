# Traffic-state-estimation-at-sensor-free-locations-TSES
This work considers combining traffic flow models with deep learning models to achieve accurate traffic state estimation in sparse sensor coverage areas of highway networks.

# Bi-TSENet：双向时空编码网络用于交通流量预测

## 项目概述

Bi-TSENet（双向时空编码网络）是一个专为交通流量预测设计的深度学习模型，结合了图卷积网络（GCN）和时间卷积网络（TCN）的优势，能够同时捕获交通数据中的空间和时间依赖关系。该模型创新性地引入了多关系图卷积和双向时间处理机制，有效处理复杂的交通网络动态。

## 模型架构

Bi-TSENet的架构由三个主要组件构成：

### 1. 多关系图卷积网络 (Multi-Relation GCN)

处理交通网络中的空间依赖关系，能够同时考虑三种不同类型的图关系：
- **邻接关系**：描述道路网络的直接连接
- **距离关系**：基于地理距离的权重连接
- **相似度关系**：基于历史交通模式的节点相似性

多关系GCN层支持三种聚合方式：
- `weighted_sum`：通过可学习权重加权不同关系
- `attention`：通过注意力机制自适应关注重要关系
- `concat`：直接连接不同关系的特征表示

### 2. 双向时间卷积网络 (Bidirectional TCN)

处理时间序列数据，捕获交通流的时间模式：
- 使用因果卷积（Causal Convolution）处理时间数据
- 采用膨胀卷积（Dilated Convolution）扩大感受野，捕获长期依赖
- 双向处理机制同时考虑正向和反向的时间信息
- 残差连接确保深层网络的有效训练

### 3. 预测层

将提取的时空特征转换为未来交通流预测：
- 投影层将TCN输出转换为节点特征
- 预测层为每个节点生成多步预测

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## 项目结构

```
project_root/
├── configs.py               # 配置管理
├── data/                    # 数据存储
│   ├── data1/               # 数据集1
│   │   ├── data1_adj.csv    # 图数据邻接矩阵：M*M
│   │   ├── data1_distance.csv  # 图数据距离矩阵：M*M
│   │   ├── data1_similarity.csv  # 图数据相似性矩阵：M*M
│   │   └── data1_trafficflow.csv  # 交通流数据：N*M
├── models/                  # 模型定义
│   ├── stgcn/               # STGCN相关模块
│   │   ├── tcn.py           # 时间特征提取
│   │   └── gcn.py           # 图卷积层
│   ├── bi_tsenet.py         # 双向时空编码网络模型
├── preprocess.py            # 数据预处理
├── train.py                 # 训练入口脚本
├── test.py                  # 测试入口脚本
├── metrics.py               # 评估指标
├── visualization.py         # 可视化模块
├── outputs/                 # 结果输出
│   ├── checkpoints/         # 模型权重保存
│   ├── logs/                # 训练日志
│   ├── loss_curves/         # 损失曲线
│   └── predictions/         # 预测结果
├── main.py                  # 主程序入口
```

## 数据格式

本模型需要以下输入数据：

1. **交通流数据** (`data*_trafficflow.csv`): N行M列的CSV文件，其中：
   - N: 时间步数量
   - M: 节点（监测站点）数量
   - 每个单元格表示特定时间步特定节点的交通流量

2. **图关系数据**:
   - **邻接矩阵** (`data*_adj.csv`): M×M的矩阵，表示节点间的连接关系
   - **距离矩阵** (`data*_distance.csv`): M×M的矩阵，表示节点间的物理距离
   - **相似度矩阵** (`data*_similarity.csv`): M×M的矩阵，表示节点间的相似度

## 使用说明

### 训练模型

```bash
python main.py --dataset data1 --mode train --batch_size 64 --epochs 100 --lr 0.0005 --bidirectional
```

### 测试模型

```bash
python main.py --dataset data1 --mode test
```

### 可视化结果

```bash
python main.py --dataset data1 --mode visualize
```

### 完整流程（训练、测试和可视化）

```bash
python main.py --dataset data1 --mode all
```

## 参数配置

在`configs.py`中可以配置以下关键参数：

### 数据参数
- `DATASETS`: 可用数据集列表
- `CURRENT_DATASET`: 当前使用的数据集
- `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`: 数据集划分比例
- `NORMALIZATION`: 数据归一化方法（'min-max'或'z-score'）

### 模型参数
- `GCN_HIDDEN_CHANNELS`: GCN隐藏层通道数
- `GCN_DROPOUT`: GCN dropout率
- `NUM_RELATIONS`: 图关系类型数量
- `RELATION_AGGREGATION`: 关系聚合方法
- `TCN_KERNEL_SIZE`: TCN卷积核大小
- `TCN_CHANNELS`: TCN通道数配置
- `TCN_DROPOUT`: TCN dropout率
- `SEQUENCE_LENGTH`: 输入序列长度
- `HORIZON`: 预测时长
- `BIDIRECTIONAL`: 是否使用双向TCN
- `FINAL_FC_HIDDEN`: 最终全连接层隐藏单元数

### 训练参数
- `BATCH_SIZE`: 批次大小
- `LEARNING_RATE`: 学习率
- `WEIGHT_DECAY`: 权重衰减系数
- `EPOCHS`: 训练轮数
- `PATIENCE`: 早停耐心值
- `SCHEDULER_STEP`: 学习率调度步长
- `SCHEDULER_GAMMA`: 学习率衰减系数

## 评估指标

模型使用以下指标评估性能：
- MAE (平均绝对误差)
- MSE (均方误差)
- RMSE (均方根误差)
- MAPE (平均绝对百分比误差)
- R² (决定系数)

评估结果会以CSV格式保存，并生成可视化图表，包括：
- 预测与实际值的拟合散点图
- 时间序列比较图
- 整体性能指标比较图
- 误差分布图

## 模型优势

1. **多关系空间建模**：同时考虑邻接、距离和相似度三种空间关系
2. **双向时间处理**：通过正反向处理捕获更丰富的时间上下文信息
3. **膨胀卷积**：高效捕获长期时间依赖性
4. **端到端学习**：直接从原始交通流数据学习，无需手工特征工程

## 注意事项

- 交通流数据应确保为非负值
- 建议使用GPU进行训练以获得更好的性能
- 可根据具体数据集特性调整`SEQUENCE_LENGTH`和`HORIZON`参数

---

## 引用

如果您在研究中使用了Bi-TSENet模型，请引用以下论文：

```
待添加
```



如有任何问题，请通过 [ttshi3514@163.com] 联系我们。
