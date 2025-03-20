# BiTSENet交通状态估计

## 项目概述

Bi-TSENet是一个专为交通流量预测设计的深度学习模型，结合了图卷积网络（GCN）和时间卷积网络（TCN）的优势，能够同时捕获交通数据中的空间和时间依赖关系。该模型创新性地引入了多关系图卷积和双向时间处理机制，有效处理复杂的交通网络动态。

模型特点：
- 支持多种关系类型的图卷积（邻接、距离、相似性）
- 双向时间编码，更好地捕捉长期和短期依赖关系
- 多步时间预测范围（5分钟、10分钟、15分钟、30分钟、60分钟）
- 针对不同车辆类型（B1-B3, T1-T3）的分类预测

## 项目结构

```
project_root/
├── configs.py               # 配置管理
├── data/                    # 数据存储
│   ├── data1/               # 数据集
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
│       ├── pred_flow/       # 预测流量
│       └── real_flow/       # 真实流量
├── main.py                  # 主程序入口
```

## 环境要求

- Python 3.8+
- torch>=1.7.0
- NumPy
- Pandas
- Matplotlib
- SciPy
- ...

可以通过以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 使用说明

### 数据准备

1. 将数据放在 `data/data1/` 目录下，包括：
   - `data1_adj.csv`: 邻接矩阵
   - `data1_distance.csv`: 距离矩阵
   - `data1_similarity.csv`: 相似性矩阵
   - `traffic/trafficflow_G*.csv`: 交通流量数据

2. 交通流量数据格式应包含以下列：
   - `Time`: 时间戳
   - `B1`, `B2`, `B3`, `T1`, `T2`, `T3`: 不同车辆类型的流量

### 运行模型

项目可以通过 `main.py` 脚本运行，支持多种参数配置：

```bash
python main.py --mode all --batch_size 64 --epochs 100 --lr 0.0001 --bidirectional
```

参数说明：
- `--mode`: 运行模式，可选 `train`（仅训练）、`test`（仅测试）、`visualize`（仅可视化）或 `all`（全部执行）
- `--batch_size`: 批处理大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--bidirectional`: 是否使用双向TCN
- `--relation_aggregation`: 关系聚合方法，可选 `weighted_sum`、`attention` 或 `concat`

### 常见用例

1. **训练模型**:
```bash
python main.py --mode train --epochs 100 --batch_size 64
```

2. **测试已训练模型**:
```bash
python main.py --mode test
```

3. **生成可视化结果**:
```bash
python main.py --mode visualize
```

4. **完整流程（训练、测试、可视化）**:
```bash
python main.py --mode all
```

### 配置调整

可以通过修改 `configs.py` 文件来调整高级配置：

- 数据预处理参数（如归一化方法）
- 模型超参数（如GCN和TCN的层数、通道数等）
- 预测时间范围
- 评估指标参数

## 输出说明

运行后将在 `outputs/` 目录下生成以下内容：

1. **模型检查点**：
   - `outputs/checkpoints/data1_best_model.pth`: 保存的最佳模型权重

2. **日志**：
   - `outputs/logs/data1_main.log`: 详细运行日志，包含训练过程和评估指标

3. **损失曲线**：
   - `outputs/loss_curves/data1_loss_curves.pdf`: 训练和验证损失曲线图

4. **预测结果**：
   - `outputs/predictions/pred_flow/prediction_G*.csv`: 各节点的预测流量
   - `outputs/predictions/real_flow/real_G*.csv`: 真实流量数据
   - `outputs/predictions/data1_h*_error_distribution.pdf`: 预测误差分布图

## 评估指标

模型使用以下指标评估预测性能：

- **MAE** (平均绝对误差): 预测值与真实值之间的平均绝对差异
- **RMSE** (均方根误差): 预测误差的平方根的平均值
- **R²** (决定系数): 模型解释的因变量变异性比例

## 模型架构

BiTSENet模型结合了图卷积网络(GCN)和时间卷积网络(TCN)：

1. **多关系GCN**：处理交通网络的空间依赖性，支持多种关系类型的图卷积
2. **双向TCN**：捕获时间序列的双向依赖关系
3. **预测层**：根据空时特征生成多时间范围的预测

模型核心思想是同时学习交通流的空间依赖性和时间动态性，从而提高预测准确度。

## 示例流程

1. 加载和预处理交通数据
2. 使用GCN提取每个时间步的空间特征
3. 通过TCN对空间特征序列进行时间编码
4. 生成未来时间点的交通流量预测
5. 计算评估指标并可视化结果

## 注意事项

- 确保数据格式正确，时间戳格式为 `%d/%m/%Y %H:%M:%S`
- 训练时间取决于数据规模和硬件配置
- 第一次运行时会自动创建所需目录结构

---

## 引用

如果您在研究中使用了Bi-TSENet模型，请引用以下论文：

```
待添加
```



如有任何问题，请通过 [ttshi3514@163.com] 或 [1765309248@qq.com]联系我们。
