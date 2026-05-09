# NYC出租车行程时间预测与性能优化项目

## 📊 项目简介

本项目基于**NYC TLC官方2025年Yellow Taxi数据**，构建完整的出租车行程时间预测系统。项目涵盖数据探索、清洗、特征工程、模型训练、性能优化和结果可视化的全流程，重点研究程序运行效率、模型训练速度、内存占用和预测精度的优化。

### ⚠️ 重要说明

**关于数据集字段：**
- 2025年新版NYC Taxi数据**不再包含经纬度字段**（pickup_longitude/latitude等）
- 仅包含`PULocationID`和`DOLocationID`（纽约市265个出租车区域编码）
- 这是NYC TLC官方的数据格式变更，项目代码已适配此变化
- 使用LocationID构建空间特征是真实可靠的方案，详见[CLEANUP_AND_POLARS_GUIDE.txt](CLEANUP_AND_POLARS_GUIDE.txt)

**关于Polars：**
- Polars是高性能DataFrame库，比Pandas快3-10倍
- 如未安装，实验1将跳过Polars对比，仅显示Pandas结果
- 建议安装：`pip install polars`

## 🎯 项目目标

1. ✅ 完成大规模出租车订单数据的清洗与预处理
2. ✅ 构建多种机器学习模型进行行程时间预测
3. ✅ 对数据处理程序和模型训练流程进行性能调优
4. ✅ 通过对比实验分析优化前后的效率与精度变化
5. ✅ 形成可复现、可扩展的机器学习工程项目

## 📁 项目结构

```
nyc_taxi_project/
├── data/
│   ├── raw/                    # 原始数据
│   │   ├── yellow_tripdata_2025-01.parquet
│   │   ├── yellow_tripdata_2025-02.parquet
│   │   └── yellow_tripdata_2025-03.parquet
│   └── processed/              # 处理后数据
├── src/                        # 源代码
│   ├── exploration.py          # 数据探索分析
│   ├── data_cleaning.py        # 数据清洗
│   ├── feature_engineering.py  # 特征工程
│   ├── model_training.py       # 模型训练
│   ├── performance_optimizer.py # 性能优化
│   └── visualization.py        # 结果可视化
├── models/                     # 保存的模型
├── results/                    # 实验结果
│   ├── figures/                # 可视化图表
│   └── experiment_report.txt   # 实验报告
├── logs/                       # 日志文件
├── main.py                     # 主程序
├── config.py                   # 配置文件
├── verify_data_integrity.py    # 数据完整性验证
├── test_data_cleaning.py       # 单元测试
├── test_feature_engineering.py # 单元测试
├── requirements.txt            # 依赖包
└── README.md                   # 项目说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将NYC Taxi数据文件放置在 `data/raw/` 目录下（支持Parquet格式）

### 3. 运行项目

```bash
# 运行完整流程（使用样本数据加速）
python main.py

# 运行数据完整性验证
python verify_data_integrity.py

# 运行单元测试
python -m pytest test_data_cleaning.py -v
python -m pytest test_feature_engineering.py -v
```

## 📋 技术路线

### 数据处理流程

```
数据读取 → 数据清洗 → 特征工程 → 基线模型训练 → 性能优化 → 优化模型训练 → 实验评估 → 结果可视化
```

### 核心功能模块

#### 1. 数据探索 (EDA)
- ✅ 行程时间分布图
- ✅ 上车时间小时分布
- ✅ 经纬度热力图（LocationID替代）
- ✅ 不同乘客数订单数量
- ✅ 高峰期订单趋势图

#### 2. 数据清洗
- ✅ 删除异常值（trip_duration < 10秒或 > 3小时）
- ✅ 删除坐标异常点
- ✅ 删除 passenger_count = 0 的记录
- ✅ 数据类型转换与内存优化

#### 3. 特征工程
- **时间特征**: hour, weekday, month, weekend, rush_hour, is_night等
- **空间特征**: LocationID差异、是否同区域、热门区域标记
- **费用特征**: 每公里费用、小费比例、拥堵费标记
- **交叉特征**: hour×passenger_count, weekend×rush_hour等

#### 4. 模型构建
- ✅ Linear Regression (Baseline)
- ✅ Random Forest
- ✅ XGBoost
- ✅ LightGBM (推荐模型)

#### 5. 性能优化
- ✅ Pandas vs Polars 数据读取速度对比
- ✅ 数据类型压缩（int32/float32/category）
- ✅ 分块读取降低内存峰值
- ✅ 向量化计算替代for循环
- ✅ LightGBM early stopping和多线程训练

## 📊 实验设计

### 实验1: 数据预处理耗时对比
- Pandas标准读取 vs Polars读取 vs 优化Pandas读取

### 实验2: 内存占用对比
- 原始数据类型 vs 压缩数据类型

### 实验3: 模型精度对比
- Linear Regression / RF / XGBoost / LightGBM

### 实验4: 训练时间对比
- Random Forest vs LightGBM

### 实验5: 特征工程消融实验
- 无特征工程 vs 时间特征 vs 时间+空间特征

## 📈 评价指标

1. **RMSE** (均方根误差)
2. **MAE** (平均绝对误差)
3. **R²** (决定系数)
4. **Log RMSE** (Kaggle比赛指标)
5. **数据处理耗时** (秒)
6. **模型训练耗时** (秒)
7. **内存占用** (MB/GB)

## 🔧 配置说明

在 `config.py` 中修改实验配置：

```python
EXPERIMENT_CONFIG = {
    "test_size": 0.2,           # 测试集比例
    "val_size": 0.1,            # 验证集比例
    "random_seed": 42,
    
    # 异常值过滤
    "min_trip_duration": 10,    # 最短10秒
    "max_trip_duration": 10800, # 最长3小时
    "min_passenger_count": 1,
    "max_passenger_count": 6,
}
```

## 📝 预期结果

1. ✅ LightGBM获得最佳综合性能
2. ✅ 预处理速度提升3~10倍
3. ✅ 内存占用下降40%以上
4. ✅ RMSE相比基线模型显著下降
5. ✅ 构建完整可复现实验流程

## 🧪 单元测试

项目包含完整的单元测试覆盖：

- ✅ 数据清洗测试 (10个测试用例)
- ✅ 特征工程测试 (7个测试用例)
- ✅ 数据完整性验证测试 (10个测试用例)

运行所有测试：
```bash
python -m pytest test_*.py -v
```

## 📊 可视化输出

项目自动生成以下图表（保存在 `results/figures/`）：

1. **trip_duration_distribution.png** - 行程时间分布
2. **pickup_hour_distribution.png** - 上车时间分布
3. **location_heatmap.png** - 位置热力图
4. **passenger_count_distribution.png** - 乘客数分布
5. **peak_hours_trend.png** - 高峰期趋势
6. **model_comparison.png** - 模型性能对比
7. **training_time_comparison.png** - 训练时间对比
8. **feature_importance_*.png** - 特征重要性
9. **prediction_vs_actual_*.png** - 预测vs实际散点图

## 💡 使用建议

### 开发模式
```python
# 使用小样本快速测试
project = NYCTaxiProject()
project.run_full_pipeline(use_sample=True, sample_size=50000)
```

### 生产模式
```python
# 使用全部数据
project = NYCTaxiProject()
project.run_full_pipeline(use_sample=False)
```

## 🐛 常见问题

**Q: 内存不足怎么办？**
A: 在 `main.py` 中使用 `use_sample=True` 减少数据量，或增加 `sample_size` 参数

**Q: 如何添加新模型？**
A: 在 `src/model_training.py` 中添加新的训练方法，并在 `MODEL_CONFIGS` 中配置

**Q: 如何自定义特征？**
A: 修改 `src/feature_engineering.py` 中的相应方法

## 📄 许可证

本项目仅供学习和研究使用。

## 👥 贡献

欢迎提交Issue和Pull Request！

---

**祝使用愉快！** 🎉
