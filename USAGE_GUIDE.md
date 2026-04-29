# 项目使用指南

## 📖 目录

1. [环境准备](#环境准备)
2. [快速开始](#快速开始)
3. [详细使用说明](#详细使用说明)
4. [常见问题](#常见问题)

---

## 环境准备

### 1. 安装Python依赖

```bash
cd e:\BigData\nyc_taxi_project
pip install -r requirements.txt
```

**核心依赖说明**:
- `pandas>=2.0.0`: 数据处理
- `pyarrow>=12.0.0`: Parquet文件读取
- `scikit-learn>=1.3.0`: 机器学习模型
- `lightgbm>=4.0.0`: 高效梯度提升树
- `xgboost>=2.0.0`: XGBoost模型
- `matplotlib>=3.7.0`: 数据可视化
- `seaborn>=0.12.0`: 统计可视化
- `polars>=0.19.0`: 高性能数据处理（可选）
- `psutil>=5.9.0`: 系统资源监控

### 2. 数据准备

确保 `data/raw/` 目录下有以下文件：
- `yellow_tripdata_2025-01.parquet`
- `yellow_tripdata_2025-02.parquet`
- `yellow_tripdata_2025-03.parquet`

如果没有这些文件，可以：
1. 从Kaggle下载NYC Taxi Trip Duration数据集
2. 转换为Parquet格式
3. 放置到 `data/raw/` 目录

---

## 快速开始

### 方式1: 运行快速演示（推荐新手）

```bash
python quick_demo.py
```

这将在5-10分钟内展示项目的核心功能，使用50,000条样本数据。

### 方式2: 运行完整流程

```bash
python main.py
```

这将处理所有数据并生成完整的实验报告（可能需要30-60分钟）。

### 方式3: 分步执行

```python
# 在Python环境中
from main import NYCTaxiProject

project = NYCTaxiProject()

# 分步执行各个阶段
project.load_data(use_sample=True)  # 加载数据
project.run_exploration()            # 数据探索
project.run_cleaning()               # 数据清洗
project.run_feature_engineering()    # 特征工程
project.run_model_training()         # 模型训练
project.run_performance_optimization() # 性能优化
project.generate_final_report()      # 生成报告
```

---

## 详细使用说明

### 一、数据探索分析

```python
from src.exploration import DataExplorer
import pandas as pd

# 加载数据
df = pd.read_parquet("data/raw/yellow_tripdata_2025-01.parquet")

# 创建探索器
explorer = DataExplorer(df, output_dir="results/figures")

# 生成完整EDA报告
explorer.generate_eda_report()

# 或者单独生成某个图表
explorer.plot_trip_duration_distribution()
explorer.plot_pickup_hour_distribution()
explorer.plot_location_heatmap()
explorer.plot_passenger_count_distribution()
explorer.plot_peak_hours_trend()
```

**生成的图表**:
- `trip_duration_distribution.png`: 行程时间分布
- `pickup_hour_distribution.png`: 上车时间小时分布
- `location_heatmap.png`: 位置热力图
- `passenger_count_distribution.png`: 乘客数分布
- `peak_hours_trend.png`: 高峰期趋势

### 二、数据清洗

```python
from src.data_cleaning import DataCleaner
from config import EXPERIMENT_CONFIG

cleaner = DataCleaner(EXPERIMENT_CONFIG)
cleaned_df = cleaner.clean(raw_df)

# 获取清洗统计
stats = cleaner.get_cleaning_stats()
print(f"删除记录: {stats['removed_count']}")
print(f"删除比例: {stats['removal_rate']:.2f}%")
```

**清洗步骤**:
1. 计算trip_duration（秒）
2. 过滤异常值（<10秒或>3小时）
3. 过滤乘客数异常（0或>6）
4. 过滤坐标异常（如果存在经纬度）
5. 数据类型优化（节省内存）

### 三、特征工程

```python
from src.feature_engineering import FeatureEngineer

fe = FeatureEngineer()
feature_df = fe.transform(cleaned_df)

print(f"新增特征: {feature_df.shape[1] - cleaned_df.shape[1]}个")
```

**构建的特征**:

**时间特征**:
- `pickup_hour`, `pickup_minute`, `pickup_weekday`
- `is_weekend`, `is_night`, `is_rush_hour`
- `pickup_period` (时段划分)

**空间特征**:
- `location_id_diff` (上下车地点ID差)
- `is_same_zone` (是否同区域)
- `is_top_pickup_zone`, `is_top_dropoff_zone`

**费用特征**:
- `fare_per_km` (每公里费用)
- `tip_ratio` (小费比例)
- `has_congestion_fee` (是否有拥堵费)

**交叉特征**:
- `hour_passenger` (小时×乘客数)
- `weekend_rush` (周末×高峰)

### 四、模型训练

```python
from src.model_training import ModelTrainer
from config import MODEL_CONFIGS

trainer = ModelTrainer(MODEL_CONFIGS)

# 准备数据
X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = \
    trainer.prepare_data(feature_df, target_col='trip_duration')

# 训练多个模型
trainer.train_linear_regression(X_train, y_train)  # Baseline
trainer.train_random_forest(X_train, y_train)       # Random Forest
trainer.train_lightgbm(X_train, y_train, X_val, y_val)  # LightGBM

# 模型对比
comparison_df = trainer.compare_models(X_test, y_test)
```

**模型说明**:
- **Linear Regression**: 基线模型，快速简单
- **Random Forest**: 集成学习，鲁棒性强
- **XGBoost**: 梯度提升树，精度高
- **LightGBM**: 高效梯度提升，推荐使用

### 五、性能优化

```python
from src.performance_optimizer import PerformanceOptimizer

optimizer = PerformanceOptimizer()

# 测量内存占用
original_memory = optimizer.measure_memory(df, "原始数据")

# 数据类型优化
optimized_df = optimizer.optimize_data_types_polars(df)

# 向量化计算距离
df_with_distance = optimizer.vectorized_distance_calculation(df)

# 生成性能报告
report = optimizer.generate_performance_report()
```

**优化方法**:
1. **数据类型压缩**: int64→int32, float64→float32
2. **Polars加速**: 更快的CSV/Parquet读取
3. **向量化计算**: 替代for循环
4. **分块读取**: 降低内存峰值
5. **并行处理**: 多核CPU利用

### 六、结果可视化

```python
from src.visualization import ResultVisualizer

visualizer = ResultVisualizer(output_dir="results/figures")

# 模型对比图
visualizer.plot_model_comparison(comparison_df)

# 训练时间对比
visualizer.plot_training_time_comparison(training_times)

# 特征重要性
visualizer.plot_feature_importance(
    feature_cols, 
    importance_scores, 
    model_name="LightGBM"
)

# 预测vs实际
visualizer.plot_prediction_vs_actual(
    y_test.values, 
    y_pred, 
    model_name="LightGBM"
)
```

---

## 常见问题

### Q1: 内存不足怎么办？

**解决方案**:
```python
# 使用更小的样本
project.load_data(use_sample=True, sample_size=100000)

# 或者分块读取
from src.performance_optimizer import PerformanceOptimizer
optimizer = PerformanceOptimizer()
df = optimizer.optimize_with_chunking("data.csv", chunk_size=50000)
```

### Q2: 如何自定义模型参数？

修改 `config.py` 中的 `MODEL_CONFIGS`:

```python
MODEL_CONFIGS = {
    "lightgbm": {
        "n_estimators": 1000,  # 增加树的数量
        "max_depth": 8,        # 调整深度
        "learning_rate": 0.05, # 降低学习率
        "num_leaves": 63,      # 增加叶子节点
    }
}
```

### Q3: 如何添加新的特征？

在 `src/feature_engineering.py` 中添加新方法：

```python
def _extract_custom_features(self, df):
    """自定义特征"""
    df['my_feature'] = df['A'] / df['B']
    return df
```

然后在 `transform()` 方法中调用。

### Q4: 如何保存和加载模型？

```python
# 保存模型
trainer.save_model('lightgbm', 'models/lightgbm_model.pkl')

# 加载模型
trainer.load_model('lightgbm', 'models/lightgbm_model.pkl')
```

### Q5: 测试失败怎么办？

```bash
# 查看详细错误
python -m pytest test_data_cleaning.py -v --tb=short

# 只运行特定测试
python -m pytest test_data_cleaning.py::TestDataCleaner::test_calculate_trip_duration -v
```

### Q6: 中文显示乱码？

确保已安装中文字体：
```python
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
```

---

## 项目扩展

### 添加新模型

1. 在 `src/model_training.py` 中添加训练方法
2. 在 `config.py` 中添加配置
3. 在主程序中调用

### 自定义评估指标

修改 `src/model_training.py` 中的 `evaluate_model()` 方法。

### 超参数搜索

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
```

---

## 技术支持

如有问题，请查看：
1. `logs/main.log` - 运行日志
2. `README.md` - 项目说明
3. 单元测试文件 - 代码示例

---

**祝使用愉快！** 🚀
