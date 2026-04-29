"""
NYC出租车行程时间预测项目 - 主配置文件
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据路径配置
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 模型保存路径
MODEL_DIR = PROJECT_ROOT / "models"

# 结果保存路径
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# 日志配置
LOG_DIR = PROJECT_ROOT / "logs"

# 创建必要的目录
for dir_path in [PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 随机种子
RANDOM_SEED = 42

# 实验配置
EXPERIMENT_CONFIG = {
    "test_size": 0.2,
    "val_size": 0.1,
    "random_seed": RANDOM_SEED,
    "target_column": "trip_duration",
    
    # 异常值过滤配置
    "min_trip_duration": 10,  # 最短10秒
    "max_trip_duration": 10800,  # 最长3小时(10800秒)
    "min_passenger_count": 1,
    "max_passenger_count": 6,
    
    # 纽约市地理范围
    "nyc_bounds": {
        "min_lat": 40.5,
        "max_lat": 40.9,
        "min_lon": -74.3,
        "max_lon": -73.7,
    },
}

# 模型配置
MODEL_CONFIGS = {
    "linear_regression": {
        "name": "Linear Regression",
        "type": "baseline",
    },
    "random_forest": {
        "name": "Random Forest",
        "type": "tree_based",
        "n_estimators": 100,
        "max_depth": 10,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    },
    "xgboost": {
        "name": "XGBoost",
        "type": "gradient_boosting",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    },
    "lightgbm": {
        "name": "LightGBM",
        "type": "gradient_boosting",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": -1,
    },
}

# 性能优化配置
OPTIMIZATION_CONFIG = {
    "data_types": {
        "int_columns": ["VendorID", "PULocationID", "DOLocationID", "passenger_count"],
        "float_columns": ["trip_distance", "fare_amount", "total_amount"],
        "category_columns": ["store_and_fwd_flag", "payment_type"],
    },
    "chunk_size": 100000,  # 分块读取大小
}
