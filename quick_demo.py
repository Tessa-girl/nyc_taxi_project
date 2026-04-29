"""
快速演示脚本 - 展示项目核心功能
使用小样本数据快速验证整个流程
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
import time
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

from config import EXPERIMENT_CONFIG, MODEL_CONFIGS, RANDOM_SEED
from exploration import DataExplorer
from data_cleaning import DataCleaner
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from visualization import ResultVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quick_demo():
    """快速演示项目核心功能"""
    
    logger.info("="*80)
    logger.info("NYC出租车行程时间预测 - 快速演示")
    logger.info("="*80)
    
    # 1. 加载少量数据
    logger.info("\n步骤1: 加载数据（样本50,000条）")
    df = pd.read_parquet("data/raw/yellow_tripdata_2025-01.parquet")
    df_sample = df.head(50000)
    logger.info(f"加载完成: {len(df_sample):,} 条记录")
    
    # 2. 数据探索
    logger.info("\n步骤2: 数据探索")
    explorer = DataExplorer(df_sample, output_dir="results/figures")
    
    # 生成关键图表
    explorer.plot_trip_duration_distribution()
    explorer.plot_pickup_hour_distribution()
    explorer.plot_passenger_count_distribution()
    
    # 3. 数据清洗
    logger.info("\n步骤3: 数据清洗")
    cleaner = DataCleaner(EXPERIMENT_CONFIG)
    cleaned_df = cleaner.clean(df_sample)
    stats = cleaner.get_cleaning_stats()
    logger.info(f"清洗后保留: {stats['cleaned_count']:,} 条记录 ({100-stats['removal_rate']:.1f}%)")
    
    # 4. 特征工程
    logger.info("\n步骤4: 特征工程")
    fe = FeatureEngineer()
    feature_df = fe.transform(cleaned_df)
    logger.info(f"特征工程后: {feature_df.shape}")
    
    # 5. 模型训练
    logger.info("\n步骤5: 模型训练（简化版）")
    trainer = ModelTrainer(MODEL_CONFIGS)
    
    # 准备数据
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = \
        trainer.prepare_data(feature_df, target_col='trip_duration')
    
    logger.info(f"训练集: {len(X_train):,}, 验证集: {len(X_val):,}, 测试集: {len(X_test):,}")
    
    # 训练基线模型
    logger.info("\n训练线性回归 (Baseline)...")
    start_time = time.time()
    trainer.train_linear_regression(X_train, y_train)
    lr_time = time.time() - start_time
    
    # 训练LightGBM
    logger.info("\n训练LightGBM (推荐)...")
    start_time = time.time()
    trainer.train_lightgbm(X_train, y_train, X_val, y_val)
    lgbm_time = time.time() - start_time
    
    # 6. 模型评估
    logger.info("\n步骤6: 模型评估")
    comparison_df = trainer.compare_models(X_test, y_test)
    
    # 7. 可视化
    logger.info("\n步骤7: 生成可视化")
    visualizer = ResultVisualizer(output_dir="results/figures")
    
    # 模型对比图
    visualizer.plot_model_comparison(comparison_df)
    
    # 训练时间对比
    training_times = {
        'Linear Regression': lr_time,
        'LightGBM': lgbm_time
    }
    visualizer.plot_training_time_comparison(training_times)
    
    # 特征重要性
    if 'lightgbm' in trainer.feature_importance:
        visualizer.plot_feature_importance(
            feature_cols,
            trainer.feature_importance['lightgbm'],
            model_name="LightGBM",
            top_n=15
        )
    
    # 预测vs实际
    if 'lightgbm' in trainer.results:
        best_result = trainer.results['lightgbm']
        visualizer.plot_prediction_vs_actual(
            y_test.values,
            best_result['predictions'],
            model_name="LightGBM"
        )
    
    # 8. 总结
    logger.info("\n" + "="*80)
    logger.info("演示完成总结")
    logger.info("="*80)
    logger.info(f"\n✅ 生成的文件:")
    logger.info(f"   - results/figures/: 可视化图表")
    logger.info(f"   - logs/: 运行日志")
    
    logger.info(f"\n📊 最佳模型性能:")
    if 'lightgbm' in trainer.results:
        metrics = trainer.results['lightgbm']
        logger.info(f"   - RMSE: {metrics['RMSE']:.2f}")
        logger.info(f"   - MAE: {metrics['MAE']:.2f}")
        logger.info(f"   - R2: {metrics['R2']:.4f}")
    
    logger.info(f"\n⏱️  总耗时: {(time.time() - start_time)/60:.2f}分钟")
    
    print("\n🎉 快速演示完成！查看 results/figures/ 目录获取可视化结果。")


if __name__ == "__main__":
    try:
        quick_demo()
    except Exception as e:
        logger.error(f"演示失败: {str(e)}", exc_info=True)
        sys.exit(1)
