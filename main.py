"""
NYC出租车行程时间预测 - 主程序
整合数据探索、清洗、特征工程、模型训练和性能优化
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
from performance_optimizer import PerformanceOptimizer, DataComparisonExperiment
from visualization import ResultVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NYCTaxiProject:
    """NYC出租车行程时间预测项目主类"""
    
    def __init__(self):
        """初始化项目"""
        self.data_files = [
            "data/raw/yellow_tripdata_2025-01.parquet",
            "data/raw/yellow_tripdata_2025-02.parquet",
            "data/raw/yellow_tripdata_2025-03.parquet"
        ]
        
        self.df = None
        self.cleaned_df = None
        self.feature_df = None
        
        self.cleaner = DataCleaner(EXPERIMENT_CONFIG)
        self.feature_engineer = FeatureEngineer()
        self.trainer = None
        self.optimizer = PerformanceOptimizer()
        self.visualizer = ResultVisualizer()
        
        logger.info("="*80)
        logger.info("NYC出租车行程时间预测系统")
        logger.info("="*80)
    
    def load_data(self, use_sample: bool = True, sample_size: int = 500000):
        """
        加载数据
        
        Args:
            use_sample: 是否使用样本数据（加速开发）
            sample_size: 样本大小
        """
        logger.info("\n" + "="*80)
        logger.info("步骤1: 数据加载")
        logger.info("="*80)
        
        dfs = []
        
        for file_path in self.data_files:
            if Path(file_path).exists():
                logger.info(f"加载文件: {file_path}")
                
                # 测量读取时间
                start_time = time.time()
                df_temp = pd.read_parquet(file_path)
                read_time = time.time() - start_time
                
                logger.info(f"  - 记录数: {len(df_temp):,}")
                logger.info(f"  - 读取时间: {read_time:.2f}秒")
                
                dfs.append(df_temp)
            else:
                logger.warning(f"文件不存在: {file_path}")
        
        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
            logger.info(f"\n总记录数: {len(self.df):,}")
            
            # 如果使用样本
            if use_sample and len(self.df) > sample_size:
                logger.info(f"使用样本数据: {sample_size:,} 条")
                self.df = self.df.sample(sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
            
            # 测量内存占用
            memory_mb = self.df.memory_usage(deep=True).sum() / (1024 ** 2)
            logger.info(f"内存占用: {memory_mb:.2f} MB")
        else:
            raise FileNotFoundError("未找到任何数据文件！")
    
    def run_exploration(self):
        """运行数据探索分析"""
        logger.info("\n" + "="*80)
        logger.info("步骤2: 数据探索分析 (EDA)")
        logger.info("="*80)
        
        explorer = DataExplorer(self.df.head(100000))  # 使用样本加速
        explorer.generate_eda_report()
    
    def run_cleaning(self):
        """运行数据清洗"""
        logger.info("\n" + "="*80)
        logger.info("步骤3: 数据清洗")
        logger.info("="*80)
        
        self.cleaned_df = self.cleaner.clean(self.df)
        
        stats = self.cleaner.get_cleaning_stats()
        logger.info(f"\n清洗统计:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
    
    def run_feature_engineering(self):
        """运行特征工程"""
        logger.info("\n" + "="*80)
        logger.info("步骤4: 特征工程")
        logger.info("="*80)
        
        self.feature_df = self.feature_engineer.transform(self.cleaned_df)
        
        logger.info(f"特征工程后数据形状: {self.feature_df.shape}")
    
    def run_model_training(self):
        """运行模型训练"""
        logger.info("\n" + "="*80)
        logger.info("步骤5: 模型训练")
        logger.info("="*80)
        
        self.trainer = ModelTrainer(MODEL_CONFIGS)
        
        # 准备数据
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = \
            self.trainer.prepare_data(self.feature_df, target_col='trip_duration')
        
        # 训练不同模型
        models_to_train = ['linear_regression', 'random_forest', 'lightgbm']
        
        training_times = {}
        
        for model_name in models_to_train:
            start_time = time.time()
            
            if model_name == 'linear_regression':
                self.trainer.train_linear_regression(X_train, y_train)
            elif model_name == 'random_forest':
                self.trainer.train_random_forest(X_train, y_train)
            elif model_name == 'xgboost':
                self.trainer.train_xgboost(X_train, y_train, X_val, y_val)
            elif model_name == 'lightgbm':
                self.trainer.train_lightgbm(X_train, y_train, X_val, y_val)
            
            training_time = time.time() - start_time
            training_times[model_name] = training_time
        
        # 模型对比
        comparison_df = self.trainer.compare_models(X_test, y_test)
        
        # 可视化
        self.visualizer.plot_model_comparison(comparison_df)
        self.visualizer.plot_training_time_comparison(training_times)
        
        # 绘制最佳模型的特征重要性
        if 'lightgbm' in self.trainer.feature_importance:
            self.visualizer.plot_feature_importance(
                feature_cols,
                self.trainer.feature_importance['lightgbm'],
                model_name="LightGBM",
                top_n=20
            )
        
        # 绘制预测vs实际图
        if 'lightgbm' in self.trainer.results:
            best_result = self.trainer.results['lightgbm']
            self.visualizer.plot_prediction_vs_actual(
                y_test.values,
                best_result['predictions'],
                model_name="LightGBM"
            )
        
        return comparison_df
    
    def run_performance_optimization(self):
        """运行性能优化实验"""
        logger.info("\n" + "="*80)
        logger.info("步骤6: 性能优化实验")
        logger.info("="*80)
        
        # 获取系统信息
        self.optimizer.get_system_info()
        
        # 数据类型优化对比
        logger.info("\n数据类型优化对比:")
        original_memory = self.optimizer.measure_memory(self.df, "原始数据")
        optimized_memory = self.optimizer.measure_memory(self.cleaned_df, "优化后数据")
        
        savings = original_memory - optimized_memory
        savings_pct = (savings / original_memory) * 100
        logger.info(f"内存节省: {savings:.2f} MB ({savings_pct:.1f}%)")
    
    def generate_final_report(self, comparison_df=None):
        """生成最终报告"""
        logger.info("\n" + "="*80)
        logger.info("最终实验报告")
        logger.info("="*80)
        
        report = "\n" + "="*80 + "\n"
        report += "NYC出租车行程时间预测 - 实验报告\n"
        report += "="*80 + "\n\n"
        
        # 数据概览
        report += "1. 数据概览\n"
        report += f"   - 总记录数: {len(self.df):,}\n"
        report += f"   - 清洗后记录数: {len(self.cleaned_df):,}\n"
        report += f"   - 特征数量: {len(self.feature_df.columns)}\n\n"
        
        # 模型性能
        if comparison_df is not None:
            report += "2. 模型性能对比\n"
            report += comparison_df.to_string(index=False) + "\n\n"
        
        # 性能优化
        report += "3. 性能优化成果\n"
        perf_report = self.optimizer.generate_performance_report()
        report += perf_report + "\n"
        
        # 保存报告
        report_path = Path("results/experiment_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(report)
        logger.info(f"\n报告已保存至: {report_path}")
    
    def run_full_pipeline(self, use_sample=True):
        """
        运行完整流程
        
        Args:
            use_sample: 是否使用样本数据
        """
        try:
            start_time = time.time()
            
            # 1. 加载数据
            self.load_data(use_sample=use_sample)
            
            # 2. 数据探索
            self.run_exploration()
            
            # 3. 数据清洗
            self.run_cleaning()
            
            # 4. 特征工程
            self.run_feature_engineering()
            
            # 5. 模型训练
            comparison_df = self.run_model_training()
            
            # 6. 性能优化
            self.run_performance_optimization()
            
            # 7. 生成报告
            self.generate_final_report(comparison_df)
            
            total_time = time.time() - start_time
            logger.info(f"\n{'='*80}")
            logger.info(f"🎉 完整流程执行完成！")
            logger.info(f"⏱️  总耗时: {total_time/60:.2f}分钟")
            logger.info(f"{'='*80}")
            
        except Exception as e:
            logger.error(f"\n❌ 流程执行失败: {str(e)}", exc_info=True)
            raise


def main():
    """主函数"""
    project = NYCTaxiProject()
    
    # 运行完整流程（使用样本数据加速演示）
    project.run_full_pipeline(use_sample=True)
    
    print("\n✅ 项目执行完成！")
    print("\n生成的文件:")
    print("  - results/figures/: 所有可视化图表")
    print("  - results/experiment_report.txt: 实验报告")
    print("  - logs/main.log: 运行日志")


if __name__ == "__main__":
    main()
