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
import json
import warnings
from datetime import datetime
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


class ExperimentRecorder:
    """实验结果记录器"""
    
    def __init__(self):
        self.experiments = {}
        self.start_time = datetime.now()
        
    def record_experiment(self, exp_name, exp_type, results, metadata=None):
        """记录单个实验结果"""
        self.experiments[exp_name] = {
            'type': exp_type,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'metadata': metadata or {}
        }
        logger.info(f"✅ 实验记录已保存: {exp_name}")
        
    def save_all_experiments(self, filepath="results/experiment_results.json"):
        """保存所有实验结果到JSON文件"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            'project': 'NYC Taxi Trip Duration Prediction',
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'experiments': self.experiments
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📊 所有实验结果已保存至: {filepath}")
        
    def generate_summary_report(self):
        """生成实验总结报告"""
        report = "\n" + "="*80 + "\n"
        report += "实验结果总结报告\n"
        report += "="*80 + "\n\n"
        
        for exp_name, exp_data in self.experiments.items():
            report += f"\n【{exp_name}】\n"
            report += f"类型: {exp_data['type']}\n"
            report += f"时间: {exp_data['timestamp']}\n"
            
            if isinstance(exp_data['results'], dict):
                for key, value in exp_data['results'].items():
                    if isinstance(value, float):
                        report += f"  {key}: {value:.4f}\n"
                    else:
                        report += f"  {key}: {value}\n"
            else:
                report += f"  结果: {exp_data['results']}\n"
            
            report += "-" * 80 + "\n"
        
        return report


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
        self.recorder = ExperimentRecorder()
        
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
    
    def run_complete_experiments(self, use_sample=True, sample_size=500000):
        """
        运行完整的对比实验流程（包含所有5个实验）
        
        Args:
            use_sample: 是否使用样本数据
            sample_size: 样本大小
        """
        try:
            start_time = time.time()
            
            logger.info("\n" + "="*80)
            logger.info("🚀 开始完整实验流程")
            logger.info("="*80)
            
            # ========== 实验1: 数据加载对比 ==========
            exp1_results = self.run_experiment_1_data_loading_comparison(
                use_sample=use_sample, 
                sample_size=min(100000, sample_size)
            )
            
            # 加载数据用于后续实验
            self.load_data(use_sample=use_sample, sample_size=sample_size)
            
            # ========== 实验2: 内存优化对比 ==========
            exp2_results = self.run_experiment_2_memory_optimization()
            
            # 数据清洗
            self.run_cleaning()
            
            # ========== 实验3: 向量化加速对比 ==========
            exp3_results = self.run_experiment_3_vectorization_speedup()
            
            # 特征工程
            self.run_feature_engineering()
            
            # ========== 实验4: 模型性能对比 ==========
            exp4_results = self.run_experiment_4_model_comparison()
            
            # ========== 实验5: 特征消融实验 ==========
            exp5_results = self.run_experiment_5_feature_ablation()
            
            # ========== 生成综合报告 ==========
            total_time = time.time() - start_time
            
            # 保存所有实验结果
            self.recorder.save_all_experiments()
            
            # 生成文本报告
            summary_report = self.recorder.generate_summary_report()
            summary_report += f"\n总耗时: {total_time/60:.2f}分钟 ({total_time:.2f}秒)\n"
            
            report_path = Path("results/comprehensive_experiment_report.txt")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(summary_report)
            
            logger.info(summary_report)
            logger.info(f"\n📄 综合报告已保存至: {report_path}")
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🎉 所有实验执行完成！")
            logger.info(f"⏱️  总耗时: {total_time/60:.2f}分钟")
            logger.info(f"📊 实验结果已保存至 results/ 目录")
            logger.info(f"{'='*80}")
            
        except Exception as e:
            logger.error(f"\n❌ 实验流程执行失败: {str(e)}", exc_info=True)
            raise
    
    def run_full_pipeline(self, use_sample=True):
        """
        运行完整流程（原有方法，保持兼容）
        
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
    
    def run_experiment_1_data_loading_comparison(self, use_sample=True, sample_size=100000):
        """
        实验1: 数据加载方法对比（Pandas vs Polars）
        
        Args:
            use_sample: 是否使用样本
            sample_size: 样本大小
        """
        logger.info("\n" + "="*80)
        logger.info("🔬 实验1: 数据读取方法性能对比")
        logger.info("="*80)
        
        # 选择第一个文件进行测试
        test_file = self.data_files[0]
        if not Path(test_file).exists():
            logger.warning(f"测试文件不存在: {test_file}，跳过实验1")
            return
        
        results = {}
        
        # 方法1: Pandas标准读取
        logger.info("\n【方法1】Pandas标准读取")
        start_time = time.time()
        df_pandas = pd.read_parquet(test_file)
        pandas_time = time.time() - start_time
        pandas_memory = df_pandas.memory_usage(deep=True).sum() / (1024 ** 2)
        
        if use_sample and len(df_pandas) > sample_size:
            df_pandas = df_pandas.sample(sample_size, random_state=RANDOM_SEED)
        
        results['pandas'] = {
            'method': 'Pandas Standard',
            'read_time_seconds': round(pandas_time, 3),
            'memory_mb': round(pandas_memory, 2),
            'num_records': len(df_pandas)
        }
        logger.info(f"  ✓ 读取时间: {pandas_time:.3f}秒")
        logger.info(f"  ✓ 内存占用: {pandas_memory:.2f} MB")
        logger.info(f"  ✓ 记录数: {len(df_pandas):,}")
        
        # 方法2: Polars读取（如果可用）
        try:
            import polars as pl
            logger.info("\n【方法2】Polars高性能读取")
            
            start_time = time.time()
            df_polars = pl.read_parquet(test_file).to_pandas()
            polars_time = time.time() - start_time
            polars_memory = df_polars.memory_usage(deep=True).sum() / (1024 ** 2)
            
            if use_sample and len(df_polars) > sample_size:
                df_polars = df_polars.sample(sample_size, random_state=RANDOM_SEED)
            
            results['polars'] = {
                'method': 'Polars',
                'read_time_seconds': round(polars_time, 3),
                'memory_mb': round(polars_memory, 2),
                'num_records': len(df_polars)
            }
            
            speedup = pandas_time / polars_time if polars_time > 0 else 0
            logger.info(f"  ✓ 读取时间: {polars_time:.3f}秒 (加速 {speedup:.2f}x)")
            logger.info(f"  ✓ 内存占用: {polars_memory:.2f} MB")
            logger.info(f"  ✓ 记录数: {len(df_polars):,}")
            
        except ImportError:
            logger.warning("  ⚠️  Polars未安装，跳过此方法")
            results['polars'] = {'method': 'Polars', 'status': 'Not Installed'}
        
        # 保存实验结果
        self.recorder.record_experiment(
            exp_name="Experiment 1: Data Loading Comparison",
            exp_type="Performance Benchmark",
            results=results,
            metadata={'test_file': test_file, 'sample_size': sample_size}
        )
        
        # 可视化对比
        if 'polars' in results and 'read_time_seconds' in results.get('polars', {}):
            self.visualizer.plot_performance_comparison(
                methods=['Pandas', 'Polars'],
                times=[results['pandas']['read_time_seconds'], results['polars']['read_time_seconds']],
                title="Data Loading Time Comparison",
                save_path="results/figures/exp1_loading_comparison.png"
            )
        
        logger.info(f"\n✅ 实验1完成！结果已记录")
        return results
    
    def run_experiment_2_memory_optimization(self):
        """
        实验2: 内存优化对比（原始数据类型 vs 压缩数据类型）
        """
        logger.info("\n" + "="*80)
        logger.info("🔬 实验2: 内存优化效果对比")
        logger.info("="*80)
        
        if self.df is None:
            logger.warning("⚠️  数据未加载，请先运行load_data()")
            return
        
        results = {}
        
        # 测量原始数据内存
        logger.info("\n【优化前】原始数据类型")
        original_memory = self.optimizer.measure_memory(self.df, "Original Data Types")
        results['before_optimization'] = {
            'memory_mb': round(original_memory, 2),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }
        
        # 应用数据类型优化
        logger.info("\n【优化后】压缩数据类型")
        cleaned_df = self.cleaner._convert_data_types(self.df.copy())
        optimized_memory = self.optimizer.measure_memory(cleaned_df, "Optimized Data Types")
        
        results['after_optimization'] = {
            'memory_mb': round(optimized_memory, 2),
            'dtypes': {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()}
        }
        
        # 计算优化效果
        memory_saved = original_memory - optimized_memory
        savings_pct = (memory_saved / original_memory * 100) if original_memory > 0 else 0
        
        results['optimization_effect'] = {
            'memory_saved_mb': round(memory_saved, 2),
            'savings_percentage': round(savings_pct, 2),
            'compression_ratio': round(original_memory / optimized_memory, 2) if optimized_memory > 0 else 0
        }
        
        logger.info(f"\n📊 优化效果总结:")
        logger.info(f"  原始内存: {original_memory:.2f} MB")
        logger.info(f"  优化后内存: {optimized_memory:.2f} MB")
        logger.info(f"  节省内存: {memory_saved:.2f} MB ({savings_pct:.2f}%)")
        logger.info(f"  压缩比: {results['optimization_effect']['compression_ratio']:.2f}x")
        
        # 保存实验结果
        self.recorder.record_experiment(
            exp_name="Experiment 2: Memory Optimization",
            exp_type="Memory Benchmark",
            results=results,
            metadata={'original_records': len(self.df)}
        )
        
        # 可视化
        self.visualizer.plot_memory_comparison(
            labels=['Before Optimization', 'After Optimization'],
            memories=[original_memory, optimized_memory],
            title="Memory Usage Comparison",
            save_path="results/figures/exp2_memory_comparison.png"
        )
        
        logger.info(f"\n✅ 实验2完成！结果已记录")
        return results
    
    def run_experiment_3_vectorization_speedup(self):
        """
        实验3: 向量化计算加速对比（循环 vs 向量化）
        使用真实的LocationID字段计算区域差异，禁止使用模拟经纬度数据
        """
        logger.info("\n" + "="*80)
        logger.info("🔬 实验3: 向量化计算加速效果")
        logger.info("="*80)
        
        if self.df is None:
            logger.warning("⚠️  数据未加载，请先运行load_data()")
            return
        
        # 使用小样本进行测试
        test_size = min(10000, len(self.df))
        test_df = self.df.head(test_size).copy()
        
        results = {}
        
        # 检查是否有LocationID字段（2025年数据的真实字段）
        has_location_ids = all(col in test_df.columns for col in 
                              ['PULocationID', 'DOLocationID'])
        
        if not has_location_ids:
            logger.error("❌ 数据中缺少PULocationID或DOLocationID字段，无法进行实验")
            logger.error("   请确保使用NYC Taxi 2025年及以后的数据格式")
            return None
        
        logger.info(f"\n✅ 使用真实LocationID字段进行向量化计算对比")
        logger.info(f"   PULocationID范围: {test_df['PULocationID'].min()} - {test_df['PULocationID'].max()}")
        logger.info(f"   DOLocationID范围: {test_df['DOLocationID'].min()} - {test_df['DOLocationID'].max()}")
        
        # 方法1: 循环计算LocationID差值（慢）
        logger.info(f"\n【方法1】Python循环计算区域ID差值 ({test_size:,} 条记录)")
        
        def loop_location_diff(df):
            """使用循环计算上下车区域ID的绝对差值"""
            location_diffs = []
            for i in range(len(df)):
                diff = abs(df.iloc[i]['DOLocationID'] - df.iloc[i]['PULocationID'])
                location_diffs.append(diff)
            return location_diffs
        
        start_time = time.time()
        _ = loop_location_diff(test_df)
        loop_time = time.time() - start_time
        
        results['loop_method'] = {
            'computation_time_seconds': round(loop_time, 4),
            'num_records': test_size,
            'computation_type': 'LocationID Difference (Loop)'
        }
        logger.info(f"  ✓ 计算时间: {loop_time:.4f}秒")
        
        # 方法2: 向量化计算LocationID差值（快）
        logger.info(f"\n【方法2】NumPy向量化计算区域ID差值 ({test_size:,} 条记录)")
        
        start_time = time.time()
        # 执行多次向量化计算以获得可测量的时间
        for _ in range(100):
            vectorized_result = np.abs(test_df['DOLocationID'].values - test_df['PULocationID'].values)
        vectorized_time = (time.time() - start_time) / 100  # 取平均时间
        
        results['vectorized_method'] = {
            'computation_time_seconds': round(vectorized_time, 6),
            'num_records': test_size,
            'computation_type': 'LocationID Difference (Vectorized)'
        }
        logger.info(f"  ✓ 计算时间: {vectorized_time:.6f}秒")
        
        # 计算加速比
        speedup = loop_time / vectorized_time if vectorized_time > 0 else float('inf')
        results['speedup'] = {
            'acceleration_factor': round(speedup, 2),
            'time_saved_seconds': round(loop_time - vectorized_time, 4),
            'efficiency_improvement_pct': round((1 - vectorized_time/loop_time) * 100, 2) if loop_time > 0 else 0
        }
        
        logger.info(f"\n📊 加速效果总结:")
        logger.info(f"  循环耗时: {loop_time:.4f}秒")
        logger.info(f"  向量化耗时: {vectorized_time:.6f}秒")
        logger.info(f"  加速倍数: {speedup:.2f}x")
        logger.info(f"  效率提升: {results['speedup']['efficiency_improvement_pct']:.2f}%")
        logger.info(f"  计算方法: LocationID差值（真实数据，无模拟）")
        
        # 保存实验结果
        self.recorder.record_experiment(
            exp_name="Experiment 3: Vectorization Speedup",
            exp_type="Computation Benchmark",
            results=results,
            metadata={
                'test_size': test_size,
                'has_real_location_ids': has_location_ids,
                'uses_simulated_data': False,
                'computation_field': 'LocationID Difference'
            }
        )
        
        # 可视化 - 如果时间差异太大,不使用对数刻度
        use_log_scale = vectorized_time > 0 and (loop_time / vectorized_time) < 1000
        self.visualizer.plot_performance_comparison(
            methods=['Loop (Python)', 'Vectorized (NumPy)'],
            times=[loop_time, vectorized_time],
            title="LocationID Calculation Speed Comparison\n(Real Data - No Simulation)",
            save_path="results/figures/exp3_vectorization_comparison.png",
            log_scale=use_log_scale
        )
        
        logger.info(f"\n✅ 实验3完成！结果已记录（使用真实LocationID数据）")
        return results
    
    def run_experiment_4_model_comparison(self):
        """
        实验4: 模型精度对比（LR / RF / XGBoost / LightGBM）
        """
        logger.info("\n" + "="*80)
        logger.info("🔬 实验4: 模型性能对比实验")
        logger.info("="*80)
        
        if self.feature_df is None:
            logger.warning("⚠️  特征数据未准备，请先运行数据清洗和特征工程")
            return
        
        # 准备数据
        self.trainer = ModelTrainer(MODEL_CONFIGS)
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = \
            self.trainer.prepare_data(self.feature_df, target_col='trip_duration')
        
        results = {}
        training_times = {}
        
        # 训练所有模型
        models_to_train = ['linear_regression', 'random_forest', 'lightgbm']
        
        for model_name in models_to_train:
            logger.info(f"\n{'─'*80}")
            logger.info(f"训练模型: {model_name.upper()}")
            logger.info(f"{'─'*80}")
            
            start_time = time.time()
            
            try:
                if model_name == 'linear_regression':
                    self.trainer.train_linear_regression(X_train, y_train)
                elif model_name == 'random_forest':
                    self.trainer.train_random_forest(X_train, y_train)
                elif model_name == 'xgboost':
                    self.trainer.train_xgboost(X_train, y_train, X_val, y_val)
                elif model_name == 'lightgbm':
                    self.trainer.train_lightgbm(X_train, y_train, X_val, y_val)
                
                training_time = time.time() - start_time
                training_times[model_name] = round(training_time, 2)
                
                # 评估模型
                metrics = self.trainer.evaluate_model(
                    self.trainer.models[model_name], 
                    X_test, y_test, 
                    model_name
                )
                
                results[model_name] = {
                    'RMSE': round(metrics['RMSE'], 2),
                    'MAE': round(metrics['MAE'], 2),
                    'R2': round(metrics['R2'], 4),
                    'Log_RMSE': round(metrics['Log_RMSE'], 4),
                    'training_time_seconds': round(training_time, 2),
                    'prediction_time_seconds': round(metrics['prediction_time'], 4)
                }
                
                logger.info(f"  ✓ RMSE: {metrics['RMSE']:.2f}")
                logger.info(f"  ✓ R²: {metrics['R2']:.4f}")
                logger.info(f"  ✓ 训练时间: {training_time:.2f}秒")
                
            except Exception as e:
                logger.error(f"  ✗ 模型 {model_name} 训练失败: {str(e)}")
                results[model_name] = {'status': 'Failed', 'error': str(e)}
        
        # 模型对比总结
        logger.info(f"\n{'='*80}")
        logger.info("📊 模型性能对比总结")
        logger.info(f"{'='*80}")
        
        comparison_data = []
        for model_name, metrics in results.items():
            if 'RMSE' in metrics:
                comparison_data.append({
                    'model_name': model_name,
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'R2': metrics['R2'],
                    'Log_RMSE': metrics['Log_RMSE'],
                    'Train Time (s)': metrics['training_time_seconds']
                })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('RMSE')
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        # 保存实验结果
        self.recorder.record_experiment(
            exp_name="Experiment 4: Model Comparison",
            exp_type="Model Performance Benchmark",
            results={
                'model_metrics': results,
                'best_model': comparison_df.iloc[0]['model_name'] if len(comparison_df) > 0 else None,
                'best_rmse': float(comparison_df.iloc[0]['RMSE']) if len(comparison_df) > 0 else None
            },
            metadata={
                'train_size': len(X_train),
                'test_size': len(X_test),
                'num_features': len(feature_cols)
            }
        )
        
        # 可视化
        self.visualizer.plot_model_comparison(comparison_df)
        self.visualizer.plot_training_time_comparison(training_times)
        
        # 绘制最佳模型的特征重要性
        best_model_name = comparison_df.iloc[0]['model_name'] if len(comparison_df) > 0 else None
        if best_model_name and best_model_name in self.trainer.feature_importance:
            self.visualizer.plot_feature_importance(
                feature_cols,
                self.trainer.feature_importance[best_model_name],
                model_name=best_model_name,
                top_n=20
            )
        
        logger.info(f"\n✅ 实验4完成！结果已记录")
        return results
    
    def run_experiment_5_feature_ablation(self):
        """
        实验5: 特征工程消融实验
        对比不同特征组合对模型性能的影响
        """
        logger.info("\n" + "="*80)
        logger.info("🔬 实验5: 特征工程消融实验")
        logger.info("="*80)
        
        if self.cleaned_df is None:
            logger.warning("⚠️  清洗数据未准备，请先运行数据清洗")
            return
        
        results = {}
        
        # 实验配置：不同的特征组合
        feature_configs = {
            'baseline_no_features': {
                'name': 'Baseline (No Feature Engineering)',
                'add_time': False,
                'add_spatial': False,
                'add_fare': False,
                'add_interaction': False,
                'use_only_basic_features': True,  # 标记:仅使用基础特征
                'basic_features': ['passenger_count', 'trip_distance', 'fare_amount', 
                                  'PULocationID', 'DOLocationID']  # 基础特征列表
            },
            'time_features_only': {
                'name': 'Time Features Only',
                'add_time': True,
                'add_spatial': False,
                'add_fare': False,
                'add_interaction': False
            },
            'time_spatial_features': {
                'name': 'Time + Spatial Features',
                'add_time': True,
                'add_spatial': True,
                'add_fare': False,
                'add_interaction': False
            },
            'all_features': {
                'name': 'All Features (Time + Spatial + Fare + Interaction)',
                'add_time': True,
                'add_spatial': True,
                'add_fare': True,
                'add_interaction': True
            }
        }
        
        for config_name, config in feature_configs.items():
            logger.info(f"\n{'─'*80}")
            logger.info(f"实验配置: {config['name']}")
            logger.info(f"{'─'*80}")
            
            try:
                # 根据配置构建特征
                df_temp = self.cleaned_df.copy()
                
                # 基础特征工程师
                fe = FeatureEngineer()
                
                if config.get('add_time'):
                    df_temp = fe._extract_time_features(df_temp)
                    logger.info("  ✓ 添加时间特征")
                
                if config.get('add_spatial'):
                    df_temp = fe._extract_spatial_features(df_temp)
                    logger.info("  ✓ 添加空间特征")
                
                if config.get('add_fare'):
                    df_temp = fe._extract_fare_features(df_temp)
                    logger.info("  ✓ 添加费用特征")
                
                if config.get('add_interaction'):
                    df_temp = fe._create_interaction_features(df_temp)
                    logger.info("  ✓ 添加交叉特征")
                
                # 训练LightGBM模型进行评估
                trainer_temp = ModelTrainer(MODEL_CONFIGS)
                
                # 关键修改:如果是baseline配置,只使用基础特征
                if config.get('use_only_basic_features'):
                    basic_features = config.get('basic_features', [])
                    # 确保这些特征存在于数据中
                    available_basic_features = [f for f in basic_features if f in df_temp.columns]
                    X = df_temp[available_basic_features].copy()
                    y = df_temp['trip_duration'].copy()
                    
                    # 删除NaN
                    mask = ~y.isna() & X.notna().all(axis=1)
                    X = X[mask]
                    y = y[mask]
                    
                    # 分割数据
                    from sklearn.model_selection import train_test_split
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X, y, test_size=0.2, random_state=RANDOM_SEED
                    )
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED
                    )
                    
                    feature_cols = available_basic_features
                    logger.info(f"  使用基础特征: {len(feature_cols)}个 - {feature_cols}")
                else:
                    # 标准流程:使用prepare_data自动选择所有特征
                    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = \
                        trainer_temp.prepare_data(df_temp, target_col='trip_duration')
                
                logger.info(f"  特征数量: {len(feature_cols)}")
                
                # 快速训练（减少n_estimators加速实验）
                lgb_config = MODEL_CONFIGS['lightgbm'].copy()
                lgb_config['n_estimators'] = 100  # 减少树数量以加速
                
                trainer_temp.config['lightgbm'] = lgb_config
                trainer_temp.train_lightgbm(X_train, y_train, X_val, y_val)
                
                # 评估
                metrics = trainer_temp.evaluate_model(
                    trainer_temp.models['lightgbm'],
                    X_test, y_test,
                    'lightgbm'
                )
                
                results[config_name] = {
                    'config_name': config['name'],
                    'num_features': len(feature_cols),
                    'RMSE': round(metrics['RMSE'], 2),
                    'MAE': round(metrics['MAE'], 2),
                    'R2': round(metrics['R2'], 4),
                    'Log_RMSE': round(metrics['Log_RMSE'], 4)
                }
                
                logger.info(f"  ✓ RMSE: {metrics['RMSE']:.2f}")
                logger.info(f"  ✓ R²: {metrics['R2']:.4f}")
                logger.info(f"  ✓ 特征数: {len(feature_cols)}")
                
            except Exception as e:
                logger.error(f"  ✗ 配置 {config_name} 实验失败: {str(e)}")
                results[config_name] = {'status': 'Failed', 'error': str(e)}
        
        # 消融实验总结
        logger.info(f"\n{'='*80}")
        logger.info("📊 特征工程消融实验总结")
        logger.info(f"{'='*80}")
        
        ablation_data = []
        for config_name, metrics in results.items():
            if 'RMSE' in metrics:
                ablation_data.append({
                    'Feature Set': metrics['config_name'],
                    '# Features': metrics['num_features'],
                    'RMSE': metrics['RMSE'],
                    'R2': metrics['R2'],
                    'Log_RMSE': metrics['Log_RMSE']
                })
        
        ablation_df = pd.DataFrame(ablation_data)
        logger.info(f"\n{ablation_df.to_string(index=False)}")
        
        # 计算特征工程的提升
        if 'baseline_no_features' in results and 'all_features' in results:
            baseline_rmse = results['baseline_no_features'].get('RMSE', 0)
            full_rmse = results['all_features'].get('RMSE', 0)
            improvement = ((baseline_rmse - full_rmse) / baseline_rmse * 100) if baseline_rmse > 0 else 0
            
            logger.info(f"\n🎯 特征工程效果:")
            logger.info(f"  Baseline RMSE: {baseline_rmse:.2f}")
            logger.info(f"  Full Features RMSE: {full_rmse:.2f}")
            logger.info(f"  性能提升: {improvement:.2f}%")
            
            results['feature_engineering_impact'] = {
                'rmse_improvement_pct': round(improvement, 2),
                'baseline_rmse': baseline_rmse,
                'full_features_rmse': full_rmse
            }
        
        # 保存实验结果
        self.recorder.record_experiment(
            exp_name="Experiment 5: Feature Ablation Study",
            exp_type="Feature Engineering Analysis",
            results=results,
            metadata={'models_trained': len(feature_configs)}
        )
        
        # 可视化
        if len(ablation_df) > 0:
            self.visualizer.plot_feature_ablation_study(
                ablation_df,
                save_path="results/figures/exp5_feature_ablation.png"
            )
        
        logger.info(f"\n✅ 实验5完成！结果已记录")
        return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NYC Taxi Trip Duration Prediction')
    parser.add_argument('--mode', type=str, default='experiments', 
                       choices=['experiments', 'pipeline'],
                       help='运行模式: experiments(完整对比实验) 或 pipeline(标准流程)')
    parser.add_argument('--sample', action='store_true', default=True,
                       help='使用样本数据加速')
    parser.add_argument('--sample-size', type=int, default=500000,
                       help='样本大小（默认500000）')
    parser.add_argument('--full-data', action='store_true', default=False,
                       help='使用全量数据运行实验（约1120万条记录，耗时较长）')
    
    args = parser.parse_args()
    
    # 如果指定了--full-data，则禁用采样并使用全量数据
    if args.full_data:
        args.sample = False
        print("\n⚠️  警告: 使用全量数据运行，预计耗时10-30分钟")
        print("   数据总量: ~11,200,000 条记录")
        print("   请确保系统有足够内存 (建议16GB+)")
    
    project = NYCTaxiProject()
    
    if args.mode == 'experiments':
        # 运行完整的对比实验
        print("\n" + "="*80)
        print("🔬 运行模式: 完整对比实验（包含5个实验）")
        print("="*80)
        project.run_complete_experiments(
            use_sample=args.sample,
            sample_size=args.sample_size
        )
        
        print("\n✅ 所有实验执行完成！")
        print("\n生成的文件:")
        print("  📊 results/experiment_results.json - 所有实验结果（JSON格式）")
        print("  📄 results/comprehensive_experiment_report.txt - 综合实验报告")
        print("  📈 results/figures/exp*.png - 各实验可视化图表")
        print("  📝 logs/main.log - 运行日志")
        
    else:
        # 运行标准流程
        print("\n" + "="*80)
        print("🔄 运行模式: 标准Pipeline流程")
        print("="*80)
        project.run_full_pipeline(use_sample=args.sample)
        
        print("\n✅ 项目执行完成！")
        print("\n生成的文件:")
        print("  - results/figures/: 所有可视化图表")
        print("  - results/experiment_report.txt: 实验报告")
        print("  - logs/main.log: 运行日志")


if __name__ == "__main__":
    main()
