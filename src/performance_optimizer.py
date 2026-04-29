"""
性能优化模块
包含数据处理和模型训练的性能优化实现
"""
import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        """初始化性能优化器"""
        self.timings = {}
        self.memory_usage = {}
    
    def measure_time(self, func, name: str, *args, **kwargs):
        """
        测量函数执行时间
        
        Args:
            func: 要测量的函数
            name: 任务名称
            *args, **kwargs: 函数参数
            
        Returns:
            函数返回值
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        self.timings[name] = elapsed
        logger.info(f"⏱️  {name}: {elapsed:.2f}秒")
        
        return result
    
    def measure_memory(self, df: pd.DataFrame, label: str = "DataFrame"):
        """
        测量DataFrame内存占用
        
        Args:
            df: DataFrame对象
            label: 标签名称
        """
        memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        self.memory_usage[label] = memory_mb
        logger.info(f"💾 {label} 内存占用: {memory_mb:.2f} MB")
        return memory_mb
    
    def optimize_data_types_polars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用Polars优化数据类型（如果可用）
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            优化后的DataFrame
        """
        try:
            import polars as pl
            
            logger.info("使用Polars进行数据读取优化...")
            
            # 转换为Polars DataFrame
            pdf = pl.from_pandas(df)
            
            # Polars自动优化数据类型
            optimized_pdf = pdf.with_columns([
                pl.col(col).cast(pl.Int32) if col in ['VendorID', 'PULocationID', 'DOLocationID'] 
                else pl.col(col)
                for col in pdf.columns
            ])
            
            # 转回Pandas
            df_optimized = optimized_pdf.to_pandas()
            
            logger.info("✅ Polars优化完成")
            return df_optimized
            
        except ImportError:
            logger.warning("⚠️  Polars未安装，跳过此优化。安装命令: pip install polars")
            return df
    
    def optimize_pandas_read_csv(self, file_path: str):
        """
        优化的Pandas CSV读取方法
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            DataFrame
        """
        logger.info(f"使用优化方式读取CSV: {file_path}")
        
        # 方法1: 指定数据类型
        dtypes = {
            'VendorID': 'int32',
            'passenger_count': 'float32',
            'trip_distance': 'float32',
            'RatecodeID': 'float32',
            'PULocationID': 'int32',
            'DOLocationID': 'int32',
            'payment_type': 'int32',
            'fare_amount': 'float32',
            'extra': 'float32',
            'mta_tax': 'float32',
            'tip_amount': 'float32',
            'tolls_amount': 'float32',
            'improvement_surcharge': 'float32',
            'total_amount': 'float32',
            'congestion_surcharge': 'float32',
            'Airport_fee': 'float32',
            'cbd_congestion_fee': 'float32',
        }
        
        parse_dates = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
        
        df = pd.read_csv(
            file_path,
            dtype=dtypes,
            parse_dates=parse_dates,
            usecols=lambda x: x not in ['store_and_fwd_flag'],  # 暂时排除类别特征
        )
        
        return df
    
    def optimize_with_chunking(self, file_path: str, chunk_size: int = 100000):
        """
        分块读取大数据文件以节省内存
        
        Args:
            file_path: 文件路径
            chunk_size: 每块大小
            
        Returns:
            处理后的DataFrame
        """
        logger.info(f"使用分块读取，chunk_size={chunk_size}")
        
        chunks = []
        total_rows = 0
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # 在这里可以进行预处理
            total_rows += len(chunk)
            chunks.append(chunk)
            
            logger.info(f"已读取 {total_rows:,} 行")
        
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"合并完成，总共 {len(df):,} 行")
        
        return df
    
    def vectorized_distance_calculation(self, df: pd.DataFrame):
        """
        向量化计算距离（替代for循环）
        
        Args:
            df: 包含经纬度的DataFrame
            
        Returns:
            添加距离特征的DataFrame
        """
        if all(col in df.columns for col in ['pickup_longitude', 'pickup_latitude', 
                                             'dropoff_longitude', 'dropoff_latitude']):
            logger.info("使用向量化计算距离...")
            
            # 向量化曼哈顿距离
            df['manhattan_distance'] = (
                np.abs(df['dropoff_longitude'] - df['pickup_longitude']) +
                np.abs(df['dropoff_latitude'] - df['pickup_latitude'])
            )
            
            # 向量化欧氏距离
            df['euclidean_distance'] = np.sqrt(
                (df['dropoff_longitude'] - df['pickup_longitude'])**2 +
                (df['dropoff_latitude'] - df['pickup_latitude'])**2
            )
            
            logger.info("✅ 向量化计算完成")
        
        return df
    
    def parallel_processing(self, df: pd.DataFrame, func, n_workers: int = 4):
        """
        并行处理数据
        
        Args:
            df: DataFrame
            func: 处理函数
            n_workers: 工作进程数
            
        Returns:
            处理后的DataFrame
        """
        from joblib import Parallel, delayed
        
        logger.info(f"使用并行处理，n_workers={n_workers}")
        
        # 分割数据
        chunks = np.array_split(df, n_workers)
        
        # 并行处理
        results = Parallel(n_jobs=n_workers)(
            delayed(func)(chunk) for chunk in chunks
        )
        
        # 合并结果
        result_df = pd.concat(results, ignore_index=True)
        
        logger.info("✅ 并行处理完成")
        
        return result_df
    
    def get_system_info(self) -> Dict:
        """获取系统资源信息"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_percent': psutil.virtual_memory().percent,
        }
        
        logger.info(f"\n系统资源信息:")
        logger.info(f"CPU核心数: {info['cpu_count']}")
        logger.info(f"CPU使用率: {info['cpu_percent']}%")
        logger.info(f"总内存: {info['memory_total_gb']:.2f} GB")
        logger.info(f"已用内存: {info['memory_used_gb']:.2f} GB ({info['memory_percent']}%)")
        
        return info
    
    def generate_performance_report(self) -> str:
        """生成性能报告"""
        report = "\n" + "="*80 + "\n"
        report += "性能优化报告\n"
        report += "="*80 + "\n\n"
        
        if self.timings:
            report += "执行时间:\n"
            for name, timing in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
                report += f"  {name:40s} {timing:8.2f}秒\n"
            report += "\n"
        
        if self.memory_usage:
            report += "内存占用:\n"
            for name, memory in sorted(self.memory_usage.items(), key=lambda x: x[1], reverse=True):
                report += f"  {name:40s} {memory:8.2f}MB\n"
            report += "\n"
        
        return report


class DataComparisonExperiment:
    """数据读取对比实验"""
    
    def __init__(self):
        self.optimizer = PerformanceOptimizer()
        self.results = {}
    
    def compare_read_methods(self, csv_path: str):
        """
        对比不同数据读取方法
        
        Args:
            csv_path: CSV文件路径
        """
        logger.info("="*80)
        logger.info("实验：数据读取方法对比")
        logger.info("="*80)
        
        # 方法1: 标准Pandas
        logger.info("\n方法1: 标准Pandas读取")
        df1 = self.optimizer.measure_time(
            pd.read_csv, "Pandas Standard", csv_path,
            parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime']
        )
        
        # 方法2: 优化Pandas
        logger.info("\n方法2: 优化Pandas读取")
        df2 = self.optimizer.measure_time(
            self.optimizer.optimize_pandas_read_csv, "Pandas Optimized", csv_path
        )
        
        # 方法3: Polars（如果可用）
        try:
            import polars as pl
            logger.info("\n方法3: Polars读取")
            
            def read_with_polars():
                return pl.read_csv(csv_path).to_pandas()
            
            df3 = self.optimizer.measure_time(
                read_with_polars, "Polars"
            )
        except ImportError:
            logger.warning("Polars未安装，跳过")
        
        # 对比结果
        logger.info("\n" + "="*80)
        logger.info("读取速度对比结果")
        logger.info("="*80)
        
        for method, timing in sorted(self.optimizer.timings.items()):
            logger.info(f"{method:30s}: {timing:.2f}秒")
        
        return self.optimizer.timings


if __name__ == "__main__":
    optimizer = PerformanceOptimizer()
    optimizer.get_system_info()
    
    print("性能优化模块测试完成")
