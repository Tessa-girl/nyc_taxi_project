"""
数据清洗模块
负责数据质量检查、异常值处理和数据转换
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    """数据清洗类"""
    
    def __init__(self, config: dict):
        """
        初始化数据清洗器
        
        Args:
            config: 配置字典，包含过滤条件等
        """
        self.config = config
        self.stats = {}  # 记录清洗统计信息
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整的数据清洗流程
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            清洗后的DataFrame
        """
        logger.info("=" * 80)
        logger.info("开始数据清洗")
        logger.info("=" * 80)
        
        original_count = len(df)
        logger.info(f"原始数据量: {original_count:,} 条")
        
        # 1. 计算trip_duration（如果不存在）
        if 'trip_duration' not in df.columns:
            df = self._calculate_trip_duration(df)
        
        # 2. 删除trip_duration异常值
        df = self._filter_trip_duration(df)
        
        # 3. 删除passenger_count异常值
        df = self._filter_passenger_count(df)
        
        # 4. 删除坐标异常（如果有经纬度字段）
        df = self._filter_location_outliers(df)
        
        # 5. 数据类型转换
        df = self._convert_data_types(df)
        
        cleaned_count = len(df)
        removed_count = original_count - cleaned_count
        removal_rate = (removed_count / original_count * 100) if original_count > 0 else 0
        
        logger.info(f"\n清洗后数据量: {cleaned_count:,} 条")
        logger.info(f"删除记录数: {removed_count:,} 条 ({removal_rate:.2f}%)")
        logger.info("=" * 80)
        
        self.stats = {
            'original_count': original_count,
            'cleaned_count': cleaned_count,
            'removed_count': removed_count,
            'removal_rate': removal_rate
        }
        
        return df.reset_index(drop=True)
    
    def _calculate_trip_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算行程持续时间"""
        logger.info("计算 trip_duration...")
        
        df['trip_duration'] = (
            df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
        ).dt.total_seconds()
        
        # 删除负值（数据错误）
        negative_count = (df['trip_duration'] < 0).sum()
        if negative_count > 0:
            logger.warning(f"发现 {negative_count} 条 trip_duration 为负的记录，已删除")
            df = df[df['trip_duration'] >= 0]
        
        return df
    
    def _filter_trip_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤行程时间异常值"""
        min_duration = self.config.get('min_trip_duration', 10)
        max_duration = self.config.get('max_trip_duration', 10800)
        
        before_count = len(df)
        
        # 应用过滤
        mask = (df['trip_duration'] >= min_duration) & (df['trip_duration'] <= max_duration)
        df_filtered = df[mask]
        
        removed = before_count - len(df_filtered)
        logger.info(f"trip_duration过滤: 删除 {removed:,} 条记录 "
                   f"(范围: {min_duration}-{max_duration}秒)")
        
        return df_filtered
    
    def _filter_passenger_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤乘客数量异常值"""
        min_passenger = self.config.get('min_passenger_count', 1)
        max_passenger = self.config.get('max_passenger_count', 6)
        
        before_count = len(df)
        
        # 过滤空值和异常值
        mask = (df['passenger_count'].notna()) & \
               (df['passenger_count'] >= min_passenger) & \
               (df['passenger_count'] <= max_passenger)
        df_filtered = df[mask]
        
        removed = before_count - len(df_filtered)
        logger.info(f"passenger_count过滤: 删除 {removed:,} 条记录 "
                   f"(范围: {min_passenger}-{max_passenger})")
        
        return df_filtered
    
    def _filter_location_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤位置异常值（如果有经纬度字段）"""
        nyc_bounds = self.config.get('nyc_bounds', None)
        
        if nyc_bounds is None:
            logger.info("未配置NYC地理范围，跳过位置过滤")
            return df
        
        # 检查是否有经纬度字段
        if 'pickup_longitude' not in df.columns or 'pickup_latitude' not in df.columns:
            logger.info("未找到经纬度字段，跳过位置过滤")
            return df
        
        before_count = len(df)
        
        mask = (
            (df['pickup_longitude'] >= nyc_bounds['min_lon']) &
            (df['pickup_longitude'] <= nyc_bounds['max_lon']) &
            (df['pickup_latitude'] >= nyc_bounds['min_lat']) &
            (df['pickup_latitude'] <= nyc_bounds['max_lat']) &
            (df['dropoff_longitude'] >= nyc_bounds['min_lon']) &
            (df['dropoff_longitude'] <= nyc_bounds['max_lon']) &
            (df['dropoff_latitude'] >= nyc_bounds['min_lat']) &
            (df['dropoff_latitude'] <= nyc_bounds['max_lat'])
        )
        
        df_filtered = df[mask]
        removed = before_count - len(df_filtered)
        logger.info(f"位置过滤: 删除 {removed:,} 条超出NYC范围的记录")
        
        return df_filtered
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化数据类型以减少内存占用"""
        logger.info("优化数据类型...")
        
        original_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
        
        # 转换数值类型
        type_conversions = {
            'VendorID': 'int32',
            'PULocationID': 'int32',
            'DOLocationID': 'int32',
            'RatecodeID': 'float32',
            'passenger_count': 'float32',
            'trip_distance': 'float32',
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
            'trip_duration': 'float32',
        }
        
        for col, target_type in type_conversions.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(target_type)
                except Exception as e:
                    logger.warning(f"无法转换 {col} 到 {target_type}: {e}")
        
        # 转换类别特征
        if 'store_and_fwd_flag' in df.columns:
            df['store_and_fwd_flag'] = df['store_and_fwd_flag'].astype('category')
        
        if 'payment_type' in df.columns:
            df['payment_type'] = df['payment_type'].astype('category')
        
        new_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
        memory_saved = original_memory - new_memory
        memory_saved_pct = (memory_saved / original_memory) * 100
        
        logger.info(f"内存优化: {original_memory:.2f} MB → {new_memory:.2f} MB "
                   f"(节省 {memory_saved:.2f} MB, {memory_saved_pct:.1f}%)")
        
        return df
    
    def get_cleaning_stats(self) -> dict:
        """获取清洗统计信息"""
        return self.stats


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import EXPERIMENT_CONFIG
    
    # 测试数据清洗
    import pyarrow.parquet as pq
    
    print("加载测试数据...")
    df = pd.read_parquet("data/raw/yellow_tripdata_2025-01.parquet")
    
    cleaner = DataCleaner(EXPERIMENT_CONFIG)
    cleaned_df = cleaner.clean(df)
    
    print("\n清洗统计:")
    for key, value in cleaner.get_cleaning_stats().items():
        print(f"{key}: {value}")
