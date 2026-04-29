"""
特征工程模块
构建时间、空间、类别和交叉特征
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self):
        """初始化特征工程师"""
        self.feature_columns = []
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整的特征工程流程
        
        Args:
            df: 清洗后的数据DataFrame
            
        Returns:
            添加特征后的DataFrame
        """
        logger.info("=" * 80)
        logger.info("开始特征工程")
        logger.info("=" * 80)
        
        original_cols = set(df.columns)
        
        # 1. 时间特征
        df = self._extract_time_features(df)
        
        # 2. 空间特征（如果有经纬度）或使用LocationID
        df = self._extract_spatial_features(df)
        
        # 3. 费用相关特征
        df = self._extract_fare_features(df)
        
        # 4. 交叉特征
        df = self._create_interaction_features(df)
        
        new_cols = set(df.columns) - original_cols
        logger.info(f"新增特征数量: {len(new_cols)} 个")
        logger.info(f"新增特征: {', '.join(sorted(new_cols))}")
        logger.info("=" * 80)
        
        return df
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取时间特征"""
        logger.info("提取时间特征...")
        
        if 'tpep_pickup_datetime' in df.columns:
            pickup = df['tpep_pickup_datetime']
            
            # 基础时间特征
            df['pickup_hour'] = pickup.dt.hour
            df['pickup_minute'] = pickup.dt.minute
            df['pickup_weekday'] = pickup.dt.dayofweek  # 0=周一, 6=周日
            df['pickup_day'] = pickup.dt.day
            df['pickup_month'] = pickup.dt.month
            df['pickup_year'] = pickup.dt.year
            
            # 二元特征
            df['is_weekend'] = (df['pickup_weekday'] >= 5).astype(int)
            df['is_night'] = ((df['pickup_hour'] < 6) | (df['pickup_hour'] >= 22)).astype(int)
            
            # 高峰期特征
            morning_peak = (df['pickup_hour'] >= 7) & (df['pickup_hour'] <= 9)
            evening_peak = (df['pickup_hour'] >= 17) & (df['pickup_hour'] <= 19)
            df['is_rush_hour'] = (morning_peak | evening_peak).astype(int)
            
            # 时段划分
            def categorize_hour(hour):
                if 0 <= hour < 6:
                    return 0  # 凌晨
                elif 6 <= hour < 10:
                    return 1  # 早高峰
                elif 10 <= hour < 16:
                    return 2  # 白天
                elif 16 <= hour < 20:
                    return 3  # 晚高峰
                else:
                    return 4  # 晚上
            
            df['pickup_period'] = df['pickup_hour'].apply(categorize_hour)
            
            logger.info("已提取时间特征")
        
        if 'tpep_dropoff_datetime' in df.columns:
            dropoff = df['tpep_dropoff_datetime']
            df['dropoff_hour'] = dropoff.dt.hour
            df['dropoff_weekday'] = dropoff.dt.dayofweek
        
        return df
    
    def _extract_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取空间特征"""
        logger.info("提取空间特征...")
        
        # 如果有经纬度字段，计算精确距离
        if all(col in df.columns for col in ['pickup_longitude', 'pickup_latitude', 
                                             'dropoff_longitude', 'dropoff_latitude']):
            # Haversine距离
            df['haversine_distance'] = self._calculate_haversine(
                df['pickup_latitude'], df['pickup_longitude'],
                df['dropoff_latitude'], df['dropoff_longitude']
            )
            
            # 曼哈顿距离
            df['manhattan_distance'] = (
                abs(df['dropoff_longitude'] - df['pickup_longitude']) +
                abs(df['dropoff_latitude'] - df['pickup_latitude'])
            )
            
            # 经纬度差值
            df['lon_diff'] = df['dropoff_longitude'] - df['pickup_longitude']
            df['lat_diff'] = df['dropoff_latitude'] - df['pickup_latitude']
            
            logger.info("已基于经纬度计算距离特征")
        else:
            # 使用LocationID作为替代
            logger.info("未找到经纬度字段，使用LocationID构建特征...")
            
            # Location ID差异
            if 'PULocationID' in df.columns and 'DOLocationID' in df.columns:
                df['location_id_diff'] = df['DOLocationID'] - df['PULocationID']
                
                # 是否同一区域
                df['is_same_zone'] = (df['PULocationID'] == df['DOLocationID']).astype(int)
                
                # Top热门区域标记
                top_pickup_zones = df['PULocationID'].value_counts().head(20).index
                top_dropoff_zones = df['DOLocationID'].value_counts().head(20).index
                
                df['is_top_pickup_zone'] = df['PULocationID'].isin(top_pickup_zones).astype(int)
                df['is_top_dropoff_zone'] = df['DOLocationID'].isin(top_dropoff_zones).astype(int)
        
        return df
    
    def _calculate_haversine(self, lat1, lon1, lat2, lon2):
        """
        计算Haversine距离（公里）
        
        Args:
            lat1, lon1: 起点经纬度
            lat2, lon2: 终点经纬度
            
        Returns:
            距离（公里）
        """
        R = 6371  # 地球半径（公里）
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _extract_fare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取费用相关特征"""
        logger.info("提取费用特征...")
        
        # 费率相关
        if 'fare_amount' in df.columns and 'trip_distance' in df.columns:
            # 每公里费用（排除距离为0的情况）
            df['fare_per_km'] = np.where(
                df['trip_distance'] > 0,
                df['fare_amount'] / df['trip_distance'],
                0
            )
        
        # 小费比例
        if 'tip_amount' in df.columns and 'fare_amount' in df.columns:
            df['tip_ratio'] = np.where(
                df['fare_amount'] > 0,
                df['tip_amount'] / df['fare_amount'],
                0
            )
        
        # 额外费用比例
        if 'extra' in df.columns and 'total_amount' in df.columns:
            df['extra_ratio'] = np.where(
                df['total_amount'] > 0,
                df['extra'] / df['total_amount'],
                0
            )
        
        # 是否有拥堵费
        if 'congestion_surcharge' in df.columns:
            df['has_congestion_fee'] = (df['congestion_surcharge'] > 0).astype(int)
        
        if 'cbd_congestion_fee' in df.columns:
            df['has_cbd_fee'] = (df['cbd_congestion_fee'] > 0).astype(int)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交叉特征"""
        logger.info("创建交叉特征...")
        
        # 小时 × 乘客数
        if 'pickup_hour' in df.columns and 'passenger_count' in df.columns:
            df['hour_passenger'] = df['pickup_hour'] * df['passenger_count']
        
        # 工作日 × 高峰期
        if 'is_weekend' in df.columns and 'is_rush_hour' in df.columns:
            df['weekend_rush'] = df['is_weekend'] * df['is_rush_hour']
        
        # 上车地点 × 时间段
        if 'PULocationID' in df.columns and 'pickup_period' in df.columns:
            df['zone_period'] = df['PULocationID'].astype(str) + '_' + df['pickup_period'].astype(str)
        
        # 乘客数 × 距离
        if 'passenger_count' in df.columns and 'trip_distance' in df.columns:
            df['passenger_distance_ratio'] = np.where(
                df['trip_distance'] > 0,
                df['passenger_count'] / df['trip_distance'],
                0
            )
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """获取所有特征列名"""
        return self.feature_columns


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    # 测试特征工程
    import pyarrow.parquet as pq
    from data_cleaning import DataCleaner
    from config import EXPERIMENT_CONFIG
    
    print("加载数据...")
    df = pd.read_parquet("data/raw/yellow_tripdata_2025-01.parquet")
    
    print("数据清洗...")
    cleaner = DataCleaner(EXPERIMENT_CONFIG)
    cleaned_df = cleaner.clean(df.head(50000))
    
    print("特征工程...")
    fe = FeatureEngineer()
    feature_df = fe.transform(cleaned_df)
    
    print("\n特征工程完成！")
    print(f"最终列数: {len(feature_df.columns)}")
