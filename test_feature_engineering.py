"""
特征工程模块的单元测试
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))
from feature_engineering import FeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    """测试特征工程师"""
    
    def setUp(self):
        """准备测试数据"""
        base_time = datetime(2025, 1, 15, 10, 30)  # 周三
        
        self.test_data = pd.DataFrame({
            'tpep_pickup_datetime': [
                base_time,
                base_time.replace(hour=8),  # 早高峰
                base_time.replace(hour=18),  # 晚高峰
                base_time.replace(hour=3),   # 凌晨
            ],
            'tpep_dropoff_datetime': [
                base_time.replace(minute=45),
                base_time.replace(hour=9),
                base_time.replace(hour=19),
                base_time.replace(hour=3, minute=30),
            ],
            'passenger_count': [1, 2, 3, 1],
            'trip_distance': [2.5, 3.8, 5.0, 1.2],
            'PULocationID': [100, 200, 150, 250],
            'DOLocationID': [200, 300, 250, 100],
            'fare_amount': [10.5, 15.0, 20.0, 8.0],
            'extra': [0.5, 1.0, 1.5, 0.0],
            'total_amount': [15.0, 20.0, 25.0, 10.0],
            'tip_amount': [2.0, 3.0, 4.0, 1.0],
            'congestion_surcharge': [2.5, 2.5, 0.0, 2.5],
            'cbd_congestion_fee': [0.0, 0.0, 0.0, 0.0],
        })
        
        self.fe = FeatureEngineer()
    
    def test_extract_time_features(self):
        """测试时间特征提取"""
        df = self.fe._extract_time_features(self.test_data.copy())
        
        # 检查基础时间特征
        self.assertIn('pickup_hour', df.columns)
        self.assertIn('pickup_weekday', df.columns)
        self.assertIn('is_weekend', df.columns)
        self.assertIn('is_rush_hour', df.columns)
        
        # 检查早高峰识别
        self.assertEqual(df['is_rush_hour'].iloc[1], 1)
        
        # 检查晚高峰识别
        self.assertEqual(df['is_rush_hour'].iloc[2], 1)
        
        # 检查凌晨时段
        self.assertEqual(df['is_night'].iloc[3], 1)
    
    def test_extract_spatial_features_without_lat_lon(self):
        """测试无经纬度时的空间特征"""
        df = self.fe._extract_spatial_features(self.test_data.copy())
        
        # 应该使用LocationID
        self.assertIn('location_id_diff', df.columns)
        self.assertIn('is_same_zone', df.columns)
    
    def test_extract_fare_features(self):
        """测试费用特征提取"""
        df = self.fe._extract_fare_features(self.test_data.copy())
        
        # 检查费用特征
        self.assertIn('fare_per_km', df.columns)
        self.assertIn('tip_ratio', df.columns)
        self.assertIn('has_congestion_fee', df.columns)
        
        # 检查小费比例计算
        self.assertGreater(df['tip_ratio'].iloc[0], 0)
    
    def test_create_interaction_features(self):
        """测试交叉特征"""
        # 先添加时间特征
        df = self.fe._extract_time_features(self.test_data.copy())
        df = self.fe._create_interaction_features(df)
        
        # 检查交叉特征
        self.assertIn('hour_passenger', df.columns)
        self.assertIn('weekend_rush', df.columns)
    
    def test_full_transformation(self):
        """测试完整特征工程流程"""
        df_transformed = self.fe.transform(self.test_data.copy())
        
        # 检查是否添加了新特征
        original_cols = set(self.test_data.columns)
        new_cols = set(df_transformed.columns) - original_cols
        
        self.assertGreater(len(new_cols), 0)
        
        # 检查DataFrame形状
        self.assertEqual(len(df_transformed), len(self.test_data))
    
    def test_haversine_calculation(self):
        """测试Haversine距离计算"""
        # 模拟纽约市内的两个点
        lat1 = pd.Series([40.7128])
        lon1 = pd.Series([-74.0060])
        lat2 = pd.Series([40.7580])
        lon2 = pd.Series([-73.9855])
        
        distance = self.fe._calculate_haversine(lat1, lon1, lat2, lon2)
        
        # 曼哈顿到中城约5-6公里
        self.assertAlmostEqual(distance.iloc[0], 5.5, delta=1.0)


if __name__ == '__main__':
    unittest.main()
