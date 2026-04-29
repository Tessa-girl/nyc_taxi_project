"""
数据清洗模块的单元测试
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))
from data_cleaning import DataCleaner


class TestDataCleaner(unittest.TestCase):
    """测试数据清洗器"""
    
    def setUp(self):
        """准备测试数据"""
        base_time = datetime(2025, 1, 1, 10, 0)
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'tpep_pickup_datetime': [
                base_time,
                base_time + timedelta(hours=1),
                base_time + timedelta(hours=2),
                base_time + timedelta(hours=3),
                base_time + timedelta(hours=4),
            ],
            'tpep_dropoff_datetime': [
                base_time + timedelta(minutes=15),
                base_time + timedelta(hours=1, minutes=20),
                base_time + timedelta(hours=2, minutes=10),
                base_time + timedelta(hours=3, minutes=30),
                base_time + timedelta(hours=4, minutes=5),
            ],
            'passenger_count': [1, 2, 0, 3, 8],  # 包含0和异常值8
            'trip_distance': [2.5, 3.8, 1.2, 5.0, 2.0],
            'VendorID': [1, 2, 1, 2, 1],
            'PULocationID': [100, 200, 150, 250, 300],
            'DOLocationID': [200, 300, 250, 100, 150],
            'fare_amount': [10.5, 15.0, 8.0, 20.0, 12.0],
        })
        
        self.config = {
            'min_trip_duration': 10,
            'max_trip_duration': 10800,
            'min_passenger_count': 1,
            'max_passenger_count': 6,
            'nyc_bounds': None,  # 跳过位置过滤
        }
    
    def test_calculate_trip_duration(self):
        """测试行程时间计算"""
        cleaner = DataCleaner(self.config)
        df = cleaner._calculate_trip_duration(self.test_data.copy())
        
        # 检查trip_duration列是否存在
        self.assertIn('trip_duration', df.columns)
        
        # 检查第一个记录的持续时间（15分钟=900秒）
        self.assertAlmostEqual(df['trip_duration'].iloc[0], 900, places=0)
    
    def test_filter_trip_duration(self):
        """测试行程时间过滤"""
        cleaner = DataCleaner(self.config)
        df = cleaner._calculate_trip_duration(self.test_data.copy())
        
        # 添加一个异常值（5小时）
        df.loc[len(df)-1, 'trip_duration'] = 18000
        
        filtered_df = cleaner._filter_trip_duration(df)
        
        # 应该删除异常值记录
        self.assertLess(len(filtered_df), len(df))
    
    def test_filter_passenger_count(self):
        """测试乘客数过滤"""
        cleaner = DataCleaner(self.config)
        
        # passenger_count为0和8的记录应该被删除
        filtered_df = cleaner._filter_passenger_count(self.test_data.copy())
        
        # 应该删除2条记录（0和8）
        self.assertEqual(len(filtered_df), 3)
        
        # 剩余的乘客数应该在1-6之间
        self.assertTrue(all(filtered_df['passenger_count'] >= 1))
        self.assertTrue(all(filtered_df['passenger_count'] <= 6))
    
    def test_data_type_conversion(self):
        """测试数据类型转换"""
        cleaner = DataCleaner(self.config)
        
        df = cleaner._convert_data_types(self.test_data.copy())
        
        # 检查数据类型是否优化
        self.assertIn(df['VendorID'].dtype, ['int32', 'int64'])
    
    def test_full_cleaning_pipeline(self):
        """测试完整清洗流程"""
        cleaner = DataCleaner(self.config)
        
        cleaned_df = cleaner.clean(self.test_data.copy())
        
        # 检查返回的是DataFrame
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        
        # 检查没有空索引
        self.assertTrue(cleaned_df.index.is_unique)
        
        # 获取统计信息
        stats = cleaner.get_cleaning_stats()
        self.assertIn('original_count', stats)
        self.assertIn('cleaned_count', stats)
    
    def test_empty_dataframe(self):
        """测试空DataFrame处理"""
        cleaner = DataCleaner(self.config)
        
        # 创建空的但有正确列名的DataFrame
        empty_df = pd.DataFrame({
            'tpep_pickup_datetime': pd.Series([], dtype='datetime64[ns]'),
            'tpep_dropoff_datetime': pd.Series([], dtype='datetime64[ns]'),
            'passenger_count': pd.Series([], dtype='float64'),
            'trip_distance': pd.Series([], dtype='float64'),
        })
        
        # 应该不会报错
        try:
            result = cleaner.clean(empty_df)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 0)
        except Exception as e:
            self.fail(f"处理空DataFrame时出错: {e}")


if __name__ == '__main__':
    unittest.main()
