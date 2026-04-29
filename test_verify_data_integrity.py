"""
数据完整性验证功能的单元测试
"""
import unittest
import tempfile
import os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime


class TestDataIntegrityVerification(unittest.TestCase):
    """测试数据完整性验证功能"""
    
    def setUp(self):
        """创建临时目录和测试数据"""
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.temp_dir) / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试数据
        self.create_test_parquet_files()
    
    def tearDown(self):
        """清理临时目录"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_parquet_files(self):
        """创建测试用的 Parquet 文件"""
        # 测试数据 1: 2025-01
        data_01 = {
            'VendorID': [1, 2, 1],
            'tpep_pickup_datetime': [datetime(2025, 1, 1, 10, 0), 
                                     datetime(2025, 1, 1, 11, 0),
                                     datetime(2025, 1, 1, 12, 0)],
            'tpep_dropoff_datetime': [datetime(2025, 1, 1, 10, 30), 
                                      datetime(2025, 1, 1, 11, 45),
                                      datetime(2025, 1, 1, 12, 30)],
            'passenger_count': [1, 2, None],
            'trip_distance': [2.5, 3.8, 1.2],
            'fare_amount': [10.5, 15.0, 8.0],
            'cbd_congestion_fee': [2.5, 2.5, 2.5]
        }
        df_01 = pd.DataFrame(data_01)
        table_01 = pa.Table.from_pandas(df_01)
        pq.write_table(table_01, self.raw_dir / "yellow_tripdata_2025-01.parquet")
        
        # 测试数据 2: 2025-02 (结构相同)
        data_02 = {
            'VendorID': [1, 2],
            'tpep_pickup_datetime': [datetime(2025, 2, 1, 10, 0), 
                                     datetime(2025, 2, 1, 11, 0)],
            'tpep_dropoff_datetime': [datetime(2025, 2, 1, 10, 30), 
                                      datetime(2025, 2, 1, 11, 45)],
            'passenger_count': [None, 3],
            'trip_distance': [2.0, 4.5],
            'fare_amount': [9.5, 17.0],
            'cbd_congestion_fee': [2.5, 2.5]
        }
        df_02 = pd.DataFrame(data_02)
        table_02 = pa.Table.from_pandas(df_02)
        pq.write_table(table_02, self.raw_dir / "yellow_tripdata_2025-02.parquet")
    
    def test_parquet_file_exists(self):
        """测试 Parquet 文件是否存在"""
        parquet_files = list(self.raw_dir.glob("*.parquet"))
        self.assertEqual(len(parquet_files), 2, "应该找到2个测试文件")
    
    def test_parquet_file_readable(self):
        """测试 Parquet 文件是否可读"""
        file_path = self.raw_dir / "yellow_tripdata_2025-01.parquet"
        parquet_file = pq.ParquetFile(file_path)
        self.assertIsNotNone(parquet_file.schema_arrow)
    
    def test_row_count(self):
        """测试行数是否正确"""
        file_path = self.raw_dir / "yellow_tripdata_2025-01.parquet"
        parquet_file = pq.ParquetFile(file_path)
        self.assertEqual(parquet_file.metadata.num_rows, 3)
    
    def test_column_count(self):
        """测试列数是否正确"""
        file_path = self.raw_dir / "yellow_tripdata_2025-01.parquet"
        parquet_file = pq.ParquetFile(file_path)
        self.assertEqual(parquet_file.metadata.num_columns, 7)
    
    def test_column_names(self):
        """测试列名是否正确"""
        file_path = self.raw_dir / "yellow_tripdata_2025-01.parquet"
        parquet_file = pq.ParquetFile(file_path)
        schema = parquet_file.schema_arrow
        column_names = [field.name for field in schema]
        
        expected_columns = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                           'passenger_count', 'trip_distance', 'fare_amount', 'cbd_congestion_fee']
        self.assertEqual(column_names, expected_columns)
    
    def test_null_values_detection(self):
        """测试是否能检测到空值"""
        file_path = self.raw_dir / "yellow_tripdata_2025-01.parquet"
        table = pq.read_table(file_path, columns=['passenger_count'])
        null_count = table.column('passenger_count').null_count
        
        self.assertGreater(null_count, 0, "应该检测到空值")
        self.assertEqual(null_count, 1)
    
    def test_file_size_validation(self):
        """测试文件大小验证"""
        file_path = self.raw_dir / "yellow_tripdata_2025-01.parquet"
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        self.assertGreater(file_size_mb, 0, "文件大小应该大于0")
        self.assertLess(file_size_mb, 1, "测试文件应该小于1MB")
    
    def test_consistent_schema_across_files(self):
        """测试多个文件的模式一致性"""
        files = sorted(self.raw_dir.glob("*.parquet"))
        
        schemas = []
        for file_path in files:
            parquet_file = pq.ParquetFile(file_path)
            schemas.append(parquet_file.schema_arrow)
        
        # 比较所有文件的列名
        column_sets = [set(field.name for field in schema) for schema in schemas]
        self.assertEqual(column_sets[0], column_sets[1], "两个文件的列名应该一致")
    
    def test_no_new_columns_in_second_file(self):
        """测试第二个文件没有新增列"""
        file_01 = self.raw_dir / "yellow_tripdata_2025-01.parquet"
        file_02 = self.raw_dir / "yellow_tripdata_2025-02.parquet"
        
        schema_01 = pq.ParquetFile(file_01).schema_arrow
        schema_02 = pq.ParquetFile(file_02).schema_arrow
        
        cols_01 = set(field.name for field in schema_01)
        cols_02 = set(field.name for field in schema_02)
        
        new_columns = cols_02 - cols_01
        self.assertEqual(len(new_columns), 0, "第二个文件不应该有新增列")
    
    def test_data_types_consistency(self):
        """测试数据类型一致性"""
        file_path = self.raw_dir / "yellow_tripdata_2025-01.parquet"
        parquet_file = pq.ParquetFile(file_path)
        schema = parquet_file.schema_arrow
        
        # 检查 VendorID 是 int 类型
        vendor_type = str(schema.field('VendorID').type)
        self.assertIn('int', vendor_type.lower())
        
        # 检查 trip_distance 是 double/float 类型
        distance_type = str(schema.field('trip_distance').type)
        self.assertIn('double', distance_type.lower())


if __name__ == '__main__':
    unittest.main()
