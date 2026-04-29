"""
验证 NYC 出租车数据文件的完整性并记录列信息
"""
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import os

def verify_parquet_files(data_dir):
    """
    验证 Parquet 文件的完整性并输出列信息
    
    Args:
        data_dir: 数据目录路径
    """
    data_path = Path(data_dir)
    raw_path = data_path / "raw"
    
    print("=" * 80)
    print("NYC 出租车数据文件完整性验证")
    print("=" * 80)
    print()
    
    # 存储所有文件的列信息用于比较
    all_columns = {}
    
    # 遍历 raw 目录下的所有 parquet 文件
    parquet_files = sorted(raw_path.glob("*.parquet"))
    
    if not parquet_files:
        print("未找到 Parquet 文件!")
        return
    
    for file_path in parquet_files:
        print(f"\n{'='*80}")
        print(f"文件: {file_path.name}")
        print(f"{'='*80}")
        
        try:
            # 获取文件大小
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"文件大小: {file_size_mb:.2f} MB")
            
            # 读取 Parquet 文件元数据
            parquet_file = pq.ParquetFile(file_path)
            metadata = parquet_file.metadata
            
            # 基本统计信息
            print(f"行数: {metadata.num_rows:,}")
            print(f"列数: {metadata.num_columns}")
            print(f"行组数: {metadata.num_row_groups}")
            
            # 获取列信息
            schema = parquet_file.schema_arrow
            columns_info = []
            
            print(f"\n列详细信息:")
            print("-" * 80)
            for i, field in enumerate(schema):
                col_name = field.name
                col_type = str(field.type)
                
                # 计算该列的统计信息
                table = pq.read_table(file_path, columns=[col_name])
                column_data = table.column(col_name)
                
                null_count = column_data.null_count
                
                columns_info.append({
                    'name': col_name,
                    'type': col_type,
                    'null_count': null_count
                })
                
                print(f"{i+1:3d}. {col_name:<35} {col_type:<20} (nulls: {null_count:,})")
            
            # 存储列信息用于后续比较
            all_columns[file_path.name] = [col['name'] for col in columns_info]
            
            # 检查是否有缺失值
            cols_with_nulls = [col for col in columns_info if col['null_count'] > 0]
            if cols_with_nulls:
                print(f"\n⚠️  包含缺失值的列 ({len(cols_with_nulls)} 个):")
                for col in cols_with_nulls:
                    null_percentage = (col['null_count'] / metadata.num_rows) * 100
                    print(f"   - {col['name']}: {col['null_count']:,} ({null_percentage:.2f}%)")
            
            print(f"\n✅ 文件完整性验证通过")
            
        except Exception as e:
            print(f"\n❌ 文件验证失败: {str(e)}")
            continue
    
    # 比较不同文件的列
    print(f"\n\n{'='*80}")
    print("跨文件列信息比较")
    print("=" * 80)
    
    if len(all_columns) > 1:
        file_names = list(all_columns.keys())
        base_columns = set(all_columns[file_names[0]])
        
        print(f"\n基准文件 ({file_names[0]}): {len(base_columns)} 列")
        for col in sorted(base_columns):
            print(f"  - {col}")
        
        for file_name in file_names[1:]:
            current_columns = set(all_columns[file_name])
            new_columns = current_columns - base_columns
            missing_columns = base_columns - current_columns
            
            print(f"\n{file_name}: {len(current_columns)} 列")
            
            if new_columns:
                print(f"  🆕 新增字段 ({len(new_columns)} 个):")
                for col in sorted(new_columns):
                    print(f"    + {col}")
            
            if missing_columns:
                print(f"  ⚠️  缺失字段 ({len(missing_columns)} 个):")
                for col in sorted(missing_columns):
                    print(f"    - {col}")
            
            if not new_columns and not missing_columns:
                print(f"  ✅ 列结构与基准文件一致")
    
    # 生成完整报告
    print(f"\n\n{'='*80}")
    print("数据集列信息汇总")
    print("=" * 80)
    
    # 收集所有唯一的列名
    all_unique_columns = set()
    for cols in all_columns.values():
        all_unique_columns.update(cols)
    
    print(f"\n所有文件中出现的唯一列名总数: {len(all_unique_columns)}")
    print("\n完整列名列表:")
    for i, col in enumerate(sorted(all_unique_columns), 1):
        # 显示该列出现在哪些文件中
        files_with_col = [fname for fname, cols in all_columns.items() if col in cols]
        file_indicator = ", ".join([f.replace("yellow_tripdata_", "").replace(".parquet", "") for f in files_with_col])
        print(f"{i:3d}. {col:<40} [出现在: {file_indicator}]")


if __name__ == "__main__":
    # 设置数据目录
    data_dir = r"e:\BigData\nyc_taxi_project\data"
    
    # 验证文件
    verify_parquet_files(data_dir)
