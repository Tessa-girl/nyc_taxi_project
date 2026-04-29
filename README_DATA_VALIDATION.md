# 数据验证工具使用说明

## 📋 概述

本项目包含数据完整性验证工具和单元测试，用于检查 NYC 出租车数据的完整性和列信息。

## 🛠️ 文件说明

### 主要脚本
- **verify_data_integrity.py**: 主脚本，验证所有 Parquet 文件的完整性并输出详细报告
- **test_verify_data_integrity.py**: 单元测试文件，验证功能的正确性
- **DATA_INTEGRITY_REPORT.md**: 详细的验证报告文档

## 🚀 运行方式

### 1. 运行数据验证
```bash
python verify_data_integrity.py
```

**输出内容**:
- 每个文件的大小、行数、列数
- 详细的列信息（名称、类型、空值数）
- 包含缺失值的列统计
- 跨文件列结构比较
- 新字段检测结果

### 2. 运行单元测试
```bash
python -m pytest test_verify_data_integrity.py -v
```

**测试覆盖**:
- ✅ 文件存在性
- ✅ 文件可读性
- ✅ 行列数验证
- ✅ 列名验证
- ✅ 空值检测
- ✅ 文件大小验证
- ✅ 跨文件模式一致性
- ✅ 数据类型一致性

## 📊 验证结果摘要

### 文件状态
| 文件 | 状态 | 行数 | 列数 |
|------|------|------|------|
| yellow_tripdata_2025-01.parquet | ✅ 完整 | 3,475,226 | 20 |
| yellow_tripdata_2025-02.parquet | ✅ 完整 | 3,577,543 | 20 |
| yellow_tripdata_2025-03.parquet | ✅ 完整 | 4,145,257 | 20 |

### 关键发现
- ✅ 所有文件完整性验证通过
- ✅ 三个文件的列结构完全一致
- ❌ 未发现新增字段
- ⚠️ 存在部分缺失值（约15-23%的记录）

### 重要字段
- **cbd_congestion_fee**: CBD拥堵费（新特征，无空值）
- **congestion_surcharge**: 通用拥堵附加费（有空值）
- **Airport_fee**: 机场费用（有空值）

## 🔍 如何添加新文件的验证

当有新的数据文件时：

1. 将文件放入 `data/raw/` 目录
2. 运行验证脚本：
   ```bash
   python verify_data_integrity.py
   ```
3. 查看输出，特别关注"跨文件列信息比较"部分
4. 如有新字段，会在报告中明确标注

## 📝 依赖项

```bash
pip install pyarrow pandas
```

## 💡 提示

- 验证脚本会自动检测所有 `.parquet` 文件
- 以第一个文件为基准，比较后续文件的列结构
- 新增字段会用 🆕 标记
- 缺失字段会用 ⚠️ 标记
