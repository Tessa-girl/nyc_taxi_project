"""
数据探索分析模块 - EDA (Exploratory Data Analysis)
包含可视化功能
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class DataExplorer:
    """数据探索分析类"""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "results/figures"):
        """
        初始化数据探索器
        
        Args:
            df: 数据DataFrame
            output_dir: 图表输出目录
        """
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算trip_duration（如果不存在）
        if 'trip_duration' not in self.df.columns:
            self._calculate_trip_duration()
    
    def _calculate_trip_duration(self):
        """计算行程持续时间（秒）"""
        if 'tpep_pickup_datetime' in self.df.columns and 'tpep_dropoff_datetime' in self.df.columns:
            self.df['trip_duration'] = (
                self.df['tpep_dropoff_datetime'] - self.df['tpep_pickup_datetime']
            ).dt.total_seconds()
            logger.info("已计算 trip_duration 字段")
    
    def plot_trip_duration_distribution(self, save_path: Optional[str] = None):
        """
        绘制行程时间分布图
        
        Args:
            save_path: 保存路径，默认为None时自动生成
        """
        logger.info("正在绘制行程时间分布图...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 原始数据分布（过滤异常值）
        filtered_df = self.df[
            (self.df['trip_duration'] >= 60) & 
            (self.df['trip_duration'] <= 7200)
        ]
        
        # 左图：直方图
        axes[0].hist(filtered_df['trip_duration'].values / 60, bins=50, 
                     color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('行程时间 (分钟)', fontsize=12)
        axes[0].set_ylabel('频次', fontsize=12)
        axes[0].set_title('行程时间分布', fontsize=14, fontweight='bold')
        axes[0].axvline(filtered_df['trip_duration'].mean() / 60, color='red', 
                       linestyle='--', linewidth=2, label=f"均值: {filtered_df['trip_duration'].mean()/60:.1f}分钟")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # 右图：箱线图
        duration_minutes = filtered_df['trip_duration'].values / 60
        bp = axes[1].boxplot(duration_minutes, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('行程时间 (分钟)', fontsize=12)
        axes[1].set_title('行程时间箱线图', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "trip_duration_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"行程时间分布图已保存至: {save_path}")
        plt.close()
    
    def plot_pickup_hour_distribution(self, save_path: Optional[str] = None):
        """
        绘制上车时间小时分布图
        
        Args:
            save_path: 保存路径
        """
        logger.info("正在绘制上车时间小时分布图...")
        
        if 'tpep_pickup_datetime' not in self.df.columns:
            logger.warning("未找到 tpep_pickup_datetime 字段，跳过此图表")
            return
        
        # 提取小时
        self.df['pickup_hour'] = self.df['tpep_pickup_datetime'].dt.hour
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        hour_counts = self.df['pickup_hour'].value_counts().sort_index()
        
        # 绘制柱状图和折线图组合
        bars = ax.bar(hour_counts.index, hour_counts.values, 
                     color='coral', alpha=0.6, edgecolor='black', label='订单数量')
        
        # 添加趋势线
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(hour_counts.values.astype(float), sigma=2)
        ax.plot(hour_counts.index, smoothed, color='darkred', linewidth=2.5, 
               marker='o', markersize=6, label='平滑趋势')
        
        ax.set_xlabel('小时', fontsize=12)
        ax.set_ylabel('订单数量', fontsize=12)
        ax.set_title('上车时间小时分布', fontsize=14, fontweight='bold')
        ax.set_xticks(range(24))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 标注高峰时段
        peak_hour = hour_counts.idxmax()
        peak_count = hour_counts.max()
        ax.annotate(f'高峰时段: {peak_hour}:00\n订单数: {peak_count:,}',
                   xy=(peak_hour, peak_count), xytext=(peak_hour+2, peak_count*0.9),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=10, color='red', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "pickup_hour_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"上车时间小时分布图已保存至: {save_path}")
        plt.close()
    
    def plot_location_heatmap(self, save_path: Optional[str] = None):
        """
        绘制经纬度热力图（基于LocationID的替代方案）
        由于当前数据没有经纬度，使用LocationID进行可视化
        
        Args:
            save_path: 保存路径
        """
        logger.info("正在绘制位置热力图...")
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # 上车地点ID分布
        pickup_counts = self.df['PULocationID'].value_counts().head(50)
        
        colors1 = plt.cm.viridis(np.linspace(0, 1, len(pickup_counts)))
        axes[0].barh(range(len(pickup_counts)), pickup_counts.values, 
                    color=colors1, edgecolor='gray')
        axes[0].set_yticks(range(len(pickup_counts)))
        axes[0].set_yticklabels([f"Zone {int(x)}" for x in pickup_counts.index], fontsize=8)
        axes[0].set_xlabel('订单数量', fontsize=12)
        axes[0].set_title('Top 50 上车地点', fontsize=14, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # 下车地点ID分布
        dropoff_counts = self.df['DOLocationID'].value_counts().head(50)
        
        colors2 = plt.cm.plasma(np.linspace(0, 1, len(dropoff_counts)))
        axes[1].barh(range(len(dropoff_counts)), dropoff_counts.values, 
                    color=colors2, edgecolor='gray')
        axes[1].set_yticks(range(len(dropoff_counts)))
        axes[1].set_yticklabels([f"Zone {int(x)}" for x in dropoff_counts.index], fontsize=8)
        axes[1].set_xlabel('订单数量', fontsize=12)
        axes[1].set_title('Top 50 下车地点', fontsize=14, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "location_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"位置热力图已保存至: {save_path}")
        plt.close()
    
    def plot_passenger_count_distribution(self, save_path: Optional[str] = None):
        """
        绘制不同乘客数订单数量分布图
        
        Args:
            save_path: 保存路径
        """
        logger.info("正在绘制乘客数分布图...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 过滤有效乘客数
        valid_df = self.df[self.df['passenger_count'].notna()]
        passenger_counts = valid_df['passenger_count'].value_counts().sort_index()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(passenger_counts)))
        bars = ax.bar(passenger_counts.index.astype(int).astype(str), 
                     passenger_counts.values, 
                     color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # 在柱子上标注数值
        for bar, count in zip(bars, passenger_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('乘客数量', fontsize=12)
        ax.set_ylabel('订单数量', fontsize=12)
        ax.set_title('不同乘客数订单数量分布', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "passenger_count_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"乘客数分布图已保存至: {save_path}")
        plt.close()
    
    def plot_peak_hours_trend(self, save_path: Optional[str] = None):
        """
        绘制高峰期订单趋势图
        
        Args:
            save_path: 保存路径
        """
        logger.info("正在绘制高峰期订单趋势图...")
        
        if 'tpep_pickup_datetime' not in self.df.columns:
            logger.warning("未找到 tpep_pickup_datetime 字段，跳过此图表")
            return
        
        # 提取小时和星期
        self.df['pickup_hour'] = self.df['tpep_pickup_datetime'].dt.hour
        self.df['pickup_weekday'] = self.df['tpep_pickup_datetime'].dt.dayofweek
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # 左图：按小时的订单趋势
        hourly_counts = self.df['pickup_hour'].value_counts().sort_index()
        
        # 定义高峰期颜色
        colors = []
        for hour in hourly_counts.index:
            if (7 <= hour <= 9) or (17 <= hour <= 19):  # 早晚高峰
                colors.append('#FF6B6B')  # 红色
            else:
                colors.append('#4ECDC4')  # 青色
        
        axes[0].bar(hourly_counts.index, hourly_counts.values, color=colors, 
                   edgecolor='black', alpha=0.7, linewidth=1.2)
        axes[0].set_xlabel('小时', fontsize=12)
        axes[0].set_ylabel('订单数量', fontsize=12)
        axes[0].set_title('每小时订单趋势（红色=高峰期）', fontsize=14, fontweight='bold')
        axes[0].set_xticks(range(24))
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', alpha=0.7, label='高峰期 (7-9点, 17-19点)'),
            Patch(facecolor='#4ECDC4', alpha=0.7, label='非高峰期')
        ]
        axes[0].legend(handles=legend_elements, fontsize=10)
        
        # 右图：工作日vs周末的小时对比
        weekday_data = self.df[self.df['pickup_weekday'] < 5]  # 周一到周五
        weekend_data = self.df[self.df['pickup_weekday'] >= 5]  # 周六和周日
        
        weekday_hourly = weekday_data['pickup_hour'].value_counts().sort_index()
        weekend_hourly = weekend_data['pickup_hour'].value_counts().sort_index()
        
        axes[1].plot(weekday_hourly.index, weekday_hourly.values, 'o-', 
                    linewidth=2.5, markersize=8, label='工作日', color='#2E86AB')
        axes[1].plot(weekend_hourly.index, weekend_hourly.values, 's-', 
                    linewidth=2.5, markersize=8, label='周末', color='#A23B72')
        axes[1].set_xlabel('小时', fontsize=12)
        axes[1].set_ylabel('订单数量', fontsize=12)
        axes[1].set_title('工作日 vs 周末订单对比', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(24))
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "peak_hours_trend.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"高峰期订单趋势图已保存至: {save_path}")
        plt.close()
    
    def generate_eda_report(self):
        """
        生成完整的EDA报告
        """
        logger.info("=" * 80)
        logger.info("开始生成数据探索分析报告")
        logger.info("=" * 80)
        
        # 基本统计信息
        logger.info(f"\n数据集基本信息:")
        logger.info(f"总记录数: {len(self.df):,}")
        logger.info(f"总列数: {len(self.df.columns)}")
        
        logger.info(f"\n数值型字段统计:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            logger.info(f"{col}: 均值={self.df[col].mean():.2f}, "
                       f"中位数={self.df[col].median():.2f}, "
                       f"标准差={self.df[col].std():.2f}")
        
        # 生成所有可视化图表
        logger.info(f"\n正在生成可视化图表...")
        self.plot_trip_duration_distribution()
        self.plot_pickup_hour_distribution()
        self.plot_location_heatmap()
        self.plot_passenger_count_distribution()
        self.plot_peak_hours_trend()
        
        logger.info("=" * 80)
        logger.info("EDA报告生成完成！")
        logger.info("=" * 80)


if __name__ == "__main__":
    import pyarrow.parquet as pq
    
    # 读取数据
    print("正在加载数据...")
    df = pd.read_parquet("data/raw/yellow_tripdata_2025-01.parquet")
    
    # 创建数据探索器
    explorer = DataExplorer(df.head(100000))  # 使用样本数据加速
    
    # 生成EDA报告
    explorer.generate_eda_report()
    
    print("\n✅ 数据探索分析完成！")
