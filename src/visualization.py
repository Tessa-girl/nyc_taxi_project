"""
结果可视化模块
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, output_dir: str = "results/figures"):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_model_comparison(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        绘制模型性能对比图
        
        Args:
            results_df: 模型评估结果DataFrame
            save_path: 保存路径
        """
        logger.info("绘制模型性能对比图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = results_df['model_name'].values
        x_pos = np.arange(len(models))
        
        # RMSE对比
        rmse_values = results_df['RMSE'].values
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars1 = axes[0, 0].barh(x_pos, rmse_values, color=colors, edgecolor='black', alpha=0.8)
        axes[0, 0].set_yticks(x_pos)
        axes[0, 0].set_yticklabels(models, fontsize=10)
        axes[0, 0].set_xlabel('RMSE', fontsize=12)
        axes[0, 0].set_title('RMSE 对比', fontsize=14, fontweight='bold')
        axes[0, 0].invert_xaxis()
        
        # 在柱子上标注数值
        for i, (bar, val) in enumerate(zip(bars1, rmse_values)):
            axes[0, 0].text(val + max(rmse_values)*0.01, i, f'{val:.2f}', 
                           va='center', fontsize=9, fontweight='bold')
        
        # MAE对比
        mae_values = results_df['MAE'].values
        bars2 = axes[0, 1].barh(x_pos, mae_values, color=colors, edgecolor='black', alpha=0.8)
        axes[0, 1].set_yticks(x_pos)
        axes[0, 1].set_yticklabels(models, fontsize=10)
        axes[0, 1].set_xlabel('MAE', fontsize=12)
        axes[0, 1].set_title('MAE 对比', fontsize=14, fontweight='bold')
        axes[0, 1].invert_xaxis()
        
        for i, (bar, val) in enumerate(zip(bars2, mae_values)):
            axes[0, 1].text(val + max(mae_values)*0.01, i, f'{val:.2f}', 
                           va='center', fontsize=9, fontweight='bold')
        
        # R2对比
        r2_values = results_df['R2'].values
        bars3 = axes[1, 0].barh(x_pos, r2_values, color=colors, edgecolor='black', alpha=0.8)
        axes[1, 0].set_yticks(x_pos)
        axes[1, 0].set_yticklabels(models, fontsize=10)
        axes[1, 0].set_xlabel('R2', fontsize=12)
        axes[1, 0].set_title('R2 对比', fontsize=14, fontweight='bold')
        
        for i, (bar, val) in enumerate(zip(bars3, r2_values)):
            axes[1, 0].text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
        
        # Log RMSE对比
        log_rmse_values = results_df['Log_RMSE'].values
        bars4 = axes[1, 1].barh(x_pos, log_rmse_values, color=colors, edgecolor='black', alpha=0.8)
        axes[1, 1].set_yticks(x_pos)
        axes[1, 1].set_yticklabels(models, fontsize=10)
        axes[1, 1].set_xlabel('Log RMSE', fontsize=12)
        axes[1, 1].set_title('Log RMSE 对比', fontsize=14, fontweight='bold')
        axes[1, 1].invert_xaxis()
        
        for i, (bar, val) in enumerate(zip(bars4, log_rmse_values)):
            axes[1, 1].text(val + max(log_rmse_values)*0.01, i, f'{val:.4f}', 
                           va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"模型对比图已保存至: {save_path}")
        plt.close()
    
    def plot_training_time_comparison(self, timings: Dict[str, float], 
                                     save_path: Optional[str] = None):
        """
        绘制训练时间对比图
        
        Args:
            timings: 训练时间字典
            save_path: 保存路径
        """
        logger.info("绘制训练时间对比图...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(timings.keys())
        times = list(timings.values())
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(models)))
        bars = ax.bar(range(len(models)), times, color=colors, edgecolor='black', 
                     alpha=0.8, linewidth=1.5)
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=11, rotation=15, ha='right')
        ax.set_ylabel('训练时间 (秒)', fontsize=12)
        ax.set_title('模型训练时间对比', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 标注时间
        for bar, t in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{t:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "training_time_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练时间对比图已保存至: {save_path}")
        plt.close()
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importance_scores: np.ndarray,
                               model_name: str = "Model",
                               top_n: int = 20,
                               save_path: Optional[str] = None):
        """
        绘制特征重要性图
        
        Args:
            feature_names: 特征名称列表
            importance_scores: 重要性分数数组
            model_name: 模型名称
            top_n: 显示前N个特征
            save_path: 保存路径
        """
        logger.info(f"绘制 {model_name} 特征重要性图...")
        
        # 创建特征重要性DataFrame
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        
        # 按重要性排序
        feat_imp = feat_imp.sort_values('importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.inferno(np.linspace(0, 1, len(feat_imp)))
        bars = ax.barh(range(len(feat_imp)), feat_imp['importance'].values,
                      color=colors, edgecolor='black', alpha=0.8)
        
        ax.set_yticks(range(len(feat_imp)))
        ax.set_yticklabels(feat_imp['feature'].values, fontsize=10)
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_title(f'{model_name} - Top {top_n} 特征重要性', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"feature_importance_{model_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"特征重要性图已保存至: {save_path}")
        plt.close()
    
    def plot_prediction_vs_actual(self, y_actual: np.ndarray, 
                                  y_pred: np.ndarray,
                                  model_name: str = "Model",
                                  save_path: Optional[str] = None):
        """
        绘制预测值vs实际值散点图
        
        Args:
            y_actual: 实际值
            y_pred: 预测值
            model_name: 模型名称
            save_path: 保存路径
        """
        logger.info(f"绘制 {model_name} 预测vs实际图...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 散点图（采样）
        sample_size = min(5000, len(y_actual))
        indices = np.random.choice(len(y_actual), sample_size, replace=False)
        
        axes[0].scatter(y_actual[indices], y_pred[indices], 
                       alpha=0.3, s=10, c='steelblue', edgecolors='gray', linewidth=0.5)
        
        # 理想线
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想拟合')
        
        axes[0].set_xlabel('实际值 (秒)', fontsize=12)
        axes[0].set_ylabel('预测值 (秒)', fontsize=12)
        axes[0].set_title(f'{model_name} - 预测 vs 实际', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # 残差分布
        residuals = y_actual - y_pred
        
        axes[1].hist(residuals, bins=100, color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'均值: {residuals.mean():.2f}')
        axes[1].set_xlabel('残差 (秒)', fontsize=12)
        axes[1].set_ylabel('频次', fontsize=12)
        axes[1].set_title(f'{model_name} - 残差分布', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"prediction_vs_actual_{model_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"预测vs实际图已保存至: {save_path}")
        plt.close()
    
    def plot_performance_comparison(self, methods: List[str], times: List[float], 
                                   title: str = "Performance Comparison",
                                   save_path: Optional[str] = None,
                                   log_scale: bool = False):
        """
        绘制性能对比图（通用）
        
        Args:
            methods: 方法名称列表
            times: 对应的时间列表
            title: 图表标题
            save_path: 保存路径
            log_scale: 是否使用对数刻度
        """
        logger.info(f"绘制性能对比图: {title}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(methods))
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        
        bars = ax.bar(x_pos, times, color=colors, edgecolor='black', alpha=0.8, width=0.6)
        
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('Time (seconds, log scale)', fontsize=12)
        else:
            ax.set_ylabel('Time (seconds)', fontsize=12)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 在柱子上标注数值
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            if log_scale:
                label = f'{time_val:.4f}s' if time_val < 1 else f'{time_val:.2f}s'
            else:
                label = f'{time_val:.2f}s'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "performance_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"性能对比图已保存至: {save_path}")
        plt.close()
    
    def plot_memory_comparison(self, labels: List[str], memories: List[float],
                              title: str = "Memory Usage Comparison",
                              save_path: Optional[str] = None):
        """
        绘制内存使用对比图
        
        Args:
            labels: 标签列表
            memories: 内存占用列表（MB）
            title: 图表标题
            save_path: 保存路径
        """
        logger.info(f"绘制内存对比图: {title}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(labels))
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(x_pos, memories, color=colors, edgecolor='black', alpha=0.8, width=0.6)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
        ax.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 计算节省的百分比
        if len(memories) == 2:
            savings_pct = ((memories[0] - memories[1]) / memories[0] * 100)
            ax.text(0.5, max(memories) * 0.9, 
                   f'Savings: {savings_pct:.1f}%',
                   ha='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        # 在柱子上标注数值
        for bar, mem in zip(bars, memories):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mem:.1f} MB',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "memory_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"内存对比图已保存至: {save_path}")
        plt.close()
    
    def plot_feature_ablation_study(self, ablation_df: pd.DataFrame,
                                   save_path: Optional[str] = None):
        """
        绘制特征消融实验结果
        
        Args:
            ablation_df: 消融实验结果DataFrame
            save_path: 保存路径
        """
        logger.info("绘制特征消融实验图...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 准备数据
        feature_sets = ablation_df['Feature Set'].values
        rmse_values = ablation_df['RMSE'].values
        r2_values = ablation_df['R2'].values
        num_features = ablation_df['# Features'].values
        
        x_pos = np.arange(len(feature_sets))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feature_sets)))
        
        # RMSE变化
        bars1 = axes[0].bar(x_pos, rmse_values, color=colors, edgecolor='black', alpha=0.8, width=0.6)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([f.split('(')[0].strip() for f in feature_sets], 
                               rotation=15, ha='right', fontsize=9)
        axes[0].set_ylabel('RMSE', fontsize=12)
        axes[0].set_title('RMSE vs Feature Set', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        
        # 标注数值
        for bar, val in zip(bars1, rmse_values):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # R²变化
        bars2 = axes[1].bar(x_pos, r2_values, color=colors, edgecolor='black', alpha=0.8, width=0.6)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([f.split('(')[0].strip() for f in feature_sets], 
                               rotation=15, ha='right', fontsize=9)
        axes[1].set_ylabel('R² Score', fontsize=12)
        axes[1].set_title('R² Score vs Feature Set', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        
        # 标注数值
        for bar, val in zip(bars2, r2_values):
            axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "feature_ablation_study.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"特征消融实验图已保存至: {save_path}")
        plt.close()


if __name__ == "__main__":
    print("可视化模块测试完成")
