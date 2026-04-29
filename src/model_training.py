"""
模型训练模块
包含多种机器学习模型的训练和评估
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import time
import logging
from pathlib import Path
from typing import Dict, Tuple
import joblib
import sys

# 添加父目录到路径以导入config
sys.path.append(str(Path(__file__).parent.parent))
from config import RANDOM_SEED

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: dict):
        """
        初始化模型训练器
        
        Args:
            config: 模型配置字典
        """
        self.config = config
        self.models = {}
        self.results = {}
        self.feature_importance = {}
    
    def prepare_data(self, df, target_col='trip_duration'):
        """
        准备训练数据
        
        Args:
            df: 输入DataFrame
            target_col: 目标列名
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test, feature_cols)
        """
        logger.info("准备训练数据...")
        
        # 删除包含NaN的行
        initial_rows = len(df)
        df_clean = df.dropna(subset=[target_col])
        removed_rows = initial_rows - len(df_clean)
        if removed_rows > 0:
            logger.info(f"删除{removed_rows}条包含缺失目标值的记录")
        
        # 选择数值型特征和类别型特征
        exclude_cols = [target_col, 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
        
        # 自动识别特征列
        feature_cols = [col for col in df_clean.columns 
                       if col not in exclude_cols 
                       and df_clean[col].dtype in ['int32', 'int64', 'float32', 'float64', 'uint8']]
        
        logger.info(f"特征数量: {len(feature_cols)}")
        
        # 提取特征和标签
        X = df_clean[feature_cols].copy()
        y = df_clean[target_col].copy()
        
        # 填充特征中的NaN值
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                logger.debug(f"填充列{col}的NaN值，中位数={median_val:.2f}")
        
        # 进一步确保没有NaN
        X = X.fillna(0)
        
        # 分割数据
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED
        )
        
        logger.info(f"训练集: {len(X_train):,}, 验证集: {len(X_val):,}, 测试集: {len(X_test):,}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
    
    def train_linear_regression(self, X_train, y_train):
        """训练线性回归模型（基线）"""
        logger.info("\n" + "="*80)
        logger.info("训练线性回归模型 (Baseline)")
        logger.info("="*80)
        
        start_time = time.time()
        
        model = LinearRegression(n_jobs=-1)
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        self.models['linear_regression'] = model
        
        logger.info(f"训练时间: {training_time:.2f}秒")
        
        return model
    
    def train_random_forest(self, X_train, y_train, X_val=None, y_val=None):
        """训练随机森林模型"""
        logger.info("\n" + "="*80)
        logger.info("训练随机森林模型")
        logger.info("="*80)
        
        start_time = time.time()
        
        rf_config = self.config.get('random_forest', {})
        model = RandomForestRegressor(
            n_estimators=rf_config.get('n_estimators', 100),
            max_depth=rf_config.get('max_depth', 10),
            n_jobs=rf_config.get('n_jobs', -1),
            random_state=rf_config.get('random_state', 42),
            verbose=1
        )
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        self.models['random_forest'] = model
        self.feature_importance['random_forest'] = model.feature_importances_
        
        logger.info(f"训练时间: {training_time:.2f}秒")
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """训练XGBoost模型"""
        logger.info("\n" + "="*80)
        logger.info("训练XGBoost模型")
        logger.info("="*80)
        
        start_time = time.time()
        
        xgb_config = self.config.get('xgboost', {})
        model = xgb.XGBRegressor(
            n_estimators=xgb_config.get('n_estimators', 500),
            max_depth=xgb_config.get('max_depth', 6),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            n_jobs=xgb_config.get('n_jobs', -1),
            random_state=xgb_config.get('random_state', 42),
            early_stopping_rounds=50,
            verbosity=1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
        
        training_time = time.time() - start_time
        
        self.models['xgboost'] = model
        self.feature_importance['xgboost'] = model.feature_importances_
        
        logger.info(f"训练时间: {training_time:.2f}秒")
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """训练LightGBM模型（推荐）"""
        logger.info("\n" + "="*80)
        logger.info("训练LightGBM模型 (推荐)")
        logger.info("="*80)
        
        start_time = time.time()
        
        lgb_config = self.config.get('lightgbm', {})
        model = lgb.LGBMRegressor(
            n_estimators=lgb_config.get('n_estimators', 500),
            max_depth=lgb_config.get('max_depth', 6),
            learning_rate=lgb_config.get('learning_rate', 0.1),
            num_leaves=lgb_config.get('num_leaves', 31),
            n_jobs=lgb_config.get('n_jobs', -1),
            random_state=lgb_config.get('random_state', 42),
            verbose=lgb_config.get('verbose', -1)
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
        )
        
        training_time = time.time() - start_time
        
        self.models['lightgbm'] = model
        self.feature_importance['lightgbm'] = model.feature_importances_
        
        logger.info(f"训练时间: {training_time:.2f}秒")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            X_test: 测试集特征
            y_test: 测试集标签
            model_name: 模型名称
            
        Returns:
            dict: 评估指标字典
        """
        start_time = time.time()
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 处理NaN值和负值预测
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=0.0)
        y_pred = np.maximum(y_pred, 0)  # 行程时间不能为负
        
        prediction_time = time.time() - start_time
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 对数变换的指标（处理偏态分布）
        # 确保所有值都是正数才能进行log变换
        y_test_positive = np.maximum(y_test, 0)
        y_pred_positive = np.maximum(y_pred, 0)
        
        # 添加小常数避免log(0)
        log_rmse = np.sqrt(mean_squared_error(np.log1p(y_test_positive), np.log1p(y_pred_positive)))
        log_mae = mean_absolute_error(np.log1p(y_test_positive), np.log1p(y_pred_positive))
        
        metrics = {
            'model_name': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Log_RMSE': log_rmse,
            'Log_MAE': log_mae,
            'prediction_time': prediction_time,
            'predictions': y_pred
        }
        
        logger.info(f"RMSE: {rmse:.2f}")
        logger.info(f"MAE: {mae:.2f}")
        logger.info(f"R²: {r2:.4f}")
        logger.info(f"Log RMSE: {log_rmse:.4f}")
        logger.info(f"预测时间: {prediction_time:.2f}秒")
        
        self.results[model_name] = metrics
        
        return metrics
    
    def compare_models(self, X_test, y_test):
        """比较所有训练的模型"""
        logger.info("\n" + "="*80)
        logger.info("模型性能对比")
        logger.info("="*80)
        
        results_list = []
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            results_list.append(metrics)
        
        # 创建对比表格
        comparison_df = pd.DataFrame(results_list)
        comparison_df = comparison_df.sort_values('RMSE')
        
        logger.info("\n模型性能对比表:")
        logger.info(f"\n{comparison_df[['model_name', 'RMSE', 'MAE', 'R2', 'Log_RMSE']].to_string(index=False)}")
        
        return comparison_df
    
    def save_model(self, model_name: str, save_path: str):
        """保存模型"""
        if model_name in self.models:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.models[model_name], save_path)
            logger.info(f"模型已保存至: {save_path}")
    
    def load_model(self, model_name: str, load_path: str):
        """加载模型"""
        self.models[model_name] = joblib.load(load_path)
        logger.info(f"模型已加载: {load_path}")


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import MODEL_CONFIGS
    
    print("模型训练模块测试完成")
