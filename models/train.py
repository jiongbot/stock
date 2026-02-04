"""
机器学习模型模块 - RandomForest / GradientBoosting
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import sys
sys.path.append('/home/admin/code/stock')
from features.indicators import TechnicalIndicators


class CryptoPredictor:
    """加密货币预测模型"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Args:
            model_type: 'random_forest' 或 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = TechnicalIndicators.get_feature_columns()
        self.is_trained = False
        self.training_info = {}
    
    def build_model(self, **kwargs):
        """构建模型"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 5),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        scale_features: bool = True
    ) -> Dict:
        """
        训练模型
        
        Returns:
            训练结果字典
        """
        # 时间序列分割 (避免数据泄露)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 特征标准化
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # 构建并训练模型
        if self.model is None:
            self.build_model()
        
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # 预测
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # 计算指标
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        results = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_precision': precision_score(y_test, y_test_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'test_size': len(y_test),
            'train_size': len(y_train)
        }
        
        self.training_info = results
        self.is_trained = True
        
        print(f"\nTraining Results:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test Precision: {results['test_precision']:.4f}")
        print(f"  Test Recall: {results['test_recall']:.4f}")
        print(f"  Test F1: {results['test_f1']:.4f}")
        
        return results
    
    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """预测"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        
        if return_proba:
            return self.model.predict_proba(X_scaled)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        importance = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'training_info': self.training_info,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        self.training_info = model_data['training_info']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {path}")


class Backtester:
    """回测框架"""
    
    def __init__(self, model: CryptoPredictor):
        self.model = model
    
    def walk_forward_backtest(
        self,
        df: pd.DataFrame,
        train_window: int = 2000,
        test_window: int = 500,
        step: int = 500
    ) -> Dict:
        """
        滚动窗口回测
        
        Args:
            train_window: 训练窗口大小
            test_window: 测试窗口大小
            step: 每次滚动的步长
        """
        # 准备特征
        X, y, df_features = TechnicalIndicators.prepare_features(df)
        
        all_predictions = []
        all_actuals = []
        all_timestamps = []
        
        # 滚动窗口
        for i in range(train_window, len(X) - test_window, step):
            train_start = i - train_window
            train_end = i
            test_start = i
            test_end = min(i + test_window, len(X))
            
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # 训练临时模型
            temp_model = CryptoPredictor(self.model.model_type)
            temp_model.build_model()
            
            X_train_scaled = temp_model.scaler.fit_transform(X_train)
            X_test_scaled = temp_model.scaler.transform(X_test)
            
            temp_model.model.fit(X_train_scaled, y_train)
            predictions = temp_model.model.predict(X_test_scaled)
            
            all_predictions.extend(predictions)
            all_actuals.extend(y_test)
            all_timestamps.extend(df_features['timestamp'].iloc[test_start:test_end].values)
            
            print(f"  Window {i//step + 1}: Train {train_start}-{train_end}, Test {test_start}-{test_end}")
        
        # 计算回测结果
        accuracy = accuracy_score(all_actuals, all_predictions)
        
        # 计算涨跌预测的正确率
        results_df = pd.DataFrame({
            'timestamp': all_timestamps,
            'actual': all_actuals,
            'predicted': all_predictions
        })
        
        # 上涨预测准确率
        up_mask = np.array(all_actuals) == 1
        if up_mask.sum() > 0:
            up_accuracy = (np.array(all_predictions)[up_mask] == 1).mean()
        else:
            up_accuracy = 0
        
        # 下跌预测准确率
        down_mask = np.array(all_actuals) == 0
        if down_mask.sum() > 0:
            down_accuracy = (np.array(all_predictions)[down_mask] == 0).mean()
        else:
            down_accuracy = 0
        
        results = {
            'overall_accuracy': accuracy,
            'up_accuracy': up_accuracy,
            'down_accuracy': down_accuracy,
            'total_predictions': len(all_predictions),
            'up_predictions': sum(all_predictions),
            'down_predictions': len(all_predictions) - sum(all_predictions),
            'results_df': results_df
        }
        
        return results
    
    def simple_backtest(self, df: pd.DataFrame, train_ratio: float = 0.8) -> Dict:
        """简单回测 (单次划分)"""
        X, y, df_features = TechnicalIndicators.prepare_features(df)
        
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 训练
        temp_model = CryptoPredictor(self.model.model_type)
        temp_model.build_model()
        
        X_train_scaled = temp_model.scaler.fit_transform(X_train)
        X_test_scaled = temp_model.scaler.transform(X_test)
        
        temp_model.model.fit(X_train_scaled, y_train)
        predictions = temp_model.model.predict(X_test_scaled)
        
        # 计算结果
        accuracy = accuracy_score(y_test, predictions)
        
        results_df = pd.DataFrame({
            'timestamp': df_features['timestamp'].iloc[split_idx:].values,
            'actual': y_test,
            'predicted': predictions,
            'close': df_features['close'].iloc[split_idx:].values
        })
        
        return {
            'accuracy': accuracy,
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
            'results_df': results_df
        }


def train_and_evaluate(symbol: str = 'ETH/USDT', model_type: str = 'random_forest'):
    """训练并评估模型"""
    from data.fetch_data import BinanceDataFetcher
    
    print(f"\n{'='*60}")
    print(f"Training {model_type} model for {symbol}")
    print(f"{'='*60}\n")
    
    # 加载数据
    fetcher = BinanceDataFetcher()
    df = fetcher.load_from_db(symbol)
    
    if df.empty:
        print(f"No data found for {symbol}. Please run fetch_data.py first.")
        return None
    
    print(f"Loaded {len(df)} records from database")
    
    # 准备特征
    X, y, df_features = TechnicalIndicators.prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # 创建模型
    model = CryptoPredictor(model_type)
    model.build_model(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05 if model_type == 'gradient_boosting' else None
    )
    
    # 训练
    train_results = model.train(X, y, test_size=0.2)
    
    # 特征重要性
    print("\nTop 10 Feature Importances:")
    importance_df = model.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))
    
    # 回测
    print("\n" + "="*60)
    print("Running Backtest...")
    print("="*60)
    
    backtester = Backtester(model)
    
    # 简单回测
    simple_results = backtester.simple_backtest(df, train_ratio=0.8)
    print(f"\nSimple Backtest Results:")
    print(f"  Accuracy: {simple_results['accuracy']:.4f}")
    print(f"  Precision: {simple_results['precision']:.4f}")
    print(f"  Recall: {simple_results['recall']:.4f}")
    print(f"  F1 Score: {simple_results['f1']:.4f}")
    print(f"  Confusion Matrix: {simple_results['confusion_matrix']}")
    
    # 滚动窗口回测
    print("\nWalk-Forward Backtest Results:")
    walk_results = backtester.walk_forward_backtest(
        df, train_window=3000, test_window=500, step=500
    )
    print(f"  Overall Accuracy: {walk_results['overall_accuracy']:.4f}")
    print(f"  Up Prediction Accuracy: {walk_results['up_accuracy']:.4f}")
    print(f"  Down Prediction Accuracy: {walk_results['down_accuracy']:.4f}")
    print(f"  Total Predictions: {walk_results['total_predictions']}")
    
    # 保存模型
    model_path = f"models/{symbol.replace('/', '_')}_{model_type}.pkl"
    model.save(model_path)
    
    # 保存回测结果
    results_path = f"models/{symbol.replace('/', '_')}_{model_type}_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'symbol': symbol,
            'model_type': model_type,
            'train_results': train_results,
            'simple_backtest': {
                'accuracy': simple_results['accuracy'],
                'precision': simple_results['precision'],
                'recall': simple_results['recall'],
                'f1': simple_results['f1'],
                'confusion_matrix': simple_results['confusion_matrix']
            },
            'walk_forward_backtest': {
                'overall_accuracy': walk_results['overall_accuracy'],
                'up_accuracy': walk_results['up_accuracy'],
                'down_accuracy': walk_results['down_accuracy'],
                'total_predictions': walk_results['total_predictions']
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return model, simple_results, walk_results


if __name__ == "__main__":
    # 训练ETH模型
    train_and_evaluate('ETH/USDT', 'random_forest')
    
    # 训练BTC模型
    train_and_evaluate('BTC/USDT', 'random_forest')
