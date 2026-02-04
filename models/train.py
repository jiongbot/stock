"""
机器学习模型模块 - 最终优化版
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import sys
sys.path.append('/home/admin/code/stock')
from features.indicators import TechnicalIndicators


class CryptoPredictor:
    """加密货币预测模型 - 集成模型"""
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = TechnicalIndicators.get_feature_columns()
        self.is_trained = False
        self.training_info = {}
    
    def build_model(self, **kwargs):
        """构建集成模型"""
        # 使用多个不同类型的模型进行投票
        rf = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 5),
            min_samples_split=kwargs.get('min_samples_split', 100),
            min_samples_leaf=kwargs.get('min_samples_leaf', 50),
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        et = ExtraTreesClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 5),
            min_samples_split=kwargs.get('min_samples_split', 100),
            min_samples_leaf=kwargs.get('min_samples_leaf', 50),
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 50),
            max_depth=kwargs.get('max_depth', 3),
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('et', et), ('gb', gb)],
            voting='soft'
        )
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        scale_features: bool = True
    ) -> Dict:
        """训练模型"""
        # 时间序列分割
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
        
        return results
    
    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """预测"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        
        if return_proba:
            return self.model.predict_proba(X_scaled)
        return self.model.predict(X_scaled)
    
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


class Backtester:
    """回测框架"""
    
    def __init__(self, model: CryptoPredictor):
        self.model = model
    
    def walk_forward_backtest(
        self,
        df: pd.DataFrame,
        train_window: int = 6000,
        test_window: int = 500,
        step: int = 500
    ) -> Dict:
        """滚动窗口回测"""
        X, y, df_features = TechnicalIndicators.prepare_features(df)
        
        all_predictions = []
        all_actuals = []
        all_probabilities = []
        
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
            probabilities = temp_model.model.predict_proba(X_test_scaled)
            
            all_predictions.extend(predictions)
            all_actuals.extend(y_test)
            all_probabilities.extend(probabilities.max(axis=1))
            
            print(f"  Window {i//step + 1}: Accuracy on test = {accuracy_score(y_test, predictions):.4f}")
        
        # 计算回测结果
        accuracy = accuracy_score(all_actuals, all_predictions)
        
        # 高置信度预测准确率 (概率 >= 0.55)
        high_conf_mask = np.array(all_probabilities) >= 0.55
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = accuracy_score(
                np.array(all_actuals)[high_conf_mask],
                np.array(all_predictions)[high_conf_mask]
            )
            high_conf_count = high_conf_mask.sum()
        else:
            high_conf_accuracy = 0
            high_conf_count = 0
        
        # 超高置信度预测准确率 (概率 >= 0.60)
        very_high_conf_mask = np.array(all_probabilities) >= 0.60
        if very_high_conf_mask.sum() > 0:
            very_high_conf_accuracy = accuracy_score(
                np.array(all_actuals)[very_high_conf_mask],
                np.array(all_predictions)[very_high_conf_mask]
            )
            very_high_conf_count = very_high_conf_mask.sum()
        else:
            very_high_conf_accuracy = 0
            very_high_conf_count = 0
        
        return {
            'overall_accuracy': accuracy,
            'high_conf_accuracy': high_conf_accuracy,
            'high_conf_count': int(high_conf_count),
            'high_conf_pct': high_conf_count / len(all_predictions),
            'very_high_conf_accuracy': very_high_conf_accuracy,
            'very_high_conf_count': int(very_high_conf_count),
            'very_high_conf_pct': very_high_conf_count / len(all_predictions),
            'total_predictions': len(all_predictions)
        }


def train_and_evaluate(symbol: str = 'ETH/USDT'):
    """训练并评估模型"""
    from data.fetch_data import BinanceDataFetcher
    
    print(f"\n{'='*60}")
    print(f"Training model for {symbol}")
    print(f"{'='*60}\n")
    
    # 加载数据
    fetcher = BinanceDataFetcher()
    df = fetcher.load_from_db(symbol)
    
    if df.empty:
        print(f"No data found for {symbol}")
        return None
    
    print(f"Loaded {len(df)} records from database")
    
    # 准备特征
    X, y, df_features = TechnicalIndicators.prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # 创建模型
    model = CryptoPredictor('ensemble')
    model.build_model(n_estimators=100, max_depth=5, min_samples_split=100, min_samples_leaf=50)
    
    # 训练
    train_results = model.train(X, y, test_size=0.2)
    
    # 回测
    print("\n" + "="*60)
    print("Running Walk-Forward Backtest...")
    print("="*60)
    
    backtester = Backtester(model)
    walk_results = backtester.walk_forward_backtest(df, train_window=6000, test_window=500, step=500)
    
    print(f"\nWalk-Forward Backtest Results:")
    print(f"  Overall Accuracy: {walk_results['overall_accuracy']:.4f}")
    print(f"  High Conf (≥55%) Accuracy: {walk_results['high_conf_accuracy']:.4f} ({walk_results['high_conf_count']} predictions, {walk_results['high_conf_pct']*100:.1f}%)")
    print(f"  Very High Conf (≥60%) Accuracy: {walk_results['very_high_conf_accuracy']:.4f} ({walk_results['very_high_conf_count']} predictions, {walk_results['very_high_conf_pct']*100:.1f}%)")
    
    # 保存模型
    model_path = f"models/{symbol.replace('/', '_')}_ensemble.pkl"
    model.save(model_path)
    
    # 保存回测结果
    results_path = f"models/{symbol.replace('/', '_')}_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'symbol': symbol,
            'train_results': train_results,
            'walk_forward_backtest': walk_results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return model, walk_results


if __name__ == "__main__":
    # 训练ETH模型
    train_and_evaluate('ETH/USDT')
    
    # 训练BTC模型  
    train_and_evaluate('BTC/USDT')
