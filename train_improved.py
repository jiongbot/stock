#!/usr/bin/env python3
"""
改进版训练脚本 - 使用更多数据和特征工程优化
"""
import sys
import os
sys.path.append('/home/admin/code/stock')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
from datetime import datetime

from data.fetcher import init_database, download_historical_data, get_klines_from_db
from features.technical import calculate_features, get_feature_columns

def create_advanced_features(df):
    """创建更高级的特征"""
    df = df.copy()
    
    # 价格动量
    for period in [3, 5, 10, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(period)
        df[f'volatility_{period}'] = df['returns'].rolling(period).std()
    
    # 趋势特征
    df['trend_5'] = (df['close'] > df['close'].shift(5)).astype(int)
    df['trend_10'] = (df['close'] > df['close'].shift(10)).astype(int)
    df['trend_20'] = (df['close'] > df['close'].shift(20)).astype(int)
    
    # 成交量特征
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ma_10'] = df['volume'].rolling(10).mean()
    df['volume_trend'] = df['volume_ma_5'] / df['volume_ma_10']
    
    # 价格位置
    df['price_vs_high_20'] = df['close'] / df['high'].rolling(20).max()
    df['price_vs_low_20'] = df['close'] / df['low'].rolling(20).min()
    
    # 波动率特征
    df['atr_14'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    df['atr_ratio'] = df['atr_14'] / df['close']
    
    return df

def train_with_walk_forward(df, symbol, lookback=20):
    """使用滚动窗口训练"""
    print(f"训练 {symbol}...")
    
    # 创建高级特征
    df = create_advanced_features(df)
    df = calculate_features(df)
    df = df.dropna()
    
    if len(df) < 500:
        print(f"数据不足: {len(df)}")
        return None
    
    print(f"有效数据: {len(df)} 条")
    
    # 扩展特征列
    base_features = get_feature_columns()
    extra_features = ['momentum_3', 'momentum_5', 'momentum_10', 'momentum_20',
                      'volatility_3', 'volatility_5', 'volatility_10', 'volatility_20',
                      'trend_5', 'trend_10', 'trend_20', 'volume_trend',
                      'price_vs_high_20', 'price_vs_low_20', 'atr_ratio']
    feature_cols = base_features + extra_features
    feature_cols = [f for f in feature_cols if f in df.columns]
    
    print(f"特征数: {len(feature_cols)}")
    
    # 准备序列数据
    X_data = []
    y_data = []
    
    for i in range(lookback, len(df)):
        features = []
        for j in range(lookback):
            row_features = df.iloc[i - lookback + j][feature_cols].values
            features.extend(row_features)
        X_data.append(features)
        y_data.append(df.iloc[i]['target'])
    
    X = np.array(X_data)
    y = np.array(y_data)
    
    # 使用最后20%作为测试集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练多个模型并选择最佳
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ''
    
    for name, model in models.items():
        print(f"训练 {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  {name} 准确率: {accuracy*100:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name
    
    # 最终评估
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled) if hasattr(best_model, 'predict_proba') else None
    
    results = {
        'model_name': best_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'training_time': datetime.now().isoformat(),
        'symbol': symbol,
        'data_count': len(df)
    }
    
    print(f"\n✅ 最佳模型: {best_name}")
    print(f"   准确率: {results['accuracy']*100:.2f}%")
    
    # 保存
    model_dir = f'/home/admin/code/stock/models/{symbol.replace("/", "_")}_latest'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    with open(os.path.join(model_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    init_database()
    
    # 下载更多数据
    print("下载历史数据...")
    download_historical_data('BTC/USDT', days=90)
    download_historical_data('ETH/USDT', days=90)
    
    # 训练
    results = {}
    for symbol in ['BTC/USDT', 'ETH/USDT']:
        df = get_klines_from_db(symbol, limit=10000)
        if len(df) > 1000:
            results[symbol] = train_with_walk_forward(df, symbol)
    
    # 总结
    print(f"\n{'='*60}")
    print("训练总结")
    print(f"{'='*60}")
    for symbol, result in results.items():
        if result:
            status = "✅" if result['accuracy'] >= 0.85 else "⚠️"
            print(f"{status} {symbol}: {result['accuracy']*100:.2f}%")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
