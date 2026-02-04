#!/usr/bin/env python3
"""
简化版训练脚本 - 快速训练模型
"""
import sys
import os
sys.path.append('/home/admin/code/stock')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import json
from datetime import datetime

from data.fetcher import init_database, get_klines_from_db
from features.technical import calculate_features, get_feature_columns

def train_model(symbol, lookback=10):
    """训练模型"""
    print(f"\n{'='*60}")
    print(f"训练 {symbol} 模型")
    print(f"{'='*60}")
    
    # 获取数据
    print("[1/3] 获取数据...")
    df = get_klines_from_db(symbol, limit=5000)
    if len(df) < 1000:
        print(f"数据不足: {len(df)} 条")
        return None
    
    print(f"获取到 {len(df)} 条K线数据")
    
    # 计算特征
    print("[2/3] 计算特征...")
    df_features = calculate_features(df)
    df_clean = df_features.dropna()
    print(f"特征计算后: {len(df_clean)} 条有效数据")
    
    feature_cols = get_feature_columns()
    
    # 准备数据
    X_data = []
    y_data = []
    
    for i in range(lookback, len(df_clean)):
        features = []
        for j in range(lookback):
            row_features = df_clean.iloc[i - lookback + j][feature_cols].values
            features.extend(row_features)
        X_data.append(features)
        y_data.append(df_clean.iloc[i]['target'])
    
    X = np.array(X_data)
    y = np.array(y_data)
    
    # 划分训练集和测试集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    print("[3/3] 训练 Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # 评估
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n评估结果:")
    print(f"  准确率: {accuracy*100:.2f}%")
    print(f"  精确率: {precision*100:.2f}%")
    print(f"  召回率: {recall*100:.2f}%")
    print(f"  F1分数: {f1*100:.2f}%")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:")
    print(cm)
    
    # 保存模型
    results = {
        'model_name': 'RandomForest',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'training_time': datetime.now().isoformat(),
        'symbol': symbol,
        'data_count': len(df_clean)
    }
    
    # 保存
    model_dir = f'/home/admin/code/stock/models/{symbol.replace("/", "_")}_latest'
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    with open(os.path.join(model_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n模型已保存到: {model_dir}")
    
    return results

def main():
    init_database()
    
    # 训练 BTC
    btc_results = train_model('BTC/USDT')
    
    # 训练 ETH
    eth_results = train_model('ETH/USDT')
    
    # 总结
    print(f"\n{'='*60}")
    print("训练总结")
    print(f"{'='*60}")
    
    if btc_results:
        print(f"BTC/USDT: {btc_results['accuracy']*100:.2f}%")
    if eth_results:
        print(f"ETH/USDT: {eth_results['accuracy']*100:.2f}%")
    
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
