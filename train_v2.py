"""
改进的模型训练脚本 - 使用更长的预测周期和集成学习
"""
import sys
sys.path.append('/home/admin/code/stock')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
import json
from datetime import datetime

from data.fetcher import get_klines_from_db, download_historical_data
from features.enhanced import calculate_features, get_feature_columns

# 模型保存路径
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def resample_to_timeframe(df, timeframe='1h'):
    """
    将15分钟K线重采样到更长的时间周期
    
    Args:
        df: 15分钟K线DataFrame
        timeframe: 目标周期 ('1h', '4h', '1d')
    
    Returns:
        重采样后的DataFrame
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # 重采样规则
    if timeframe == '1h':
        rule = '1H'
    elif timeframe == '4h':
        rule = '4H'
    elif timeframe == '1d':
        rule = '1D'
    else:
        rule = '15min'
    
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    resampled.reset_index(inplace=True)
    return resampled


def prepare_data_advanced(df, feature_cols, target_col='target', test_size=0.2, lookback=20):
    """
    高级数据准备 - 使用序列特征和更复杂的数据处理
    """
    # 删除缺失值
    df_clean = df.dropna()
    
    if len(df_clean) < 500:
        raise ValueError(f"数据量不足: {len(df_clean)} 行")
    
    # 添加滞后特征和滚动统计
    X_data = []
    y_data = []
    timestamps = []
    
    for i in range(lookback, len(df_clean)):
        # 获取lookback个周期的特征
        features = []
        
        for j in range(lookback):
            row_idx = i - lookback + j
            row_features = df_clean.iloc[row_idx][feature_cols].values
            features.extend(row_features)
        
        # 添加滚动统计特征
        window = df_clean.iloc[i-lookback:i]
        for col in feature_cols[:10]:  # 只对前10个特征添加统计
            features.append(window[col].mean())
            features.append(window[col].std())
            features.append(window[col].max() - window[col].min())  # range
        
        X_data.append(features)
        y_data.append(df_clean.iloc[i][target_col])
        timestamps.append(df_clean.iloc[i]['timestamp'] if 'timestamp' in df_clean.columns else i)
    
    X = np.array(X_data)
    y = np.array(y_data)
    
    # 删除目标为nan的样本
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    timestamps = [timestamps[i] for i in range(len(timestamps)) if valid_mask[i]]
    
    # 按时间顺序划分训练集和测试集
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    timestamps_train = timestamps[:split_idx]
    timestamps_test = timestamps[split_idx:]
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, timestamps_train, timestamps_test


def train_random_forest_advanced(X_train, y_train, X_test, y_test):
    """训练改进的随机森林模型"""
    print("训练 Random Forest (改进版)...")
    
    # 使用更大的参数空间
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # 处理类别不平衡
    )
    
    # 使用TimeSeriesSplit进行交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid_search = GridSearchCV(
        rf, param_grid,
        cv=tscv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        n_iter=20  # 随机搜索20种组合
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest 最佳参数: {grid_search.best_params_}")
    print(f"Random Forest 准确率: {accuracy:.4f}")
    
    # 特征重要性
    feature_importance = best_model.feature_importances_
    
    return best_model, accuracy, y_pred, feature_importance


def train_gradient_boosting_advanced(X_train, y_train, X_test, y_test):
    """训练改进的梯度提升模型"""
    print("训练 Gradient Boosting (改进版)...")
    
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    gb = GradientBoostingClassifier(
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4
    )
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid_search = GridSearchCV(
        gb, param_grid,
        cv=tscv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        n_iter=20
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Gradient Boosting 最佳参数: {grid_search.best_params_}")
    print(f"Gradient Boosting 准确率: {accuracy:.4f}")
    
    return best_model, accuracy, y_pred


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """训练逻辑回归作为基线模型"""
    print("训练 Logistic Regression...")
    
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        C=1.0,
        penalty='l2'
    )
    
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Logistic Regression 准确率: {accuracy:.4f}")
    
    return lr, accuracy, y_pred


def create_ensemble_model(models, X_train, y_train, X_test, y_test):
    """创建集成模型"""
    print("创建集成模型...")
    
    # 使用软投票（概率加权）
    ensemble = VotingClassifier(
        estimators=[
            ('rf', models['RandomForest']['model']),
            ('gb', models['GradientBoosting']['model']),
            ('lr', models['LogisticRegression']['model'])
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"集成模型准确率: {accuracy:.4f}")
    
    return ensemble, accuracy, y_pred


def evaluate_model_detailed(model, X_test, y_test, model_name='Model'):
    """详细评估模型性能"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
    }
    
    # 计算方向准确率（只看涨跌，不看幅度）
    correct_direction = (y_pred == y_test).sum()
    total = len(y_test)
    results['direction_accuracy'] = correct_direction / total
    
    print(f"\n{'='*60}")
    print(f"{model_name} 详细评估结果:")
    print(f"{'='*60}")
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"精确率: {results['precision']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    print(f"F1分数: {results['f1']:.4f}")
    print(f"方向准确率: {results['direction_accuracy']:.4f}")
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    return results


def train_symbol(symbol='BTC/USDT', timeframe='1h', lookback=20):
    """
    训练单个交易对的模型
    
    Args:
        symbol: 交易对 (BTC/USDT, ETH/USDT)
        timeframe: 时间周期 (1h, 4h)
        lookback: 回看周期数
    """
    print(f"\n{'='*70}")
    print(f"开始训练 - {symbol} - {timeframe} - lookback={lookback}")
    print(f"{'='*70}\n")
    
    # 获取数据
    df_15m = get_klines_from_db(symbol, limit=20000)
    
    if df_15m.empty:
        print(f"数据库中没有 {symbol} 数据，尝试下载...")
        download_historical_data(symbol, days=365)
        df_15m = get_klines_from_db(symbol, limit=20000)
    
    if len(df_15m) < 1000:
        print(f"数据量不足: {len(df_15m)} 行")
        return None
    
    print(f"原始15分钟数据: {len(df_15m)} 行")
    
    # 重采样到目标周期
    if timeframe != '15m':
        df = resample_to_timeframe(df_15m, timeframe)
    else:
        df = df_15m
    
    print(f"{timeframe}周期数据: {len(df)} 行")
    
    # 计算特征
    df_features = calculate_features(df, timeframe=timeframe)
    feature_cols = get_feature_columns()
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"目标分布:\n{df_features['target'].value_counts()}")
    
    # 准备数据
    X_train, X_test, y_train, y_test, scaler, ts_train, ts_test = prepare_data_advanced(
        df_features, feature_cols, lookback=lookback
    )
    
    print(f"\n训练集: {len(X_train)}, 测试集: {len(X_test)}")
    print(f"特征维度: {X_train.shape[1]}")
    
    # 训练多个模型
    models_results = {}
    
    # Random Forest
    try:
        rf_model, rf_acc, rf_pred, rf_importance = train_random_forest_advanced(
            X_train, y_train, X_test, y_test
        )
        models_results['RandomForest'] = {
            'model': rf_model,
            'accuracy': rf_acc,
            'predictions': rf_pred
        }
    except Exception as e:
        print(f"Random Forest 训练失败: {e}")
    
    # Gradient Boosting
    try:
        gb_model, gb_acc, gb_pred = train_gradient_boosting_advanced(
            X_train, y_train, X_test, y_test
        )
        models_results['GradientBoosting'] = {
            'model': gb_model,
            'accuracy': gb_acc,
            'predictions': gb_pred
        }
    except Exception as e:
        print(f"Gradient Boosting 训练失败: {e}")
    
    # Logistic Regression
    try:
        lr_model, lr_acc, lr_pred = train_logistic_regression(
            X_train, y_train, X_test, y_test
        )
        models_results['LogisticRegression'] = {
            'model': lr_model,
            'accuracy': lr_acc,
            'predictions': lr_pred
        }
    except Exception as e:
        print(f"Logistic Regression 训练失败: {e}")
    
    if len(models_results) == 0:
        print("所有模型训练失败")
        return None
    
    # 创建集成模型
    try:
        ensemble_model, ensemble_acc, ensemble_pred = create_ensemble_model(
            models_results, X_train, y_train, X_test, y_test
        )
        models_results['Ensemble'] = {
            'model': ensemble_model,
            'accuracy': ensemble_acc,
            'predictions': ensemble_pred
        }
    except Exception as e:
        print(f"集成模型创建失败: {e}")
    
    # 选择最佳模型
    best_model_name = max(models_results.keys(), key=lambda k: models_results[k]['accuracy'])
    best_model = models_results[best_model_name]['model']
    best_accuracy = models_results[best_model_name]['accuracy']
    
    print(f"\n{'='*70}")
    print(f"最佳模型: {best_model_name}, 准确率: {best_accuracy:.4f}")
    print(f"{'='*70}")
    
    # 详细评估
    final_results = evaluate_model_detailed(best_model, X_test, y_test, best_model_name)
    final_results['all_models'] = {
        name: {'accuracy': data['accuracy']}
        for name, data in models_results.items()
    }
    final_results['timeframe'] = timeframe
    final_results['lookback'] = lookback
    final_results['symbol'] = symbol
    
    # 保存模型
    model_dir = save_model_advanced(
        best_model, scaler, symbol, timeframe, best_model_name, 
        final_results, feature_cols, lookback
    )
    
    return best_model, scaler, final_results


def save_model_advanced(model, scaler, symbol, timeframe, model_name, results, feature_cols, lookback):
    """保存模型和相关信息"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_symbol = symbol.replace('/', '_')
    model_dir = os.path.join(MODELS_DIR, f"{safe_symbol}_{timeframe}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(model, model_path)
    
    # 保存scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    # 保存配置
    config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'model_name': model_name,
        'lookback': lookback,
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'timestamp': timestamp
    }
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 保存结果
    results_path = os.path.join(model_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 保存最新模型的软链接
    latest_dir = os.path.join(MODELS_DIR, f"{safe_symbol}_{timeframe}_latest")
    if os.path.islink(latest_dir):
        os.unlink(latest_dir)
    if os.path.exists(latest_dir):
        import shutil
        shutil.rmtree(latest_dir)
    os.symlink(model_dir, latest_dir)
    
    print(f"\n模型已保存到: {model_dir}")
    return model_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练加密货币预测模型')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='交易对')
    parser.add_argument('--timeframe', type=str, default='1h', choices=['15m', '1h', '4h'], help='时间周期')
    parser.add_argument('--lookback', type=int, default=20, help='回看周期数')
    
    args = parser.parse_args()
    
    train_symbol(args.symbol, args.timeframe, args.lookback)
