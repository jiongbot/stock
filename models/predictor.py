"""
机器学习模型模块 - 训练预测模型
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
import joblib
import os
import json
from datetime import datetime

# 模型保存路径
MODELS_DIR = os.path.join(os.path.dirname(__file__))

def prepare_data(df, feature_cols, target_col='target', test_size=0.2, lookback=10):
    """
    准备训练和测试数据
    
    Args:
        df: 包含特征的DataFrame
        feature_cols: 特征列名列表
        target_col: 目标列名
        test_size: 测试集比例
        lookback: 回看周期数 (用于创建序列特征)
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # 删除缺失值
    df_clean = df.dropna()
    
    if len(df_clean) < 100:
        raise ValueError(f"数据量不足: {len(df_clean)} 行")
    
    # 添加滞后特征
    X_data = []
    y_data = []
    
    for i in range(lookback, len(df_clean)):
        # 获取lookback个周期的特征
        features = []
        for j in range(lookback):
            row_features = df_clean.iloc[i - lookback + j][feature_cols].values
            features.extend(row_features)
        
        X_data.append(features)
        y_data.append(df_clean.iloc[i][target_col])
    
    X = np.array(X_data)
    y = np.array(y_data)
    
    # 划分训练集和测试集 (按时间顺序)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_random_forest(X_train, y_train, X_test, y_test):
    """训练随机森林模型"""
    print("训练 Random Forest...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # 使用较小的网格搜索
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=3, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest 最佳参数: {grid_search.best_params_}")
    print(f"Random Forest 准确率: {accuracy:.4f}")
    
    return best_model, accuracy, y_pred

def train_xgboost(X_train, y_train, X_test, y_test):
    """训练XGBoost模型"""
    if not HAS_XGBOOST:
        print("XGBoost 未安装，跳过")
        return None, 0, None
    
    print("训练 XGBoost...")
    
    # 计算类别权重
    class_counts = np.bincount(y_train.astype(int))
    scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    
    grid_search = GridSearchCV(
        xgb_model, param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"XGBoost 最佳参数: {grid_search.best_params_}")
    print(f"XGBoost 准确率: {accuracy:.4f}")
    
    return best_model, accuracy, y_pred

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """训练梯度提升模型"""
    print("训练 Gradient Boosting...")
    
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Gradient Boosting 准确率: {accuracy:.4f}")
    
    return gb, accuracy, y_pred

def evaluate_model(model, X_test, y_test, model_name='Model'):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    print(f"\n{'='*50}")
    print(f"{model_name} 评估结果:")
    print(f"{'='*50}")
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"精确率: {results['precision']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    print(f"F1分数: {results['f1']:.4f}")
    print(f"\n混淆矩阵:")
    print(results['confusion_matrix'])
    
    return results

def train_all_models(df, symbol, feature_cols, target_col='target'):
    """
    训练所有模型并选择最佳模型
    
    Returns:
        best_model, scaler, results_dict
    """
    print(f"\n{'='*60}")
    print(f"开始训练模型 - {symbol}")
    print(f"{'='*60}")
    
    # 准备数据
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, feature_cols, target_col)
    
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    print(f"特征维度: {X_train.shape[1]}")
    
    # 训练多个模型
    models_results = {}
    
    # Random Forest
    rf_model, rf_acc, rf_pred = train_random_forest(X_train, y_train, X_test, y_test)
    models_results['RandomForest'] = {
        'model': rf_model,
        'accuracy': rf_acc,
        'predictions': rf_pred
    }
    
    # XGBoost (如果可用)
    if HAS_XGBOOST:
        xgb_model, xgb_acc, xgb_pred = train_xgboost(X_train, y_train, X_test, y_test)
        if xgb_model is not None:
            models_results['XGBoost'] = {
                'model': xgb_model,
                'accuracy': xgb_acc,
                'predictions': xgb_pred
            }
    
    # Gradient Boosting
    gb_model, gb_acc, gb_pred = train_gradient_boosting(X_train, y_train, X_test, y_test)
    models_results['GradientBoosting'] = {
        'model': gb_model,
        'accuracy': gb_acc,
        'predictions': gb_pred
    }
    
    # 选择最佳模型
    best_model_name = max(models_results.keys(), key=lambda k: models_results[k]['accuracy'])
    best_model = models_results[best_model_name]['model']
    best_accuracy = models_results[best_model_name]['accuracy']
    
    print(f"\n{'='*60}")
    print(f"最佳模型: {best_model_name}, 准确率: {best_accuracy:.4f}")
    print(f"{'='*60}")
    
    # 详细评估最佳模型
    final_results = evaluate_model(best_model, X_test, y_test, best_model_name)
    final_results['all_models'] = {
        name: {'accuracy': data['accuracy']} 
        for name, data in models_results.items()
    }
    
    return best_model, scaler, final_results, X_test, y_test

def save_model(model, scaler, symbol, model_name, results):
    """保存模型和相关信息"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(MODELS_DIR, f"{symbol.replace('/', '_')}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(model, model_path)
    
    # 保存scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    # 保存结果
    results_path = os.path.join(model_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 保存最新模型的软链接
    latest_dir = os.path.join(MODELS_DIR, f"{symbol.replace('/', '_')}_latest")
    if os.path.islink(latest_dir):
        os.unlink(latest_dir)
    if os.path.exists(latest_dir):
        import shutil
        shutil.rmtree(latest_dir)
    os.symlink(model_dir, latest_dir)
    
    print(f"模型已保存到: {model_dir}")
    return model_dir

def load_model(symbol):
    """加载最新模型"""
    latest_dir = os.path.join(MODELS_DIR, f"{symbol.replace('/', '_')}_latest")
    
    if not os.path.exists(latest_dir):
        return None, None, None
    
    model_path = os.path.join(latest_dir, 'model.pkl')
    scaler_path = os.path.join(latest_dir, 'scaler.pkl')
    results_path = os.path.join(latest_dir, 'results.json')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return model, scaler, results

if __name__ == '__main__':
    # 测试模型训练
    import sys
    sys.path.append('/home/admin/code/stock')
    from data.fetcher import get_klines_from_db
    from features.technical import calculate_features, get_feature_columns
    
    df = get_klines_from_db('BTC/USDT', limit=5000)
    if not df.empty:
        df_features = calculate_features(df)
        feature_cols = get_feature_columns()
        
        model, scaler, results, _, _ = train_all_models(df_features, 'BTC/USDT', feature_cols)
        save_model(model, scaler, 'BTC/USDT', results['model_name'], results)
