#!/usr/bin/env python3
"""
测试脚本 - 验证数据重采样和训练流程
"""
import sys
sys.path.append('/home/admin/code/stock')

import pandas as pd
from data.fetcher import get_klines_from_db, download_historical_data
from features.enhanced import calculate_features, get_feature_columns
from train import resample_to_timeframe, prepare_data_advanced

def test_data_pipeline():
    """测试完整数据流程"""
    print("="*60)
    print("测试数据流程")
    print("="*60)
    
    # 1. 获取数据
    symbol = 'BTC/USDT'
    df_15m = get_klines_from_db(symbol, limit=5000)
    
    if df_15m.empty:
        print(f"数据库中没有 {symbol} 数据，尝试下载...")
        download_historical_data(symbol, days=180)
        df_15m = get_klines_from_db(symbol, limit=5000)
    
    print(f"\n1. 原始15分钟数据: {len(df_15m)} 行")
    print(f"   列: {list(df_15m.columns)}")
    print(f"   时间范围: {df_15m['timestamp'].min()} ~ {df_15m['timestamp'].max()}")
    
    if len(df_15m) == 0:
        print("错误: 没有数据!")
        return False
    
    # 2. 测试重采样
    for timeframe in ['1h', '4h']:
        print(f"\n2. 测试重采样到 {timeframe}")
        try:
            df_resampled = resample_to_timeframe(df_15m, timeframe)
            print(f"   重采样后: {len(df_resampled)} 行")
            
            if len(df_resampled) < 100:
                print(f"   警告: 数据量不足!")
                continue
            
            # 3. 计算特征
            print(f"   计算特征...")
            df_features = calculate_features(df_resampled, timeframe=timeframe)
            feature_cols = get_feature_columns()
            print(f"   特征数量: {len(feature_cols)}")
            print(f"   特征计算后: {len(df_features)} 行")
            
            # 4. 准备训练数据
            print(f"   准备训练数据...")
            X_train, X_test, y_train, y_test, scaler, ts_train, ts_test = prepare_data_advanced(
                df_features, feature_cols, lookback=20
            )
            print(f"   训练集: {len(X_train)}, 测试集: {len(X_test)}")
            print(f"   特征维度: {X_train.shape[1]}")
            print(f"   ✓ {timeframe} 测试通过!")
            
        except Exception as e:
            print(f"   ✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*60)
    print("所有测试通过!")
    print("="*60)
    return True

if __name__ == '__main__':
    success = test_data_pipeline()
    sys.exit(0 if success else 1)
