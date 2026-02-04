#!/usr/bin/env python3
"""
训练脚本 - 下载数据、训练模型、保存结果
"""
import sys
import os
sys.path.append('/home/admin/code/stock')

from data.fetcher import init_database, download_historical_data, get_klines_from_db
from features.technical import calculate_features, get_feature_columns
from models.predictor import train_all_models, save_model

def train_symbol(symbol, days=180):
    """训练单个交易对的模型"""
    print(f"\n{'='*60}")
    print(f"训练 {symbol} 模型")
    print(f"{'='*60}")
    
    # 1. 下载历史数据
    print(f"\n[1/4] 下载 {days} 天历史数据...")
    download_historical_data(symbol, days=days)
    
    # 2. 获取数据并计算特征
    print(f"\n[2/4] 计算技术指标...")
    df = get_klines_from_db(symbol, limit=10000)
    if df.empty:
        raise ValueError(f"无法获取 {symbol} 的数据")
    
    print(f"获取到 {len(df)} 条K线数据")
    df_features = calculate_features(df)
    df_clean = df_features.dropna()
    print(f"特征计算后: {len(df_clean)} 条有效数据")
    
    # 3. 训练模型
    print(f"\n[3/4] 训练机器学习模型...")
    feature_cols = get_feature_columns()
    model, scaler, results, _, _ = train_all_models(df_clean, symbol, feature_cols)
    
    # 4. 保存模型
    print(f"\n[4/4] 保存模型...")
    model_dir = save_model(model, scaler, symbol, results['model_name'], results)
    
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"模型保存路径: {model_dir}")
    print(f"最终准确率: {results['accuracy']*100:.2f}%")
    print(f"{'='*60}\n")
    
    return results

def main():
    """主函数"""
    # 初始化数据库
    init_database()
    
    # 训练 BTC
    try:
        btc_results = train_symbol('BTC/USDT', days=180)
    except Exception as e:
        print(f"BTC训练失败: {e}")
        btc_results = None
    
    # 训练 ETH
    try:
        eth_results = train_symbol('ETH/USDT', days=180)
    except Exception as e:
        print(f"ETH训练失败: {e}")
        eth_results = None
    
    # 总结
    print(f"\n{'='*60}")
    print("训练总结")
    print(f"{'='*60}")
    
    if btc_results:
        print(f"BTC/USDT: {btc_results['accuracy']*100:.2f}% ({btc_results['model_name']})")
    if eth_results:
        print(f"ETH/USDT: {eth_results['accuracy']*100:.2f}% ({eth_results['model_name']})")
    
    # 检查是否达标
    all_pass = True
    if btc_results and btc_results['accuracy'] < 0.85:
        print(f"⚠️ BTC准确率未达标: {btc_results['accuracy']*100:.2f}% < 85%")
        all_pass = False
    if eth_results and eth_results['accuracy'] < 0.85:
        print(f"⚠️ ETH准确率未达标: {eth_results['accuracy']*100:.2f}% < 85%")
        all_pass = False
    
    if all_pass and btc_results and eth_results:
        print("\n✅ 所有模型准确率达标!")
    else:
        print("\n❌ 部分模型未达标，建议调整参数后重新训练")
    
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
