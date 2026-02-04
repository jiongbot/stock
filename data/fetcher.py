"""
数据获取模块 - 使用 Binance API 直接获取K线数据
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os
import time
import json

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(__file__), 'crypto_data.db')
BINANCE_API = "https://api.binance.com"

def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """初始化数据库表"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS kline_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            quote_volume REAL,
            UNIQUE(symbol, timestamp)
        )
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
        ON kline_data(symbol, timestamp)
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            predicted_direction INTEGER NOT NULL,
            confidence REAL NOT NULL,
            actual_direction INTEGER,
            correct INTEGER,
            features TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("数据库初始化完成")

def fetch_klines_binance(symbol='BTCUSDT', interval='15m', start_time=None, end_time=None, limit=1000):
    """
    从Binance获取K线数据
    
    Args:
        symbol: 交易对，如 'BTCUSDT'
        interval: 时间周期，如 '15m'
        start_time: 开始时间戳(毫秒)
        end_time: 结束时间戳(毫秒)
        limit: 每次获取的最大条数
    
    Returns:
        list: K线数据列表
    """
    url = f"{BINANCE_API}/api/v3/klines"
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    if start_time:
        params['startTime'] = int(start_time)
    if end_time:
        params['endTime'] = int(end_time)
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # 转换格式: [timestamp, open, high, low, close, volume, close_time, quote_volume, ...]
        klines = []
        for item in data:
            klines.append([
                int(item[0]),  # timestamp
                float(item[1]),  # open
                float(item[2]),  # high
                float(item[3]),  # low
                float(item[4]),  # close
                float(item[5]),  # volume
                float(item[7]) if len(item) > 7 else None  # quote_volume
            ])
        
        return klines
    except Exception as e:
        print(f"获取数据失败: {e}")
        return []

def save_klines_to_db(symbol, klines):
    """保存K线数据到数据库"""
    if not klines:
        return 0
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    count = 0
    for kline in klines:
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO kline_data 
                (symbol, timestamp, open, high, low, close, volume, quote_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                kline[0],  # timestamp
                kline[1],  # open
                kline[2],  # high
                kline[3],  # low
                kline[4],  # close
                kline[5],  # volume
                kline[6] if len(kline) > 6 else None  # quote_volume
            ))
            count += 1
        except Exception as e:
            print(f"保存K线失败: {e}")
    
    conn.commit()
    conn.close()
    return count

def download_historical_data(symbol='BTC/USDT', days=180, interval='15m'):
    """
    下载历史K线数据
    
    Args:
        symbol: 交易对
        days: 下载多少天的数据
        interval: 时间周期
    """
    init_database()
    
    # 转换交易对格式
    binance_symbol = symbol.replace('/', '')
    
    print(f"开始下载 {symbol} 的 {days} 天历史数据...")
    
    # 计算开始时间
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_klines = []
    current_since = start_time
    batch_count = 0
    
    while current_since < end_time:
        batch_count += 1
        print(f"批次 {batch_count}: 获取 {datetime.fromtimestamp(current_since/1000)} 开始的数据...")
        
        klines = fetch_klines_binance(
            binance_symbol, 
            interval, 
            start_time=current_since,
            limit=1000
        )
        
        if not klines:
            print("没有更多数据")
            break
        
        all_klines.extend(klines)
        
        # 更新since为最后一条数据的时间 + 15分钟
        last_timestamp = klines[-1][0]
        if last_timestamp <= current_since:
            break
        
        current_since = last_timestamp + 15 * 60 * 1000  # 15分钟
        
        if batch_count % 10 == 0:
            # 每10批次保存一次
            saved_count = save_klines_to_db(symbol, all_klines)
            print(f"已保存 {saved_count} 条K线数据")
            all_klines = []
        
        time.sleep(0.2)  # 避免请求过快
    
    # 保存剩余数据
    if all_klines:
        saved_count = save_klines_to_db(symbol, all_klines)
        print(f"最后保存 {saved_count} 条K线数据")
    
    print(f"数据下载完成")
    return len(all_klines)

def get_klines_from_db(symbol='BTC/USDT', limit=5000):
    """从数据库获取K线数据"""
    conn = get_db_connection()
    
    df = pd.read_sql_query('''
        SELECT * FROM kline_data 
        WHERE symbol = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', conn, params=(symbol, limit))
    
    conn.close()
    
    if not df.empty:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def get_latest_kline(symbol='BTC/USDT'):
    """获取最新的K线数据"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM kline_data 
        WHERE symbol = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
    ''', (symbol,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def update_latest_data(symbol='BTC/USDT', interval='15m'):
    """更新最新数据"""
    latest = get_latest_kline(symbol)
    binance_symbol = symbol.replace('/', '')
    
    if latest:
        since = latest['timestamp'] + 15 * 60 * 1000
    else:
        since = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
    
    klines = fetch_klines_binance(binance_symbol, interval, start_time=since)
    
    if klines:
        saved = save_klines_to_db(symbol, klines)
        print(f"更新了 {saved} 条新数据")
        return saved
    return 0

if __name__ == '__main__':
    # 测试数据获取
    init_database()
    download_historical_data('BTC/USDT', days=30)
    download_historical_data('ETH/USDT', days=30)
