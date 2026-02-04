"""
数据获取模块 - 使用Binance API直接获取K线数据
"""
import requests
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Optional
import time


class BinanceDataFetcher:
    """Binance数据获取器"""
    
    BASE_URL = "https://api.binance.com"
    
    def __init__(self, db_path: str = "data/crypto.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # K线数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS klines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                quote_volume REAL,
                trades INTEGER,
                UNIQUE(symbol, timeframe, timestamp)
            )
        ''')
        
        # 创建索引
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_klines_symbol_time 
            ON klines(symbol, timeframe, timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '15m',
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            symbol: 交易对, e.g. 'ETHUSDT'
            timeframe: 时间周期, e.g. '15m'
            start_time: 开始时间戳 (毫秒)
            end_time: 结束时间戳 (毫秒)
            limit: 每次获取数量
            
        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        endpoint = f"{self.BASE_URL}/api/v3/klines"
        
        params = {
            'symbol': symbol.replace('/', ''),
            'interval': timeframe,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Binance kline format: [
            #   0: open time, 1: open, 2: high, 3: low, 4: close, 5: volume,
            #   6: close time, 7: quote volume, 8: trades, 9: taker buy volume, 10: taker buy quote volume, 11: ignore
            # ]
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # 转换类型
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['trades'] = pd.to_numeric(df['trades'], errors='coerce').astype(int)
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype(int)
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            return df[['timestamp', 'datetime', 'symbol', 'timeframe', 
                       'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str = '15m',
        months: int = 6
    ) -> pd.DataFrame:
        """
        获取历史数据 (支持大数据量)
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            months: 获取多少个月的数据
        """
        print(f"Fetching {months} months of {timeframe} data for {symbol}...")
        
        # 计算开始时间
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=30*months)).timestamp() * 1000)
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            df = self.fetch_ohlcv(
                symbol, 
                timeframe, 
                start_time=current_start,
                limit=1000
            )
            
            if df.empty:
                break
                
            all_data.append(df)
            
            # 更新start_time为最后一条数据的时间 + 1个周期
            last_timestamp = df['timestamp'].iloc[-1]
            current_start = last_timestamp + 1
            
            # 进度显示
            progress_date = pd.to_datetime(last_timestamp, unit='ms')
            print(f"  Fetched up to: {progress_date}")
            
            # 避免频率限制
            time.sleep(0.5)
            
            # 如果已经获取到最新数据,退出
            if last_timestamp >= end_time:
                break
        
        if not all_data:
            return pd.DataFrame()
            
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'])
        combined_df = combined_df.sort_values('timestamp')
        
        print(f"Total records fetched: {len(combined_df)}")
        return combined_df
    
    def save_to_db(self, df: pd.DataFrame):
        """保存数据到数据库"""
        if df.empty:
            return
            
        conn = sqlite3.connect(self.db_path)
        
        # 准备数据
        data_to_insert = []
        for _, row in df.iterrows():
            data_to_insert.append((
                row['symbol'],
                row['timeframe'],
                int(row['timestamp']),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                float(row['quote_volume']) if pd.notna(row['quote_volume']) else None,
                int(row['trades']) if pd.notna(row['trades']) else None
            ))
        
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR REPLACE INTO klines 
            (symbol, timeframe, timestamp, open, high, low, close, volume, quote_volume, trades)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
        print(f"Saved {len(data_to_insert)} records to database")
    
    def load_from_db(
        self,
        symbol: str,
        timeframe: str = '15m',
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """从数据库加载数据"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM klines 
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
        conn.close()
        
        if not df.empty:
            df = df.sort_values('timestamp')
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    
    def get_all_symbols(self) -> List[str]:
        """获取所有已存储的交易对"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT symbol FROM klines')
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        return symbols


def download_all_data():
    """下载所有需要的数据"""
    fetcher = BinanceDataFetcher()
    
    # 交易对列表
    symbols = ['ETH/USDT', 'BTC/USDT']
    timeframe = '15m'
    months = 6
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Processing {symbol}")
        print(f"{'='*50}")
        
        df = fetcher.fetch_historical_data(symbol, timeframe, months)
        
        if not df.empty:
            fetcher.save_to_db(df)
            print(f"Data range: {df['datetime'].min()} to {df['datetime'].max()}")
        else:
            print(f"No data fetched for {symbol}")
    
    print("\n" + "="*50)
    print("Data download completed!")
    print("="*50)


if __name__ == "__main__":
    download_all_data()
