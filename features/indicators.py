"""
技术指标计算模块
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """添加所有技术指标特征"""
        df = df.copy()
        
        # 确保数据按时间排序
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 基础价格特征
        df = TechnicalIndicators.add_price_features(df)
        
        # 趋势指标
        df = TechnicalIndicators.add_ema_features(df)
        df = TechnicalIndicators.add_macd(df)
        
        # 动量指标
        df = TechnicalIndicators.add_rsi(df)
        
        # 波动率指标
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_atr(df)
        
        # 成交量指标
        df = TechnicalIndicators.add_volume_features(df)
        
        return df
    
    @staticmethod
    def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """添加基础价格特征"""
        # 收益率
        df['returns'] = df['close'].pct_change()
        
        # 价格变化
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        
        # 高低点范围
        df['high_low_range'] = df['high'] - df['low']
        df['high_low_range_pct'] = (df['high'] - df['low']) / df['low'] * 100
        
        # 影线特征
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        
        return df
    
    @staticmethod
    def add_ema_features(df: pd.DataFrame, periods: List[int] = [9, 21, 55]) -> pd.DataFrame:
        """添加EMA特征"""
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            # EMA与价格的偏离
            df[f'ema_{period}_dist'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}'] * 100
        
        # EMA交叉信号
        if len(periods) >= 2:
            df['ema_cross'] = np.where(
                df[f'ema_{periods[0]}'] > df[f'ema_{periods[1]}'], 1, -1
            )
        
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """添加MACD指标"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # MACD交叉信号
        df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, periods: List[int] = [7, 14, 21]) -> pd.DataFrame:
        """添加RSI指标"""
        for period in periods:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # RSI超买超卖信号
            df[f'rsi_{period}_signal'] = np.where(
                df[f'rsi_{period}'] > 70, -1,
                np.where(df[f'rsi_{period}'] < 30, 1, 0)
            )
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """添加布林带指标"""
        df[f'bb_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        
        df[f'bb_upper'] = df[f'bb_middle'] + (bb_std * std_dev)
        df[f'bb_lower'] = df[f'bb_middle'] - (bb_std * std_dev)
        
        # 布林带宽度
        df[f'bb_width'] = (df[f'bb_upper'] - df[f'bb_lower']) / df[f'bb_middle']
        
        # 价格在布林带中的位置 (%B)
        df[f'bb_position'] = (df['close'] - df[f'bb_lower']) / (df[f'bb_upper'] - df[f'bb_lower'])
        
        # 突破信号
        df[f'bb_breakout'] = np.where(
            df['close'] > df[f'bb_upper'], 1,
            np.where(df['close'] < df[f'bb_lower'], -1, 0)
        )
        
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """添加ATR (平均真实波幅)"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close'] * 100
        
        return df
    
    @staticmethod
    def add_volume_features(df: pd.DataFrame, periods: List[int] = [10, 20]) -> pd.DataFrame:
        """添加成交量特征"""
        # 成交量变化
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        for period in periods:
            df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
        
        # OBV (能量潮)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_ma'] = df['obv'].rolling(window=20).mean()
        
        # 量价关系
        df['volume_price_corr'] = df['volume'].rolling(window=20).corr(df['close'])
        
        return df
    
    @staticmethod
    def create_target(df: pd.DataFrame, lookahead: int = 1) -> pd.DataFrame:
        """
        创建预测目标
        
        Args:
            lookahead: 预测未来多少个周期
            
        Returns:
            target: 1=涨, 0=跌/平
        """
        df = df.copy()
        
        # 未来收益率
        df['future_returns'] = df['close'].shift(-lookahead) / df['close'] - 1
        
        # 目标标签: 1=涨, 0=跌/平
        df['target'] = (df['future_returns'] > 0).astype(int)
        
        return df
    
    @staticmethod
    def get_feature_columns() -> List[str]:
        """获取所有特征列名"""
        return [
            # 价格特征
            'returns', 'price_change_pct', 'high_low_range_pct',
            'upper_shadow', 'lower_shadow', 'body_size',
            
            # EMA特征
            'ema_9_dist', 'ema_21_dist', 'ema_55_dist', 'ema_cross',
            
            # MACD特征
            'macd', 'macd_hist', 'macd_cross',
            
            # RSI特征
            'rsi_7', 'rsi_14', 'rsi_21',
            'rsi_7_signal', 'rsi_14_signal', 'rsi_21_signal',
            
            # 布林带特征
            'bb_width', 'bb_position', 'bb_breakout',
            
            # ATR特征
            'atr_14_pct',
            
            # 成交量特征
            'volume_change', 'volume_ma_ratio', 'volume_ratio_10', 'volume_ratio_20',
            'obv_ma', 'volume_price_corr'
        ]
    
    @staticmethod
    def prepare_features(df: pd.DataFrame) -> tuple:
        """
        准备特征矩阵和目标变量
        
        Returns:
            X: 特征矩阵
            y: 目标变量
            feature_df: 包含所有特征的DataFrame
        """
        # 添加所有技术指标
        df = TechnicalIndicators.add_all_features(df)
        
        # 创建目标
        df = TechnicalIndicators.create_target(df, lookahead=1)
        
        # 获取特征列
        feature_cols = TechnicalIndicators.get_feature_columns()
        
        # 删除包含NaN的行
        df_clean = df.dropna(subset=feature_cols + ['target'])
        
        X = df_clean[feature_cols].values
        y = df_clean['target'].values
        
        return X, y, df_clean


if __name__ == "__main__":
    # 测试
    import sys
    sys.path.append('/home/admin/code/stock')
    from data.fetch_data import BinanceDataFetcher
    
    fetcher = BinanceDataFetcher()
    df = fetcher.load_from_db('ETH/USDT', limit=1000)
    
    if not df.empty:
        X, y, df_features = TechnicalIndicators.prepare_features(df)
        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        print(f"\nFeature columns: {TechnicalIndicators.get_feature_columns()}")
    else:
        print("No data found. Please run fetch_data.py first.")
