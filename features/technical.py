"""
特征工程模块 - 计算各种技术指标
"""
import pandas as pd
import numpy as np

def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """计算布林带"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_ema(prices, periods):
    """计算EMA"""
    return prices.ewm(span=periods).mean()

def calculate_atr(high, low, close, period=14):
    """计算ATR (平均真实波幅)"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_obv(close, volume):
    """计算OBV (能量潮)"""
    obv = (np.sign(close.diff()) * volume).cumsum()
    return obv

def calculate_vwap(high, low, close, volume):
    """计算VWAP (成交量加权平均价)"""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """计算随机指标"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_adx(high, low, close, period=14):
    """计算ADX (平均趋向指数)"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di

def calculate_cci(high, low, close, period=20):
    """计算CCI (商品通道指数)"""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return cci

def calculate_momentum(close, period=10):
    """计算动量指标"""
    return close.diff(period)

def calculate_roc(close, period=10):
    """计算变化率"""
    return close.pct_change(period) * 100

def calculate_williams_r(high, low, close, period=14):
    """计算威廉指标"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r

def calculate_features(df):
    """
    计算所有特征
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    
    Returns:
        DataFrame with all features
    """
    df = df.copy()
    
    # 基础价格特征
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # 价格位置特征
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low'])
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low'])
    
    # RSI
    df['rsi_6'] = calculate_rsi(df['close'], 6)
    df['rsi_12'] = calculate_rsi(df['close'], 12)
    df['rsi_24'] = calculate_rsi(df['close'], 24)
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_histogram'] = calculate_macd(df['close'])
    
    # 布林带
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # EMA
    df['ema_9'] = calculate_ema(df['close'], 9)
    df['ema_21'] = calculate_ema(df['close'], 21)
    df['ema_55'] = calculate_ema(df['close'], 55)
    df['ema_144'] = calculate_ema(df['close'], 144)
    
    # EMA交叉
    df['ema_9_21_cross'] = (df['ema_9'] > df['ema_21']).astype(int)
    df['ema_21_55_cross'] = (df['ema_21'] > df['ema_55']).astype(int)
    
    # ATR
    df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'])
    df['atr_ratio'] = df['atr_14'] / df['close']
    
    # 成交量特征
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    df['volume_change'] = df['volume'].pct_change()
    
    # OBV
    df['obv'] = calculate_obv(df['close'], df['volume'])
    df['obv_ema'] = calculate_ema(df['obv'], 20)
    
    # VWAP
    df['vwap'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
    
    # 随机指标
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])
    
    # ADX
    df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df['high'], df['low'], df['close'])
    df['di_diff'] = abs(df['plus_di'] - df['minus_di'])
    
    # CCI
    df['cci'] = calculate_cci(df['high'], df['low'], df['close'])
    
    # 动量
    df['momentum_10'] = calculate_momentum(df['close'], 10)
    df['roc_10'] = calculate_roc(df['close'], 10)
    
    # 威廉指标
    df['williams_r'] = calculate_williams_r(df['high'], df['low'], df['close'])
    
    # 波动率
    df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(96)  # 日波动率 (96个15分钟)
    
    # 价格变化特征
    for period in [1, 3, 5, 10]:
        df[f'future_return_{period}'] = df['close'].shift(-period) / df['close'] - 1
    
    # 目标变量: 下一周期涨跌 (1=涨, 0=跌/平)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df

def get_feature_columns():
    """获取特征列名列表"""
    return [
        'returns', 'log_returns', 'price_position', 'body_size', 'upper_shadow', 'lower_shadow',
        'rsi_6', 'rsi_12', 'rsi_24', 'macd', 'macd_signal', 'macd_histogram',
        'bb_position', 'bb_width', 'ema_9_21_cross', 'ema_21_55_cross',
        'atr_ratio', 'volume_ratio', 'volume_change', 'vwap_distance',
        'stoch_k', 'stoch_d', 'adx', 'di_diff', 'cci', 'momentum_10', 'roc_10',
        'williams_r', 'volatility_20'
    ]

if __name__ == '__main__':
    # 测试特征计算
    import sys
    sys.path.append('/home/admin/code/stock')
    from data.fetcher import get_klines_from_db
    
    df = get_klines_from_db('BTC/USDT', limit=1000)
    if not df.empty:
        df_features = calculate_features(df)
        print(f"特征计算完成，共 {len(df_features)} 行")
        print(f"特征数量: {len(get_feature_columns())}")
        print(df_features[get_feature_columns()].describe())
