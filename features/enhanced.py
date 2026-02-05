"""
改进的特征工程模块 - 添加更多高级特征
"""
import pandas as pd
import numpy as np
from scipy import stats

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

def calculate_ichimoku(high, low, close):
    """计算一目均衡表指标"""
    # 转换线 (Tenkan-sen): (9-period high + 9-period low)/2
    period9_high = high.rolling(window=9).max()
    period9_low = low.rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # 基准线 (Kijun-sen): (26-period high + 26-period low)/2
    period26_high = high.rolling(window=26).max()
    period26_low = low.rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # 先行带A (Senkou Span A): (转换线 + 基准线)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # 先行带B (Senkou Span B): (52-period high + 52-period low)/2
    period52_high = high.rolling(window=52).max()
    period52_low = low.rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
    
    # 延迟线 (Chikou Span): 收盘价向后移26周期
    chikou_span = close.shift(-26)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def calculate_fibonacci_retracement(high, low, close, period=20):
    """计算斐波那契回撤水平"""
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()
    diff = highest - lowest
    
    # 当前价格在回撤区间的位置
    retracement = (highest - close) / diff
    
    return retracement

def calculate_pivot_points(high, low, close):
    """计算枢轴点"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    
    # 价格在枢轴点的相对位置
    pivot_position = (close - s2) / (r2 - s2)
    
    return pivot_position

def calculate_money_flow_index(high, low, close, volume, period=14):
    """计算资金流量指标 (MFI)"""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    money_flow_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
    signed_money_flow = raw_money_flow * money_flow_sign
    
    positive_flow = signed_money_flow.where(signed_money_flow > 0, 0).rolling(window=period).sum()
    negative_flow = abs(signed_money_flow.where(signed_money_flow < 0, 0)).rolling(window=period).sum()
    
    money_ratio = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi

def calculate_keltner_channels(high, low, close, ema_period=20, atr_period=10):
    """计算肯特纳通道"""
    ema = calculate_ema(close, ema_period)
    atr = calculate_atr(high, low, close, atr_period)
    
    upper_channel = ema + 2 * atr
    lower_channel = ema - 2 * atr
    
    # 价格在通道中的位置
    channel_position = (close - lower_channel) / (upper_channel - lower_channel)
    
    return channel_position

def calculate_features(df, timeframe='1h'):
    """
    计算所有特征
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        timeframe: 时间周期 ('15m', '1h', '4h')
    
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
    
    # 多周期RSI
    for period in [6, 12, 24, 48]:
        df[f'rsi_{period}'] = calculate_rsi(df['close'], period)
    
    # RSI背离
    df['rsi_24_diff'] = df['rsi_24'].diff()
    df['price_diff'] = df['close'].diff()
    df['rsi_divergence'] = np.where(
        (df['price_diff'] > 0) & (df['rsi_24_diff'] < 0), -1,  # 顶背离
        np.where((df['price_diff'] < 0) & (df['rsi_24_diff'] > 0), 1, 0)  # 底背离
    )
    
    # MACD多周期
    for fast, slow, signal in [(12, 26, 9), (6, 13, 5), (24, 52, 18)]:
        df[f'macd_{fast}_{slow}'], df[f'macd_signal_{fast}_{slow}'], df[f'macd_hist_{fast}_{slow}'] = \
            calculate_macd(df['close'], fast, slow, signal)
    
    # 布林带
    for period in [20, 50]:
        df[f'bb_upper_{period}'], df[f'bb_middle_{period}'], df[f'bb_lower_{period}'] = \
            calculate_bollinger_bands(df['close'], period)
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / \
                                       (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_middle_{period}']
    
    # 多周期EMA和交叉
    ema_periods = [9, 21, 55, 144, 233]
    for period in ema_periods:
        df[f'ema_{period}'] = calculate_ema(df['close'], period)
    
    # EMA交叉信号
    df['ema_9_21_cross'] = (df['ema_9'] > df['ema_21']).astype(int)
    df['ema_21_55_cross'] = (df['ema_21'] > df['ema_55']).astype(int)
    df['ema_55_144_cross'] = (df['ema_55'] > df['ema_144']).astype(int)
    
    # EMA距离
    df['ema_9_distance'] = (df['close'] - df['ema_9']) / df['close']
    df['ema_21_distance'] = (df['close'] - df['ema_21']) / df['close']
    
    # ATR和波动率
    for period in [14, 28]:
        df[f'atr_{period}'] = calculate_atr(df['high'], df['low'], df['close'], period)
        df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']
    
    # 历史波动率
    for period in [20, 50]:
        df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
    
    # 成交量特征
    for period in [10, 20, 50]:
        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
    
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    df['volume_change'] = df['volume'].pct_change()
    df['volume_price_trend'] = df['volume'] * df['returns']
    
    # OBV和VWAP
    df['obv'] = calculate_obv(df['close'], df['volume'])
    df['obv_ema'] = calculate_ema(df['obv'], 20)
    df['obv_ratio'] = df['obv'] / df['obv_ema']
    
    df['vwap'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
    
    # 随机指标
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])
    df['stoch_cross'] = (df['stoch_k'] > df['stoch_d']).astype(int)
    
    # ADX和趋势强度
    df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df['high'], df['low'], df['close'])
    df['di_diff'] = abs(df['plus_di'] - df['minus_di'])
    df['trend_direction'] = (df['plus_di'] > df['minus_di']).astype(int)
    
    # CCI
    df['cci'] = calculate_cci(df['high'], df['low'], df['close'])
    df['cci_normalized'] = df['cci'] / 200  # 归一化到 -1 到 1
    
    # 动量和ROC
    for period in [10, 20]:
        df[f'momentum_{period}'] = calculate_momentum(df['close'], period)
        df[f'roc_{period}'] = calculate_roc(df['close'], period)
    
    # 威廉指标
    df['williams_r'] = calculate_williams_r(df['high'], df['low'], df['close'])
    
    # 一目均衡表
    df['tenkan_sen'], df['kijun_sen'], df['senkou_span_a'], df['senkou_span_b'], df['chikou_span'] = \
        calculate_ichimoku(df['high'], df['low'], df['close'])
    df['ichimoku_cloud'] = (df['senkou_span_a'] > df['senkou_span_b']).astype(int)
    
    # 斐波那契回撤
    df['fib_retracement'] = calculate_fibonacci_retracement(df['high'], df['low'], df['close'])
    
    # 枢轴点
    df['pivot_position'] = calculate_pivot_points(df['high'], df['low'], df['close'])
    
    # 资金流量指标
    df['mfi'] = calculate_money_flow_index(df['high'], df['low'], df['close'], df['volume'])
    
    # 肯特纳通道
    df['keltner_position'] = calculate_keltner_channels(df['high'], df['low'], df['close'])
    
    # 统计特征
    for period in [20, 50]:
        df[f'skewness_{period}'] = df['returns'].rolling(window=period).skew()
        df[f'kurtosis_{period}'] = df['returns'].rolling(window=period).kurt()
    
    # 价格形态特征
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
    df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    
    # 趋势特征
    df['uptrend'] = ((df['higher_high'] == 1) & (df['higher_low'] == 1)).astype(int)
    df['downtrend'] = ((df['lower_high'] == 1) & (df['lower_low'] == 1)).astype(int)
    
    # 目标变量: 下一周期涨跌 (根据时间周期调整)
    if timeframe == '15m':
        lookahead = 4  # 1小时 = 4个15分钟
    elif timeframe == '1h':
        lookahead = 4  # 4小时 = 4个1小时
    elif timeframe == '4h':
        lookahead = 6  # 1天 = 6个4小时
    else:
        lookahead = 1
    
    # 使用更大的阈值来过滤噪声
    threshold = 0.001  # 0.1%的变动阈值
    future_return = df['close'].shift(-lookahead) / df['close'] - 1
    df['target'] = np.where(future_return > threshold, 1,
                           np.where(future_return < -threshold, 0, np.nan))
    
    return df

def get_feature_columns():
    """获取特征列名列表 - 改进版"""
    base_features = [
        # 价格特征
        'returns', 'log_returns', 'price_position', 'body_size', 'upper_shadow', 'lower_shadow',
        
        # RSI
        'rsi_6', 'rsi_12', 'rsi_24', 'rsi_48', 'rsi_divergence',
        
        # MACD
        'macd_12_26', 'macd_signal_12_26', 'macd_hist_12_26',
        'macd_6_13', 'macd_signal_6_13', 'macd_hist_6_13',
        
        # 布林带
        'bb_position_20', 'bb_width_20', 'bb_position_50', 'bb_width_50',
        
        # EMA
        'ema_9_21_cross', 'ema_21_55_cross', 'ema_55_144_cross',
        'ema_9_distance', 'ema_21_distance',
        
        # ATR和波动率
        'atr_ratio_14', 'atr_ratio_28', 'volatility_20', 'volatility_50',
        
        # 成交量
        'volume_ratio', 'volume_change', 'volume_price_trend', 'obv_ratio', 'vwap_distance',
        
        # 动量指标
        'stoch_k', 'stoch_d', 'stoch_cross', 'cci_normalized',
        'momentum_10', 'momentum_20', 'roc_10', 'roc_20', 'williams_r',
        
        # 趋势指标
        'adx', 'di_diff', 'trend_direction', 'ichimoku_cloud',
        
        # 其他技术指标
        'fib_retracement', 'pivot_position', 'mfi', 'keltner_position',
        
        # 统计特征
        'skewness_20', 'kurtosis_20',
        
        # 价格形态
        'uptrend', 'downtrend'
    ]
    
    return base_features

if __name__ == '__main__':
    import sys
    sys.path.append('/home/admin/code/stock')
    from data.fetcher import get_klines_from_db
    
    df = get_klines_from_db('BTC/USDT', limit=5000)
    if not df.empty:
        df_features = calculate_features(df, timeframe='1h')
        feature_cols = get_feature_columns()
        print(f"特征计算完成，共 {len(df_features)} 行")
        print(f"特征数量: {len(feature_cols)}")
        print(f"目标分布:\n{df_features['target'].value_counts()}")
