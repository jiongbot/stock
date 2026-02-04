"""
回测框架 - 验证模型在历史数据上的表现
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import sys
sys.path.append('/home/admin/code/stock')

from data.fetcher import get_klines_from_db, get_db_connection
from features.technical import calculate_features, get_feature_columns
from models.predictor import load_model, prepare_data

class Backtester:
    def __init__(self, symbol='BTC/USDT', initial_capital=10000, lookback=10):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.lookback = lookback
        self.feature_cols = get_feature_columns()
        
        # 加载模型
        self.model, self.scaler, self.model_info = load_model(symbol)
        if self.model is None:
            raise ValueError(f"未找到 {symbol} 的模型，请先训练模型")
        
        print(f"加载模型: {self.model_info.get('model_name', 'Unknown')}")
        print(f"模型准确率: {self.model_info.get('accuracy', 0):.4f}")
    
    def prepare_sequence(self, df, idx):
        """准备单个序列特征"""
        features = []
        for j in range(self.lookback):
            row_features = df.iloc[idx - self.lookback + j][self.feature_cols].values
            features.extend(row_features)
        return np.array(features).reshape(1, -1)
    
    def run_backtest(self, df=None, start_date=None, end_date=None, fee_rate=0.001):
        """
        运行回测
        
        Args:
            df: 特征数据DataFrame，如果为None则从数据库加载
            start_date: 回测开始日期
            end_date: 回测结束日期
            fee_rate: 交易手续费率
        
        Returns:
            backtest_results: 回测结果字典
        """
        if df is None:
            df = get_klines_from_db(self.symbol, limit=10000)
        
        df = calculate_features(df)
        df = df.dropna()
        
        # 过滤日期范围
        if start_date:
            df = df[df['timestamp'] >= int(start_date.timestamp() * 1000)]
        if end_date:
            df = df[df['timestamp'] <= int(end_date.timestamp() * 1000)]
        
        if len(df) < self.lookback + 100:
            raise ValueError(f"数据量不足: {len(df)} 行")
        
        print(f"回测数据范围: {len(df)} 条K线")
        print(f"时间范围: {datetime.fromtimestamp(df['timestamp'].iloc[0]/1000)} - {datetime.fromtimestamp(df['timestamp'].iloc[-1]/1000)}")
        
        # 回测状态
        capital = self.initial_capital
        position = 0  # 0=空仓, 1=持仓
        trades = []
        equity_curve = []
        predictions = []
        actuals = []
        
        # 遍历数据
        for i in range(self.lookback, len(df) - 1):
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            # 准备特征
            X = self.prepare_sequence(df, i)
            X_scaled = self.scaler.transform(X)
            
            # 预测
            pred = self.model.predict(X_scaled)[0]
            pred_proba = self.model.predict_proba(X_scaled)[0] if hasattr(self.model, 'predict_proba') else [0.5, 0.5]
            confidence = max(pred_proba)
            
            # 实际结果
            actual = 1 if next_row['close'] > current_row['close'] else 0
            
            predictions.append(pred)
            actuals.append(actual)
            
            # 交易逻辑 (只在高置信度时交易)
            if confidence >= 0.55:  # 置信度阈值
                if pred == 1 and position == 0:  # 预测涨，买入
                    position = 1
                    entry_price = current_row['close']
                    trades.append({
                        'type': 'buy',
                        'timestamp': current_row['timestamp'],
                        'price': entry_price,
                        'confidence': confidence
                    })
                elif pred == 0 and position == 1:  # 预测跌，卖出
                    exit_price = current_row['close']
                    pnl = (exit_price - entry_price) / entry_price - fee_rate * 2
                    capital *= (1 + pnl)
                    position = 0
                    trades.append({
                        'type': 'sell',
                        'timestamp': current_row['timestamp'],
                        'price': exit_price,
                        'pnl': pnl,
                        'confidence': confidence
                    })
            
            # 计算当前权益
            if position == 1:
                current_equity = capital * (current_row['close'] / entry_price)
            else:
                current_equity = capital
            
            equity_curve.append({
                'timestamp': current_row['timestamp'],
                'equity': current_equity,
                'price': current_row['close'],
                'prediction': pred,
                'actual': actual,
                'confidence': confidence
            })
        
        # 如果最后还有持仓，平仓
        if position == 1:
            exit_price = df.iloc[-1]['close']
            pnl = (exit_price - entry_price) / entry_price - fee_rate * 2
            capital *= (1 + pnl)
            trades.append({
                'type': 'sell',
                'timestamp': df.iloc[-1]['timestamp'],
                'price': exit_price,
                'pnl': pnl,
                'confidence': 0
            })
        
        # 计算回测指标
        results = self.calculate_metrics(predictions, actuals, trades, equity_curve)
        results['symbol'] = self.symbol
        results['trades'] = trades
        results['equity_curve'] = equity_curve
        
        return results
    
    def calculate_metrics(self, predictions, actuals, trades, equity_curve):
        """计算回测指标"""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # 预测准确率
        correct = (predictions == actuals).sum()
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0
        
        # 上涨预测准确率
        up_mask = actuals == 1
        up_correct = ((predictions == 1) & up_mask).sum()
        up_total = up_mask.sum()
        up_accuracy = up_correct / up_total if up_total > 0 else 0
        
        # 下跌预测准确率
        down_mask = actuals == 0
        down_correct = ((predictions == 0) & down_mask).sum()
        down_total = down_mask.sum()
        down_accuracy = down_correct / down_total if down_total > 0 else 0
        
        # 交易统计
        buy_trades = [t for t in trades if t['type'] == 'buy']
        sell_trades = [t for t in trades if t['type'] == 'sell']
        
        profits = [t['pnl'] for t in sell_trades if 'pnl' in t]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]
        
        # 权益曲线分析
        if equity_curve:
            equity_values = [e['equity'] for e in equity_curve]
            final_equity = equity_values[-1]
            max_equity = max(equity_values)
            min_equity = min(equity_values)
            
            # 计算最大回撤
            peak = equity_values[0]
            max_drawdown = 0
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # 收益率
            total_return = (final_equity - self.initial_capital) / self.initial_capital
        else:
            final_equity = self.initial_capital
            max_drawdown = 0
            total_return = 0
        
        results = {
            'total_predictions': int(total),
            'correct_predictions': int(correct),
            'accuracy': float(accuracy),
            'up_accuracy': float(up_accuracy),
            'down_accuracy': float(down_accuracy),
            'total_trades': len(buy_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(sell_trades) if sell_trades else 0,
            'avg_profit': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf'),
            'initial_capital': self.initial_capital,
            'final_equity': float(final_equity),
            'total_return': float(total_return),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': self.calculate_sharpe(equity_values) if equity_curve else 0
        }
        
        return results
    
    def calculate_sharpe(self, equity_values, risk_free_rate=0.02):
        """计算夏普比率"""
        returns = pd.Series(equity_values).pct_change().dropna()
        if len(returns) < 2 or returns.std() == 0:
            return 0
        
        # 年化 (假设96个15分钟/天，252个交易日)
        annual_return = returns.mean() * 96 * 252
        annual_volatility = returns.std() * np.sqrt(96 * 252)
        
        sharpe = (annual_return - risk_free_rate) / annual_volatility
        return sharpe
    
    def print_report(self, results):
        """打印回测报告"""
        print(f"\n{'='*60}")
        print(f"回测报告 - {results['symbol']}")
        print(f"{'='*60}")
        print(f"\n预测性能:")
        print(f"  总预测次数: {results['total_predictions']}")
        print(f"  正确预测: {results['correct_predictions']}")
        print(f"  总体准确率: {results['accuracy']*100:.2f}%")
        print(f"  上涨预测准确率: {results['up_accuracy']*100:.2f}%")
        print(f"  下跌预测准确率: {results['down_accuracy']*100:.2f}%")
        
        print(f"\n交易统计:")
        print(f"  总交易次数: {results['total_trades']}")
        print(f"  盈利交易: {results['winning_trades']}")
        print(f"  亏损交易: {results['losing_trades']}")
        print(f"  胜率: {results['win_rate']*100:.2f}%")
        print(f"  平均盈利: {results['avg_profit']*100:.2f}%")
        print(f"  平均亏损: {results['avg_loss']*100:.2f}%")
        print(f"  盈亏比: {results['profit_factor']:.2f}")
        
        print(f"\n资金曲线:")
        print(f"  初始资金: ${results['initial_capital']:,.2f}")
        print(f"  最终权益: ${results['final_equity']:,.2f}")
        print(f"  总收益率: {results['total_return']*100:.2f}%")
        print(f"  最大回撤: {results['max_drawdown']*100:.2f}%")
        print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"{'='*60}\n")

def run_full_backtest(symbol='BTC/USDT'):
    """运行完整回测"""
    print(f"开始回测 {symbol}...")
    
    backtester = Backtester(symbol)
    results = backtester.run_backtest()
    backtester.print_report(results)
    
    # 保存结果
    results_to_save = {k: v for k, v in results.items() if k not in ['trades', 'equity_curve']}
    
    output_file = f'/home/admin/code/stock/backtest_results_{symbol.replace("/", "_")}.json'
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"回测结果已保存到: {output_file}")
    
    return results

if __name__ == '__main__':
    # 运行回测
    try:
        results = run_full_backtest('BTC/USDT')
        
        # 检查准确率是否达标
        if results['accuracy'] >= 0.85:
            print(f"✅ 准确率达标: {results['accuracy']*100:.2f}% >= 85%")
        else:
            print(f"❌ 准确率未达标: {results['accuracy']*100:.2f}% < 85%")
            print("建议: 调整特征工程或尝试其他模型")
    except Exception as e:
        print(f"回测失败: {e}")
