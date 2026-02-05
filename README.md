# Crypto Predictor - 加密货币价格预测系统

基于机器学习的 ETH/BTC 价格趋势预测系统，支持多时间周期分析和实时预测。

## 功能特性

- **多时间周期支持**: 15分钟 / 1小时 / 4小时 / 1天
- **50+ 技术指标**: RSI, MACD, 布林带, EMA, VWAP, 一目均衡表等
- **集成学习**: RandomForest + GradientBoosting + LogisticRegression
- **实时预测**: FastAPI 后端 + React 前端
- **自动训练**: 定时任务优化模型参数

## 快速开始

### 配置环境变量（可选）

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置（可选）
vim .env
```

支持的环境变量：
- `STOCK_PROJECT_DIR` - 项目根目录
- `STOCK_DATA_DIR` - 数据目录
- `STOCK_DB_PATH` - 数据库路径
- `STOCK_MODELS_DIR` - 模型保存目录
- `STOCK_TARGET_ACCURACY` - 目标准确率（默认0.85）

### 安装依赖

```bash
cd /home/admin/code/stock
pip3 install -r requirements.txt
```

### 下载数据

```bash
python3 -c "from data.fetcher import download_historical_data; download_historical_data('BTC/USDT', days=365)"
python3 -c "from data.fetcher import download_historical_data; download_historical_data('ETH/USDT', days=365)"
```

### 训练模型

```bash
# 基础训练
python3 train.py --symbol BTC/USDT --timeframe 1h --lookback 20

# 或运行自动训练（尝试多种参数组合）
./training_task.sh
```

### 启动服务

```bash
# 一键启动
./start.sh

# 或分别启动
python3 api/main.py          # 后端 http://localhost:8000
cd web && npm start          # 前端 http://localhost:3000
```

## 项目结构

```
stock/
├── data/
│   ├── fetcher.py           # Binance API 数据获取
│   └── crypto_data.db       # SQLite 数据库
├── features/
│   ├── technical.py         # 基础技术指标 (29个)
│   └── enhanced.py          # 高级特征 (50+指标)
├── models/
│   ├── predictor.py         # 预测模型封装
│   └── *_latest/            # 训练好的模型
├── api/
│   └── main.py              # FastAPI 服务
├── web/                     # React 前端
├── train.py                 # 模型训练脚本 ⭐
├── backtest.py              # 回测框架
├── training_task.sh         # 自动训练任务
└── test_pipeline.py         # 数据流程测试
```

## API 接口

| 接口 | 说明 |
|:---|:---|
| `GET /api/health` | 健康检查 |
| `GET /api/predict/{symbol}` | 获取预测 (BTC-USDT, ETH-USDT) |
| `GET /api/history/{symbol}` | 历史K线数据 |
| `GET /api/performance` | 模型性能指标 |
| `GET /api/indicators/{symbol}` | 技术指标 |

## 核心算法

### 特征工程
- **价格特征**: 收益率、对数收益率、价格位置
- **动量指标**: RSI(6/12/24), MACD, 随机指标, 威廉指标
- **趋势指标**: EMA(9/21/55/144), ADX, 一目均衡表
- **波动率**: ATR, 布林带, 历史波动率
- **成交量**: OBV, VWAP, MFI, 成交量比率
- **统计特征**: 偏度, 峰度, 斐波那契回撤

### 模型架构
- **数据预处理**: 时间序列交叉验证, 标准化
- **基学习器**: RandomForest, GradientBoosting, LogisticRegression
- **集成策略**: Soft Voting (概率加权)
- **超参数优化**: GridSearchCV

## 性能目标

| 指标 | 目标 | 当前 |
|:---|:---|:---|
| 预测准确率 | ≥85% | ~53% |
| 预测延迟 | <1秒 | <100ms |
| 支持交易对 | BTC, ETH | ✅ |

> ⚠️ **风险提示**: 短期价格预测极其困难，当前准确率接近随机。仅供学习研究，不构成投资建议。

## 开发计划

- [ ] 添加更多数据源 (订单簿, 链上数据)
- [ ] 尝试深度学习模型 (LSTM, Transformer)
- [ ] 优化特征选择
- [ ] 添加模型解释性 (SHAP)

## 技术栈

- **后端**: Python 3.8+, FastAPI, SQLite
- **前端**: React, Recharts
- **ML**: scikit-learn, pandas, numpy
- **数据**: CCXT, Binance API

## 许可证

MIT License
