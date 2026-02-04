# Crypto Predictor Project

## 项目结构

```
stock/
├── data/               # 数据存储
│   ├── raw/           # 原始K线数据
│   └── processed/     # 特征数据
├── features/          # 特征工程
├── models/            # 机器学习模型
├── api/               # FastAPI后端
├── web/               # React前端
├── backtest.py        # 回测脚本
└── requirements.txt   # Python依赖
```

## 快速开始

### 1. 安装依赖
```bash
cd /home/admin/code/stock
pip install -r requirements.txt
```

### 2. 下载历史数据
```bash
python -c "from data.fetcher import download_historical_data; download_historical_data('BTC/USDT', days=180)"
python -c "from data.fetcher import download_historical_data; download_historical_data('ETH/USDT', days=180)"
```

### 3. 训练模型并回测
```bash
python backtest.py
```

### 4. 启动API服务
```bash
cd api && uvicorn main:app --reload --port 8000
```

### 5. 启动前端
```bash
cd web && npm install && npm start
```

## API端点

- `GET /api/predict/{symbol}` - 获取预测结果 (BTC/USDT 或 ETH/USDT)
- `GET /api/history/{symbol}` - 获取历史K线数据
- `GET /api/performance` - 获取模型性能指标

## 模型性能

回测结果保存在 `backtest_results.json`
