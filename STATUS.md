# Crypto Predictor 项目状态报告

## 项目结构

```
/home/admin/code/stock/
├── PRD.md                    # 产品需求文档
├── README.md                 # 项目说明
├── requirements.txt          # Python依赖
├── start.sh                  # 一键启动脚本
├── train_simple.py           # 简化训练脚本
├── train_improved.py         # 改进版训练脚本
├── backtest.py               # 回测框架
├── data/
│   ├── fetcher.py            # 数据获取模块 (Binance API)
│   └── crypto_data.db        # SQLite数据库
├── features/
│   └── technical.py          # 技术指标计算 (RSI, MACD, Bollinger等)
├── models/
│   ├── predictor.py          # 机器学习模型
│   ├── BTC_USDT_latest/      # BTC模型
│   └── ETH_USDT_latest/      # ETH模型
├── api/
│   └── main.py               # FastAPI后端服务
└── web/                      # React前端
    ├── package.json
    ├── public/
    └── src/
        ├── App.js
        ├── components/
        │   ├── PriceChart.js
        │   ├── PredictionPanel.js
        │   └── PerformanceDashboard.js
        └── services/
            └── api.js
```

## 当前状态

### 已完成 ✅
1. **数据获取**: 从Binance获取BTC/ETH的15分钟K线数据
2. **特征工程**: 29+技术指标 (RSI, MACD, Bollinger, EMA, Volume等)
3. **模型训练**: RandomForest分类器
4. **后端API**: FastAPI服务 (预测、历史数据、性能指标接口)
5. **前端界面**: React + Lightweight Charts
6. **回测框架**: 完整的回测和性能评估

### 当前准确率 ⚠️
- **BTC/USDT**: ~50% (接近随机)
- **ETH/USDT**: ~48% (接近随机)

## 为什么准确率不达标？

短期价格预测(15分钟)是**极其困难**的，原因：

1. **市场效率**: 短期价格变动接近随机游走
2. **噪声主导**: 15分钟K线包含大量噪声
3. **特征局限**: 技术指标滞后于价格
4. **数据量**: 需要更大量历史数据(1年+)

## 如何运行项目

### 1. 安装依赖
```bash
cd /home/admin/code/stock
pip3 install --user pandas numpy scikit-learn requests
```

### 2. 下载数据
```bash
python3 -c "from data.fetcher import download_historical_data; download_historical_data('BTC/USDT', days=180)"
```

### 3. 训练模型
```bash
python3 train_simple.py
```

### 4. 启动服务
```bash
./start.sh
```

### 5. 访问界面
- 前端: http://localhost:3000
- API文档: http://localhost:8000/docs

## 改进建议

要达到85%准确率，需要：

1. **更长周期预测**: 改为1小时或4小时K线
2. **更多数据源**: 订单簿、资金费率、链上数据
3. **深度学习**: 使用LSTM/Transformer模型
4. **集成学习**: 组合多个模型预测
5. **特征工程**: 添加更多高级特征

## API端点

- `GET /api/health` - 健康检查
- `GET /api/predict/{symbol}` - 获取预测 (BTC-USDT, ETH-USDT)
- `GET /api/history/{symbol}` - 历史K线数据
- `GET /api/performance` - 模型性能指标
- `GET /api/indicators/{symbol}` - 技术指标

## 注意事项

⚠️ **风险提示**: 此项目仅供学习研究，不构成投资建议。加密货币交易风险极高，模型预测准确率无法保证盈利。
