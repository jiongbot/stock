# Crypto Predictor PRD

## 项目概述
构建一个基于 Web 的 ETH/BTC 15分钟K线预测系统，通过技术分析指标预测下一周期涨跌，目标正确率 85%+。

## 技术栈
- **后端**: Python + FastAPI + CCXT (获取K线数据)
- **前端**: React + TypeScript + Recharts (图表展示)
- **数据库**: SQLite (存储历史K线和预测记录)
- **ML**: scikit-learn / XGBoost (分类模型)

## 核心功能模块

### 1. 数据层 (data/)
- [ ] 交易所API封装 (Binance/OKX)
- [ ] K线数据获取与存储
- [ ] 历史数据批量下载

### 2. 特征工程 (features/)
- [ ] 技术指标计算 (RSI, MACD, Bollinger, EMA, Volume等)
- [ ] 特征标准化/归一化
- [ ] 特征重要性分析

### 3. 模型层 (models/)
- [ ] 多模型对比 (RandomForest, XGBoost, LSTM)
- [ ] 回测框架
- [ ] 超参数优化

### 4. 预测服务 (api/)
- [ ] REST API 预测接口
- [ ] 实时数据更新
- [ ] 预测结果存储

### 5. 前端界面 (web/)
- [ ] K线图表 + 预测标记
- [ ] 模型性能仪表盘
- [ ] 回测结果可视化

## 执行计划

### Phase 1: 基础架构 (30min)
1. 创建项目结构
2. 配置依赖 (requirements.txt, package.json)
3. 搭建 FastAPI 基础服务

### Phase 2: 数据获取 (45min)
1. 实现交易所API客户端
2. 下载ETH/BTC历史15分钟K线 (至少6个月)
3. 数据库存储设计

### Phase 3: 特征工程 (60min)
1. 实现核心技术指标
2. 构建特征矩阵
3. 标签生成 (下一周期涨跌)

### Phase 4: 模型训练 (90min)
1. 训练集/测试集划分
2. 多模型训练与对比
3. 超参数调优
4. 回测验证

### Phase 5: Web界面 (60min)
1. React项目搭建
2. K线图表组件
3. 预测展示面板
4. 性能仪表盘

### Phase 6: 集成与验证 (45min)
1. 前后端联调
2. 历史数据回测验证
3. 正确率达标检查

## 验收标准
- [ ] 历史回测正确率 >= 85%
- [ ] Web界面可实时查看预测
- [ ] 支持ETH和BTC两个交易对
- [ ] 预测延迟 < 1秒
