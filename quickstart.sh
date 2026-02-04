#!/bin/bash

# 快速启动脚本 (不重新下载数据/训练模型)

echo "=========================================="
echo "  Crypto Predictor - 快速启动"
echo "=========================================="

# 启动后端
echo "启动后端服务..."
python api/main.py &
cd web

# 启动前端
echo "启动前端服务..."
npm start
