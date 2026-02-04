#!/bin/bash

# Crypto Predictor 启动脚本

echo "🚀 启动 Crypto Predictor..."

# 检查Python环境
echo "📦 检查Python依赖..."
cd /home/admin/code/stock
pip install -q -r requirements.txt

# 检查Node环境
echo "📦 检查Node依赖..."
cd /home/admin/code/stock/web
if [ ! -d "node_modules" ]; then
    echo "安装前端依赖..."
    npm install
fi

# 检查模型是否存在
if [ ! -d "/home/admin/code/stock/models/BTC_USDT_latest" ] || [ ! -d "/home/admin/code/stock/models/ETH_USDT_latest" ]; then
    echo "⚠️ 模型不存在，开始训练..."
    cd /home/admin/code/stock
    python train.py
fi

# 启动后端API
echo "🔧 启动后端API服务..."
cd /home/admin/code/stock/api
python -c "import sys; sys.path.append('/home/admin/code/stock'); from data.fetcher import init_database; init_database()"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# 等待后端启动
sleep 3

# 启动前端
echo "🌐 启动前端服务..."
cd /home/admin/code/stock/web
npm start &
WEB_PID=$!

echo ""
echo "✅ 服务已启动!"
echo "📊 前端界面: http://localhost:3000"
echo "🔌 API文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待中断
trap "kill $API_PID $WEB_PID 2>/dev/null; exit" INT
wait
