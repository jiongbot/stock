"""
简化版HTTP API服务 - 使用标准库
"""
import sys
import os
import json
import sqlite3
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

sys.path.append('/home/admin/code/stock')
from data.fetcher import get_klines_from_db, init_database
from features.technical import calculate_features, get_feature_columns

# 尝试加载模型
try:
    from models.predictor import load_model
    MODELS_AVAILABLE = True
except:
    MODELS_AVAILABLE = False

# 全局缓存
models_cache = {}

def get_model(symbol):
    if not MODELS_AVAILABLE:
        return None, None, None
    if symbol not in models_cache:
        model, scaler, info = load_model(symbol)
        if model:
            models_cache[symbol] = (model, scaler, info)
    return models_cache.get(symbol, (None, None, None))

def make_prediction(symbol):
    """生成预测"""
    import numpy as np
    
    model, scaler, model_info = get_model(symbol)
    if model is None:
        return None
    
    df = get_klines_from_db(symbol, limit=500)
    if df.empty or len(df) < 50:
        return None
    
    df_features = calculate_features(df)
    df_features = df_features.dropna()
    
    if len(df_features) < 10:
        return None
    
    feature_cols = get_feature_columns()
    lookback = 10
    
    latest_idx = len(df_features) - 1
    features = []
    for j in range(lookback):
        row_features = df_features.iloc[latest_idx - lookback + j][feature_cols].values
        features.extend(row_features)
    
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    prediction_proba = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
    
    return {
        "direction": "up" if prediction == 1 else "down",
        "confidence": float(max(prediction_proba)),
        "probability_up": float(prediction_proba[1]),
        "probability_down": float(prediction_proba[0]),
        "current_price": float(df_features.iloc[-1]['close']),
        "model_accuracy": model_info.get('accuracy', 0) if model_info else 0
    }

class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # 健康检查
        if path == '/api/health':
            self.send_json({"status": "ok", "timestamp": datetime.now().isoformat()})
            return
        
        # 获取历史数据
        if path.startswith('/api/history/'):
            symbol = path.split('/')[-1].replace('-', '/')
            df = get_klines_from_db(symbol, limit=500)
            
            if df.empty:
                self.send_json({"error": "No data"}, 404)
                return
            
            data = []
            for _, row in df.iterrows():
                data.append({
                    "timestamp": int(row['timestamp']),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume'])
                })
            
            self.send_json({"symbol": symbol, "count": len(data), "data": data})
            return
        
        # 获取预测
        if path.startswith('/api/predict/'):
            symbol = path.split('/')[-1].replace('-', '/')
            prediction = make_prediction(symbol)
            
            if prediction is None:
                self.send_json({"error": "Model not available"}, 404)
                return
            
            self.send_json({
                "symbol": symbol,
                "timestamp": int(datetime.now().timestamp() * 1000),
                "prediction": prediction
            })
            return
        
        # 获取性能指标
        if path == '/api/performance':
            performance = {}
            for symbol in ["BTC/USDT", "ETH/USDT"]:
                model, scaler, model_info = get_model(symbol)
                if model_info:
                    performance[symbol] = {
                        "accuracy": model_info.get('accuracy', 0),
                        "precision": model_info.get('precision', 0),
                        "recall": model_info.get('recall', 0)
                    }
            self.send_json({"models": performance})
            return
        
        # 前端静态文件
        if path == '/' or path == '/index.html':
            self.send_html(HTML_PAGE)
            return
        
        self.send_json({"error": "Not found"}, 404)
    
    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def send_html(self, html):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

# 简单的HTML页面
HTML_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Predictor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 32px;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .symbol-selector {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 20px;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
            color: white;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
        }
        .btn:hover, .btn.active {
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        }
        .prediction-box {
            text-align: center;
            padding: 30px;
            border-radius: 12px;
            margin: 20px 0;
        }
        .prediction-box.up {
            background: linear-gradient(135deg, rgba(38, 166, 154, 0.2), rgba(38, 166, 154, 0.05));
            border: 2px solid #26a69a;
        }
        .prediction-box.down {
            background: linear-gradient(135deg, rgba(239, 83, 80, 0.2), rgba(239, 83, 80, 0.05));
            border: 2px solid #ef5350;
        }
        .direction {
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .direction.up { color: #26a69a; }
        .direction.down { color: #ef5350; }
        .confidence {
            font-size: 48px;
            font-weight: 800;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .price {
            font-size: 24px;
            margin: 20px 0;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .stat-item {
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 20px;
            font-weight: 600;
        }
        .stat-label {
            font-size: 12px;
            color: rgba(255,255,255,0.6);
            margin-top: 5px;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: rgba(255,255,255,0.5);
        }
        .error {
            text-align: center;
            padding: 40px;
            color: #ef5350;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔮 Crypto Predictor</h1>
        <p>BTC/ETH 15分钟K线预测</p>
    </div>
    
    <div class="container">
        <div class="card">
            <div class="symbol-selector">
                <button class="btn active" onclick="selectSymbol('BTC-USDT')">₿ BTC/USDT</button>
                <button class="btn" onclick="selectSymbol('ETH-USDT')">Ξ ETH/USDT</button>
            </div>
            
            <div id="prediction">
                <div class="loading">加载中...⏳</div>
            </div>
        </div>
        
        <div class="card">
            <h3>📊 模型性能</h3>
            <div id="performance">
                <div class="loading">加载中...⏳</div>
            </div>
        </div>
    </div>
    
    <script>
        let currentSymbol = 'BTC-USDT';
        
        async function fetchPrediction() {
            try {
                const response = await fetch(`/api/predict/${currentSymbol}`);
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('prediction').innerHTML = 
                        `<div class="error">${data.error}</div>`;
                    return;
                }
                
                const pred = data.prediction;
                const isUp = pred.direction === 'up';
                
                document.getElementById('prediction').innerHTML = `
                    <div class="prediction-box ${isUp ? 'up' : 'down'}">
                        <div class="direction ${isUp ? 'up' : 'down'}">
                            ${isUp ? '📈 上涨' : '📉 下跌'}
                        </div>
                        <div class="confidence">${(pred.confidence * 100).toFixed(1)}%</div>
                        <div style="color: rgba(255,255,255,0.6); margin-top: 10px;">置信度</div>
                        
                        <div class="price">当前价格: $${pred.current_price.toLocaleString(undefined, {minimumFractionDigits: 2})}</div>
                        
                        <div class="stats">
                            <div class="stat-item">
                                <div class="stat-value" style="color: #26a69a">${(pred.probability_up * 100).toFixed(1)}%</div>
                                <div class="stat-label">上涨概率</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value" style="color: #ef5350">${(pred.probability_down * 100).toFixed(1)}%</div>
                                <div class="stat-label">下跌概率</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value" style="color: ${pred.model_accuracy >= 0.5 ? '#26a69a' : '#ffa726'}">${(pred.model_accuracy * 100).toFixed(1)}%</div>
                                <div class="stat-label">模型准确率</div>
                            </div>
                        </div>
                    </div>
                `;
            } catch (e) {
                document.getElementById('prediction').innerHTML = 
                    `<div class="error">加载失败: ${e.message}</div>`;
            }
        }
        
        async function fetchPerformance() {
            try {
                const response = await fetch('/api/performance');
                const data = await response.json();
                
                let html = '<div class="stats">';
                for (const [symbol, stats] of Object.entries(data.models)) {
                    html += `
                        <div class="stat-item">
                            <div class="stat-label">${symbol}</div>
                            <div class="stat-value" style="color: ${stats.accuracy >= 0.5 ? '#26a69a' : '#ffa726'}">
                                ${(stats.accuracy * 100).toFixed(1)}%
                            </div>
                            <div class="stat-label">准确率</div>
                        </div>
                    `;
                }
                html += '</div>';
                document.getElementById('performance').innerHTML = html;
            } catch (e) {
                document.getElementById('performance').innerHTML = 
                    `<div class="error">加载失败</div>`;
            }
        }
        
        function selectSymbol(symbol) {
            currentSymbol = symbol;
            document.querySelectorAll('.btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            fetchPrediction();
        }
        
        // 初始化
        fetchPrediction();
        fetchPerformance();
        
        // 每分钟刷新
        setInterval(fetchPrediction, 60000);
    </script>
</body>
</html>
'''

def run_server(port=8000):
    init_database()
    server = HTTPServer(('0.0.0.0', port), APIHandler)
    print(f"API服务启动在 http://0.0.0.0:{port}")
    server.serve_forever()

if __name__ == '__main__':
    run_server()
