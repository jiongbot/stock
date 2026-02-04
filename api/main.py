"""
FastAPI 后端服务
"""
import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Dict

# 简单的HTTP服务
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse

sys.path.append('/home/admin/code/stock')
from data.fetch_data import BinanceDataFetcher
from features.indicators import TechnicalIndicators
from models.train import CryptoPredictor


# 全局变量
models_cache = {}
fetcher = None

def load_models():
    """加载已有模型"""
    global models_cache
    
    symbols = ['ETH/USDT', 'BTC/USDT']
    model_types = ['random_forest']
    
    for symbol in symbols:
        for model_type in model_types:
            model_path = f"models/{symbol.replace('/', '_')}_{model_type}.pkl"
            if os.path.exists(model_path):
                try:
                    model = CryptoPredictor(model_type)
                    model.load(model_path)
                    models_cache[f"{symbol}_{model_type}"] = model
                    print(f"Loaded model: {model_path}")
                except Exception as e:
                    print(f"Failed to load model {model_path}: {e}")


class APIHandler(BaseHTTPRequestHandler):
    """简单的HTTP API处理器"""
    
    def log_message(self, format, *args):
        # 简化日志输出
        pass
    
    def _set_headers(self, content_type='application/json'):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _send_json(self, data):
        self._set_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _send_error(self, code, message):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({'error': message}).encode())
    
    def do_OPTIONS(self):
        self._set_headers()
    
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)
        
        if path == '/':
            self._send_json({
                "message": "Crypto Predictor API",
                "version": "1.0.0",
                "endpoints": [
                    "/api/symbols",
                    "/api/klines",
                    "/api/performance"
                ]
            })
        
        elif path == '/api/symbols':
            self._send_json({
                "symbols": ["ETH/USDT", "BTC/USDT"],
                "timeframes": ["15m"]
            })
        
        elif path == '/api/klines':
            symbol = params.get('symbol', [''])[0]
            timeframe = params.get('timeframe', ['15m'])[0]
            limit = int(params.get('limit', ['500'])[0])
            
            if not symbol:
                self._send_error(400, "Missing symbol parameter")
                return
            
            try:
                df = fetcher.load_from_db(symbol, timeframe, limit=limit)
                
                if df.empty:
                    self._send_error(404, "No data found")
                    return
                
                klines = []
                for _, row in df.iterrows():
                    klines.append({
                        "timestamp": int(row['timestamp']),
                        "datetime": datetime.fromtimestamp(row['timestamp']/1000).isoformat(),
                        "open": float(row['open']),
                        "high": float(row['high']),
                        "low": float(row['low']),
                        "close": float(row['close']),
                        "volume": float(row['volume'])
                    })
                
                self._send_json({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "count": len(klines),
                    "data": klines
                })
            except Exception as e:
                self._send_error(500, str(e))
        
        elif path == '/api/performance':
            results = []
            symbol = params.get('symbol', [None])[0]
            
            symbols_to_check = [symbol] if symbol else ['ETH/USDT', 'BTC/USDT']
            
            for sym in symbols_to_check:
                results_path = f"models/{sym.replace('/', '_')}_random_forest_results.json"
                
                if os.path.exists(results_path):
                    with open(results_path, 'r') as f:
                        data = json.load(f)
                    
                    backtest = data.get('simple_backtest', {})
                    results.append({
                        "symbol": sym,
                        "model_type": data.get('model_type', 'random_forest'),
                        "accuracy": backtest.get('accuracy', 0),
                        "precision": backtest.get('precision', 0),
                        "recall": backtest.get('recall', 0),
                        "f1_score": backtest.get('f1', 0),
                        "total_predictions": backtest.get('total_predictions', 0)
                    })
            
            self._send_json({"performance": results})
        
        else:
            self._send_error(404, "Not found")
    
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            body = json.loads(post_data.decode())
        except:
            body = {}
        
        if path == '/api/predict':
            symbol = body.get('symbol')
            timeframe = body.get('timeframe', '15m')
            
            if not symbol:
                self._send_error(400, "Missing symbol")
                return
            
            model_key = f"{symbol}_random_forest"
            if model_key not in models_cache:
                self._send_error(404, f"Model not found for {symbol}")
                return
            
            model = models_cache[model_key]
            
            try:
                df = fetcher.load_from_db(symbol, timeframe, limit=100)
                
                if df.empty or len(df) < 50:
                    self._send_error(404, "Insufficient data for prediction")
                    return
                
                df_features = TechnicalIndicators.add_all_features(df)
                feature_cols = TechnicalIndicators.get_feature_columns()
                
                latest_features = df_features[feature_cols].iloc[-1:].values
                
                prediction = model.predict(latest_features)[0]
                probabilities = model.predict(latest_features, return_proba=True)[0]
                
                prob_up = probabilities[1]
                prob_down = probabilities[0]
                max_prob = max(prob_up, prob_down)
                
                if max_prob >= 0.7:
                    confidence = "high"
                elif max_prob >= 0.6:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                current_price = df['close'].iloc[-1]
                latest_timestamp = int(df['timestamp'].iloc[-1])
                
                self._send_json({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": latest_timestamp,
                    "datetime": datetime.fromtimestamp(latest_timestamp/1000).isoformat(),
                    "current_price": round(current_price, 2),
                    "prediction": int(prediction),
                    "probability": round(float(max_prob), 4),
                    "confidence": confidence,
                    "model_accuracy": round(model.training_info.get('test_accuracy', 0), 4)
                })
                
            except Exception as e:
                self._send_error(500, f"Prediction error: {str(e)}")
        
        else:
            self._send_error(404, "Not found")


def start_server(host='0.0.0.0', port=8000):
    """启动服务器"""
    global fetcher
    
    print("Initializing API...")
    fetcher = BinanceDataFetcher()
    load_models()
    
    server = HTTPServer((host, port), APIHandler)
    print(f"Server running at http://{host}:{port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    start_server()
