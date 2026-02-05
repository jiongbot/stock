#!/bin/bash
# 训练任务脚本 - 修复版

TARGET_ACCURACY=0.85
LOG_FILE="/home/admin/code/stock/training.log"
RESULTS_FILE="/home/admin/code/stock/best_results.json"
LOCK_FILE="/home/admin/code/stock/training.lock"

cd /home/admin/code/stock

# 防止重复运行
if [ -f "$LOCK_FILE" ]; then
    PID=$(cat "$LOCK_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "$(date): 训练任务已在运行 (PID: $PID)，跳过" >> $LOG_FILE
        exit 0
    else
        rm -f "$LOCK_FILE"
    fi
fi

echo $$ > "$LOCK_FILE"

echo "$(date): ========== 开始训练任务 ==========" >> $LOG_FILE

# 检查数据
python3 << 'EOF'
import sys
sys.path.append('/home/admin/code/stock')
from data.fetcher import get_klines_from_db

for symbol in ['BTC/USDT', 'ETH/USDT']:
    df = get_klines_from_db(symbol, limit=100)
    print(f"{symbol}: {len(df)} 行")
EOF

# 尝试不同参数组合
PARAMS_LIST=(
    "BTC/USDT:1h:20"
    "BTC/USDT:4h:10"
    "ETH/USDT:1h:20"
    "ETH/USDT:4h:10"
)

for PARAMS in "${PARAMS_LIST[@]}"; do
    IFS=':' read -r SYMBOL TIMEFRAME LOOKBACK <<< "$PARAMS"
    
    echo "$(date): 训练 $SYMBOL - $TIMEFRAME - lookback=$LOOKBACK" >> $LOG_FILE
    
    # 运行训练并捕获输出
    OUTPUT=$(python3 train.py --symbol "$SYMBOL" --timeframe "$TIMEFRAME" --lookback "$LOOKBACK" 2>&1)
    EXIT_CODE=$?
    
    echo "$OUTPUT" >> $LOG_FILE
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "$(date): 训练失败 (exit code: $EXIT_CODE)" >> $LOG_FILE
        continue
    fi
    
    # 检查最新结果
    LATEST_RESULT=$(find models -name "results.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_RESULT" ] && [ -f "$LATEST_RESULT" ]; then
        ACCURACY=$(python3 -c "
import json
import sys
try:
    with open('$LATEST_RESULT', 'r') as f:
        data = json.load(f)
        print(data.get('accuracy', 0))
except:
    print(0)
" 2>/dev/null)
        
        echo "$(date): $SYMBOL $TIMEFRAME 准确率: $ACCURACY" >> $LOG_FILE
        
        # 检查是否达到目标
        if python3 -c "import sys; sys.exit(0 if float('$ACCURACY') >= $TARGET_ACCURACY else 1)" 2>/dev/null; then
            echo "$(date): 达到目标准确率! $ACCURACY" >> $LOG_FILE
            echo '{"status": "success", "accuracy": '$ACCURACY', "config": "'$SYMBOL'-'$TIMEFRAME'-'$LOOKBACK'"}' > $RESULTS_FILE
            rm -f "$LOCK_FILE"
            exit 0
        fi
    fi
done

echo "$(date): 本次训练完成，未达到目标准确率" >> $LOG_FILE

# 检查运行次数
RUN_COUNT=$(grep "开始训练任务" $LOG_FILE 2>/dev/null | wc -l)
if [ "$RUN_COUNT" -ge 50 ]; then
    echo "$(date): 达到最大运行次数 ($RUN_COUNT)，停止任务" >> $LOG_FILE
    echo '{"status": "stopped", "reason": "max_runs_reached", "runs": '$RUN_COUNT'}' > $RESULTS_FILE
    rm -f "$LOCK_FILE"
    exit 1
fi

rm -f "$LOCK_FILE"
exit 0
