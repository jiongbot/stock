#!/bin/bash
# 训练任务脚本 - 尝试不同参数直到达到目标准确率

TARGET_ACCURACY=0.85
LOG_FILE="/home/admin/code/stock/training.log"
RESULTS_FILE="/home/admin/code/stock/best_results.json"

cd /home/admin/code/stock

echo "$(date): 开始训练任务" >> $LOG_FILE

# 尝试不同参数组合
for TIMEFRAME in "1h" "4h"; do
    for LOOKBACK in 10 20 30; do
        for SYMBOL in "BTC/USDT" "ETH/USDT"; do
            echo "$(date): 训练 $SYMBOL - $TIMEFRAME - lookback=$LOOKBACK" >> $LOG_FILE
            
            # 运行训练
            python3 train_v2.py --symbol $SYMBOL --timeframe $TIMEFRAME --lookback $LOOKBACK 2>&1 | tee -a $LOG_FILE
            
            # 检查最新结果
            LATEST_RESULT=$(find models -name "results.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
            if [ -n "$LATEST_RESULT" ]; then
                ACCURACY=$(python3 -c "import json; print(json.load(open('$LATEST_RESULT'))['accuracy'])" 2>/dev/null || echo "0")
                echo "$(date): $SYMBOL $TIMEFRAME 准确率: $ACCURACY" >> $LOG_FILE
                
                # 如果达到目标，记录并退出
                if (( $(echo "$ACCURACY >= $TARGET_ACCURACY" | bc -l) )); then
                    echo "$(date): 达到目标准确率! $ACCURACY" >> $LOG_FILE
                    echo '{"status": "success", "accuracy": '$ACCURACY', "config": "'$SYMBOL'-'$TIMEFRAME'-'$LOOKBACK'"}' > $RESULTS_FILE
                    exit 0
                fi
            fi
        done
    done
done

echo "$(date): 本次训练完成，未达到目标准确率" >> $LOG_FILE

# 检查是否需要继续（最多运行100次）
RUN_COUNT=$(grep "开始训练任务" $LOG_FILE | wc -l)
if [ $RUN_COUNT -ge 100 ]; then
    echo "$(date): 达到最大运行次数，停止任务" >> $LOG_FILE
    # 删除cron任务
    crontab -l | grep -v "stock_training" | crontab -
    echo '{"status": "stopped", "reason": "max_runs_reached", "runs": '$RUN_COUNT'}' > $RESULTS_FILE
    exit 1
fi

exit 0
