#!/bin/bash
# runtime_listener.sh

# 定义最大运行时间为 4 小时（14400 秒）
MAX_RUNTIME=14400

# 记录任务启动时间的文件
RUNTIME_FILE="/tmp/my_program_runtime"

while read line; do
  # 检查任务是否启动
  if echo "$line" | grep -q "PROCESS_STATE_RUNNING"; then
    # 记录启动时间
    date +%s > "$RUNTIME_FILE"
  fi
  
  # 检查任务是否退出
  if echo "$line" | grep -q "PROCESS_STATE_EXITED"; then
    # 计算运行时间
    start_time=$(cat "$RUNTIME_FILE")
    current_time=$(date +%s)
    runtime=$((current_time - start_time))
    
    # 检查是否超过最大运行时间
    if [ "$runtime" -ge "$MAX_RUNTIME" ]; then
      # 超过最大运行时间，停止重启
      supervisorctl stop my_program
      echo "Program ran for $runtime seconds, stopping further restarts." >> /tmp/runtime_listener.log
      exit 0  # 停止监听器
    fi
  fi
done
