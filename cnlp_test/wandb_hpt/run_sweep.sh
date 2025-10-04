#!/bin/bash

# 检查输入参数
if [ $# -ne 2 ]; then
    echo "Usage: $0 <sweep_id> <model_type>"
    echo "Example: $0 nkw3mr-university-of-virginia/VIB-Intepretable-Text-Classification/k8d29whu bert"
    echo "Supported model types: bert, cnn, lstm, hier"
    exit 1
fi

SWEEP_ID="$1"
MODEL_TYPE="$2"

# 如果SWEEP_ID不包含完整路径，自动添加项目前缀
if [[ "$SWEEP_ID" != *"/"* ]]; then
    SWEEP_ID="nkw3mr-university-of-virginia/Concept-Text-Classification/$SWEEP_ID"
    echo "Auto-completing sweep ID to: $SWEEP_ID"
fi
GPU_LIST=(1 2 6 7)  # 定义GPU列表
LOG_DIR="sweep_${MODEL_TYPE}_logs"

# 验证model_type
case "$MODEL_TYPE" in
    bert|cnn|lstm|hier)
        echo "Using model type: $MODEL_TYPE"
        ;;
    *)
        echo "Error: Unsupported model type '$MODEL_TYPE'"
        echo "Supported model types: bert, cnn, lstm, hier"
        exit 1
        ;;
esac

export HF_HOME=/bigtemp/nkw3mr/huggingface_home
export HF_DATASETS_CACHE=/bigtemp/nkw3mr/huggingface_cache/datasets
export TRANSFORMERS_CACHE=/bigtemp/nkw3mr/huggingface_cache/transformers

mkdir -p $LOG_DIR

echo "Starting ${#GPU_LIST[@]} agents for sweep: $SWEEP_ID"
echo "Model type: $MODEL_TYPE"
echo "Log directory: $LOG_DIR"
echo "Using GPUs: ${GPU_LIST[*]}"

# 遍历GPU列表
for gpu in "${GPU_LIST[@]}"; do
    echo "Starting agent on GPU $gpu..."
    nohup bash -c "CUDA_VISIBLE_DEVICES=$gpu wandb agent $SWEEP_ID" > "$LOG_DIR/sweep_gpu_$gpu.log" 2>&1 &
    sleep 15
done


echo "All agents started. Check logs: $LOG_DIR/sweep_gpu_*.log"
echo "Process IDs:"
ps aux | grep "wandb agent" | grep -v grep

echo ""
echo "To stop all agents: pkill -f 'wandb agent'"
echo "To monitor logs: tail -f $LOG_DIR/sweep_gpu_0.log"