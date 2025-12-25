#!/bin/bash
# Usage: bash scripts/msrvtt_eval.sh <checkpoint_dir> <do_jsloss_tag>
# <checkpoint_dir> should contain ckpt.best.pth.tar (or adjust --resume below).

set -euo pipefail

MODEL_DIR=${1:-logs/latest_msrvtt_model}
TAG=${2:-eval}

export CUDA_VISIBLE_DEVICES=0
echo "Evaluating MSRVTT model from ${MODEL_DIR} (tag=${TAG})"

# paths for evaluation split with captions
DATA_PATH=/home/shenzhennan/vla/CVPR2025-STOP/dataset/msrvtt
val_csv=${DATA_PATH}/MSRVTT_JSFUSION_test.csv
features_path=/home/shenzhennan/vla/MSRVTT/videos/all
pretrained_dir=/home/shenzhennan/vla/
data_path=/home/shenzhennan/vla/msrvtt_data/MSRVTT_data.json

pretrained_clip_name=ViT-B/32
batch_size_val=16
num_workers=8
fps=3

resume=${MODEL_DIR}/ckpt.best.pth.tar
output_dir=${MODEL_DIR}/eval_${TAG}
mkdir -p "${output_dir}"

echo "Writing eval logs to ${output_dir}"

python main.py \
    --do_train 0 \
    --do_eval 1 \
    --num_thread_reader ${num_workers} \
    --epochs 1 \
    --batch_size 32 \
    --n_display 50 \
    --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
    --val_csv ${val_csv} \
    --data_path ${data_path} \
    --features_path ${features_path} \
    --output_dir ${output_dir} \
    --optim AdamW \
    --lr 1e-4 \
    --coef_lr 5e-4 \
    --wd 0.2 \
    --max_words 32 \
    --max_frames 12 \
    --batch_size_val ${batch_size_val} \
    --datatype msrvtt \
    --feature_framerate ${fps} \
    --freeze_layer_num 12 \
    --slice_framepos 2 \
    --loose_type \
    --linear_patch 2d \
    --sim_header meanP \
    --pretrained_clip_name ${pretrained_clip_name} \
    --precision amp \
    --init_method tcp://127.0.0.1:6010 \
    --pretrained_dir ${pretrained_dir} \
    --freeze_clip 1 \
    --time_embedding 0 \
    --load_from_pretrained 0 \
    --shared_latent_space transformer \
    --temporal_prompt group2-2 \
    --resume ${resume} \
    "$@"

echo "Evaluation Finished"
