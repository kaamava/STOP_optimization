#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Using local machine for training (with relation JS loss)"

set -euo pipefail

# experiment tag
group=group2-2

# dataset configuration
dataset=hmdb
fps=3

DATA_PATH=/home/shenzhennan/vla/CVPR2025-STOP/dataset/hmdb
train_csv=${DATA_PATH}/hmdb_train.csv
val_csv=${DATA_PATH}/hmdb_val.csv
features_path=/home/shenzhennan/vla/hmdb51_export/videos/all_videos
pretrained_dir=/home/shenzhennan/vla/
data_path=${DATA_PATH}/hmdb_data.json

# train/eval switches
do_train=1
do_eval=0

# optimization setup
pretrained_clip_name=ViT-B/32
lr=1e-3
coef_lr=5e-4
wd=0.2
epochs=5
optim=AdamW
max_words=32
max_frames=12
temperature_new=1.0
resume=
load_from_pretrained=0
batch_size=32
batch_size_val=16
num_workers=8
n_display=50
precision=amp

freeze_clip=1
time_embedding=0
shared_latent_space=transformer

# relation loss knobs
use_relation_loss=0
relation_loss_weight=0.1
relation_loss_tau=0.1

# distributed training init
init_method='tcp://127.0.0.1:6010'

weight_tag=${relation_loss_weight//./p}
tau_tag=${relation_loss_tau//./p}
current_datetime=$(TZ="Asia/Shanghai" date +"%Y-%m-%d-%H:%M:%S")
model_dir=logs/${current_datetime}_${dataset}_STOP_jsloss_w${weight_tag}_tau${tau_tag}

mkdir -p "${model_dir}" "logs"

echo "The model dir is ${model_dir}"

python  main.py \
        --do_train ${do_train} \
        --do_eval ${do_eval} \
        --num_thread_reader ${num_workers} \
        --epochs ${epochs} \
        --batch_size ${batch_size} \
        --n_display ${n_display} \
        --train_csv ${train_csv} \
        --val_csv ${val_csv} \
        --data_path ${data_path} \
        --features_path ${features_path} \
        --output_dir ${model_dir} \
        --optim ${optim} \
        --lr ${lr} \
        --coef_lr ${coef_lr} \
        --wd ${wd} \
        --max_words ${max_words} \
        --max_frames ${max_frames} \
        --batch_size_val ${batch_size_val} \
        --datatype ${dataset} \
        --expand_msrvtt_sentences \
        --feature_framerate ${fps} \
        --freeze_layer_num 12 \
        --slice_framepos 2 \
        --loose_type \
        --linear_patch 2d \
        --sim_header meanP \
        --pretrained_clip_name ${pretrained_clip_name} \
        --precision ${precision} \
        --init_method ${init_method} \
        --pretrained_dir ${pretrained_dir} \
        --freeze_clip ${freeze_clip} \
        --time_embedding ${time_embedding} \
        --load_from_pretrained ${load_from_pretrained} \
        --shared_latent_space ${shared_latent_space} \
        --temporal_prompt ${group} \
        --use_relation_loss ${use_relation_loss} \
        --relation_loss_weight ${relation_loss_weight} \
        --relation_loss_tau ${relation_loss_tau} \
        --use_attn_for_discriminative 1 \
        --attn_alpha 0.5

echo "Training Finished!!!"
