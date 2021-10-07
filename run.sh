#!/bin/bash

ep=750
model="TPA"
dataset="solar"
output_dir="outputs"

python3 main.py \
  --output_dir $output_dir \
  --n_epochs $ep \
  --bad_limit 25 \
  --lr 3e-3 \
  --rho_lr 1e-2 \
  --one_rho --inp_adj --out_adj \
  --batch_size 64 \
  --series_len 60 \
  --norm_type standard \
  forecasting \
  --model_type $model \
  --dataset $dataset
