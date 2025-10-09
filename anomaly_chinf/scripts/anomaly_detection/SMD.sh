export CUDA_VISIBLE_DEVICES=1

nohup python -u run_anomaly.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model iTransformer \
  --data SMD \
  --features M \
  --seq_len 10 \
  --pred_len 0 \
  --d_model 256 \
  --d_ff 128 \
  --e_layers 1 \
  --enc_in 38 \
  --c_out 38 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --cp 10 \
  --tracin_layers 2 \
  --train_epochs 11 > output.log 2>&1 & 

 