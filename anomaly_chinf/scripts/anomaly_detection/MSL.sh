export CUDA_VISIBLE_DEVICES=0

nohup python -u run_anomaly.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/processed/MSL \
  --model_id MSL \
  --model iTransformer \
  --data MSL \
  --features M \
  --seq_len 10 \
  --pred_len 0 \
  --d_model 256 \
  --d_ff 64 \
  --e_layers 1 \
  --enc_in 55 \
  --c_out 55 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --cp 10 \
  --tracin_layers 1 \
  --train_epochs 20 > output.log 2>&1 & 
