export CUDA_VISIBLE_DEVICES=1

nohup python -u run_anomaly.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/processed/SMAP \
  --model_id SMAP \
  --model iTransformer \
  --data SMAP \
  --features M \
  --seq_len 10 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 1 \
  --enc_in 25 \
  --c_out 25 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --cp 10 \
  --tracin_layers 1 \
  --train_epochs 20 > output.log 2>&1 & 
