export CUDA_VISIBLE_DEVICES=0

nohup python -u run_anomaly.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path ./dataset/WADI \
  --model_id WADI \
  --model iTransformer \
  --data WADI \
  --features M \
  --seq_len 10 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 3 \
  --enc_in 127 \
  --c_out 127 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --cp 10 \
  --tracin_layers 1 \
  --train_epochs 11  > output.log 2>&1 
 