export CUDA_VISIBLE_DEVICES=0

python -u run_anomaly.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path ./dataset/SWaT/GDN \
  --model_id SWaT \
  --model iTransformer \
  --data SWaT \
  --features M \
  --seq_len 10 \
  --pred_len 0 \
  --d_model 32 \
  --d_ff 32 \
  --e_layers 2 \
  --enc_in 51 \
  --c_out 51 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --cp 10 \
  --tracin_layers 2 \
  --train_epochs 20  

