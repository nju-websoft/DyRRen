export RUN_NAME=row_as_sentence
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 python run_finqa_retriever.py \
  --eval_only true \
  --eval_checkpoints outputs/$RUN_NAME \
  --pred_mode train \
  --do_train false \
  --do_eval false \
  --do_predict \
  --use_2_encoder false \
  --model_name_or_path ../bert-base-uncased \
  --learning_rate 2e-5 \
  --max_texts_training_retrieval 25 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 8 \
  --per_device_eval_batch_size 1 \
  --dataloader_num_workers 16 \
  --logging_steps 25 \
  --overwrite_output_dir \
  --eval_steps 100 \
  --evaluation_strategy steps \
  --task_name FinQA \
  --data_dir FinQADataset \
  --output_dir outputs/$RUN_NAME \
  --fp16 true \
  --save_total_limit 3 \
  --save_strategy steps \
  --save_steps 100 \
  --load_best_model_at_end true \
  --metric_for_best_model eval_recall \
  --warmup_ratio 0.1 \
