export RUN_NAME=dyrren_run_2e-5_16
export WANDB_DISABLED=false
export TOKENIZERS_PARALLELISM=false

export WANDB_PROJECT=DyRRen_project

CUDA_VISIBLE_DEVICES=1 python run_finqa.py \
  --do_train true \
  --do_eval \
  --do_predict \
  --seed 8 \
  --topn_from_retrieval_texts 3 \
  --dropout_rate 0.1 \
  --model_name_or_path /home/xli/PTLMs/bert-base-uncased \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 300 \
  --per_device_eval_batch_size 4 \
  --dataloader_num_workers 16 \
  --logging_steps 50 \
  --overwrite_output_dir \
  --eval_steps 300 \
  --evaluation_strategy steps \
  --task_name FinQA \
  --data_dir FinQADataset \
  --output_dir outputs/$RUN_NAME \
  --fp16 true \
  --save_total_limit 3 \
  --save_strategy steps \
  --save_steps 300 \
  --load_best_model_at_end true \
  --metric_for_best_model eval_dev_exe_acc \
  --warmup_steps 0 \
  --max_question_length 0 \
  --max_seq_length 256
