
python outline_whole_generation.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_predict \
    --train_file= ./data/WPOG/train.csv \
    --validation_file=.data/WPOG/validation.csv \
    --test_file=.data/WPOG/test.csv \
    --num_train_epochs=3 \
    --output_dir ./result/ \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --save_steps=30000 \
    --predict_with_generate \
    --overwrite_output_dir \

