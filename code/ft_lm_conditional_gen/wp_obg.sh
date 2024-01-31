HOME_PATH=./wp_dataset/

# Use the variable in your command
python outline_whole_generation.py \
    --model_name_or_path  \
    --do_predict \
    --train_file="${HOME_PATH}wp_train.csv" \
    --validation_file="${HOME_PATH}wp_valid.csv" \
    --test_file="${HOME_PATH}wp_test.csv" \
    --num_train_epochs=40 \
    --max_source_length=512 \
    --max_target_length=1024 \
    --max_predict_samples=200 \
    --output_dir obg_wp_generation_result \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --save_steps=8000 \
    --predict_with_generate \
    --overwrite_output_dir
