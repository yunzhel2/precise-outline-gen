# Precise outline-conditioned text generation

---

Implementation for our work: Advancing Precise Outline-Conditioned Text Generation with Task Duality and Explicit Outline Control.  (EACL24)

## Environment

- Python 3.9
- datasets  1.8.0
- transformers 4.33.2
- torch 2.0.0
- tokenizers 0.13.3
- numpy 2.4.0

## Usage

### Data
Datasets available in [Google Drive](https://drive.google.com/drive/folders/19WbUelezNzaYGqDf82LrEqZoAuwUdjo1?usp=sharing).

### Code
Here we provide code for two scenarios. Fine-tuned methods are in the folder`ft_lm_conditional_gen`. Please change the parameters(e.g., path) in `outline_based_generation.sh`. Run it directly by following script: 

```angular2html
python outline_whole_generation.py 
    --model_name_or_path facebook/bart-base 
    --do_train 
    --do_predict 
    --train_file= ./data/WPOG/train.csv 
    --validation_file= .data/WPOG/validation.csv 
    --test_file= .data/WPOG/test.csv 
    --num_train_epochs=10 
    --max_source_length=512
    --max_target_length=1024
    --output_dir ./result/ 
    --per_device_train_batch_size=8 
    --per_device_eval_batch_size=8 
    --save_steps=30000 
    --predict_with_generate 
    --overwrite_output_dir
```

The zero-shot inference(LLMs) could be found at `llm_inference.py`. You could directly run by 

``
python llm_inference.py
``

