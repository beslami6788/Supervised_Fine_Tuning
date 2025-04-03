# Supervised Fine Tuning
CUDA_VISIBLE_DEVICES=0 python qlora_training.py -i training_dataset.jsonl -d validation_dataset.jsonl -m /path/to/llama3.1-8B -f output_dir_for_finetuned_model -p none
# SFT runs for the default of 2 epochs
CUDA_VISIBLE_DEVICES=0 python hf_lora_inference.py -i test_dataset.jsonl -d output_dir_for_finetuned_model -m /path/to/llama3.1-8B --outputFile SFT_predictions.csv
