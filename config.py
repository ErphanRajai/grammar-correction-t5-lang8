"""
Configuration file for the project.
This file centralizes all hyperparameters, file paths, and other
configuration settings.
"""
from transformers import Seq2SeqTrainingArguments

CHECKPOINT = "google-t5/t5-small"

DATASET_NAME = "MohamedAshraf701/lang-8"

TRAINING_ARGS = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    learning_rate=5e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="sacrebleu",
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,
    fp16=True,
    report_to="none"
)
