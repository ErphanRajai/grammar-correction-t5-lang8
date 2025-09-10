"""
Defines the `compute_metrics` function for evaluation during training.
"""
import numpy as np
from evaluate import load
from transformers import AutoTokenizer
from config import CHECKPOINT

sacrebleu_metric = load("sacrebleu")
rouge_metric = load("rouge")

def compute_metrics(eval_pred, tokenizer=None):
    """
    Computes SacreBLEU and ROUGE scores.
    
    Args:
        eval_pred (tuple): A tuple containing predictions and labels.
        tokenizer (AutoTokenizer): The tokenizer used for decoding.
        
    Returns:
        dict: A dictionary of computed metrics.
    """
    if tokenizer is None:

        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    predictions, labels = eval_pred


    vocab_size = tokenizer.vocab_size
    safe_predictions = predictions.copy()
    safe_predictions[safe_predictions >= vocab_size] = tokenizer.pad_token_id
    safe_predictions[safe_predictions < 0] = tokenizer.pad_token_id
    

    decoded_preds = tokenizer.batch_decode(safe_predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    references = [[l] for l in decoded_labels]
    
    bleu_result = sacrebleu_metric.compute(predictions=decoded_preds, references=references)
    
    rouge_result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    result = {
        "sacrebleu": bleu_result["score"],
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
    }
    
    return result
