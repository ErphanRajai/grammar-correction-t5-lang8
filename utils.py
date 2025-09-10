"""
Contains utility functions for the training and evaluation process.
"""
import torch
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from config import CHECKPOINT

def get_device():
    """
    Checks for CUDA availability and returns the appropriate device.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_collator(device, tokenizer):
    """
    Loads the model and data collator.
    
    Args:
        device (str): The device to load the model on ("cuda" or "cpu").
        tokenizer (AutoTokenizer): The tokenizer to use with the data collator.
        
    Returns:
        tuple: A tuple containing the model and data collator.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT).to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return model, data_collator
