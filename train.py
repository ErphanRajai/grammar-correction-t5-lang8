"""
The main script to run the training and push the model to Hugging Face Hub.
"""
import torch
from transformers import Seq2SeqTrainer, AutoTokenizer
from data import load_and_preprocess_data
from utils import get_device, load_model_and_collator
from metrics import compute_metrics
from config import TRAINING_ARGS, CHECKPOINT
from huggingface_hub import login

def main():
    """
    Main function to run the training process.
    """
    print("Step 1: Loading and preprocessing data...")
    train_dataset, test_dataset, tokenizer = load_and_preprocess_data()
    
    print("\nStep 2: Setting up device, model, and data collator...")
    device = get_device()
    print(f"Using device: {device}")
    model, data_collator = load_model_and_collator(device, tokenizer)
    
    print("\nStep 3: Initializing the trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer)
    )

    print("\nStep 4: Starting training...")
    trainer.train()

    print("\nStep 5: Pushing model and tokenizer to Hugging Face Hub...")
    try:
        # You may need to run `huggingface-cli login` in your terminal
        # or `login()` and follow the prompts in a notebook environment
        login()
        trainer.push_to_hub("itserphan/grammar-correction-t5-lang8")
        print("Model pushed to the Hub successfully!")
    except Exception as e:
        print(f"Failed to push to Hub: {e}")

if __name__ == "__main__":
    main()
