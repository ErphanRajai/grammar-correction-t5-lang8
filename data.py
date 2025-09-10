"""
Handles data loading and preprocessing for the training pipeline.
"""
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from config import CHECKPOINT

def load_and_preprocess_data():
    """
    Loads the dataset, cleans it, and tokenizes it for training.

    Returns:
        tuple: A tuple containing the tokenized train and test datasets.
    """

    dataset = load_dataset(DATASET_NAME, split="train")

    dataset = dataset.rename_column("processed_input", "input")
    dataset = dataset.rename_column("processed_output", "output")

    clean_dataset = dataset.filter(
        lambda x: x["input"] is not None and x["output"] is not None
    )
    clean_dataset = clean_dataset.filter(
        lambda x: x["input"].strip() != "" and x["output"].strip() != ""
    )
    
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    
    def preprocess_function(examples):
        """
        Tokenizes the input and output texts.
        """
        model_inputs = tokenizer(
            examples["input"],
            truncation=True
        )
        labels = tokenizer(
            text_target=examples["output"],
            truncation=True
        )

        model_inputs["labels"] = [list(map(int, l)) for l in labels["input_ids"]]
        return model_inputs

    tokenized_dataset = clean_dataset.map(preprocess_function, batched=True)
    
    tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])

    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    
    return tokenized_dataset["train"], tokenized_dataset["test"], tokenizer

if __name__ == "__main__":
    from config import DATASET_NAME
    train_dataset, test_dataset, tokenizer = load_and_preprocess_data()
    print("Training dataset:", train_dataset)
    print("Test dataset:", test_dataset)
    print("Tokenizer:", tokenizer)
