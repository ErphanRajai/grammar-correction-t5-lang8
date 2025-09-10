Grammar Correction with T5-smallThis repository contains the code for fine-tuning a T5-small model for grammatical error correction (GEC) on the Lang-8 dataset. The project is designed to be a clean, modular, and reproducible training pipeline.Project StructureThe code is organized into several distinct files to improve readability and maintainability:train.py: The main script that orchestrates the entire training process. It loads the data, initializes the model and trainer, and starts training. This is the file you run to begin fine-tuning.data.py: Handles all data-related tasks, including loading the MohamedAshraf701/lang-8 dataset, cleaning it by removing null or empty values, and tokenizing the data for the T5 model.utils.py: Contains helper functions for common tasks, such as detecting the available device (cuda or cpu) and loading the model and data collator.config.py: Centralizes all the hyperparameters and configuration settings for the training, making it easy to adjust parameters like batch size, learning rate, and epochs in a single place.metrics.py: Defines the compute_metrics function, which is crucial for evaluating the model during training. It calculates important metrics like SacreBLEU and ROUGE to measure the model's performance.Achieved ResultsThe fine-tuning process yielded the following results, which were recorded at the end of each training epoch:EpochTraining LossValidation LossSacrebleuRouge1Rouge2Rougel10.7358000.69626967.0067180.8549750.7104850.84357220.7441000.68099467.2300550.8559860.7126850.84462230.7191000.67662067.3702840.8565890.7138770.84520940.7033000.67553267.4235330.8570270.7147680.845684Analysis of Results:Your model's performance is quite strong and shows significant learning throughout the training process. The key metrics to focus on are SacreBLEU and ROUGE.SacreBLEU: This metric is widely used in machine translation and is a good indicator of how closely the corrected text matches the reference (human-corrected) text. Your final SacreBLEU score of ~67.42 is excellent for this task. SacreBLEU scores of 60 or higher on similar tasks are often considered very competitive. It indicates that your model is generating high-quality corrections that are not only grammatically correct but also stylistically similar to the original corrected sentences in the dataset.ROUGE: This family of metrics (ROUGE-1, ROUGE-2, ROUGE-L) focuses on recall and measures the overlap of n-grams (sequences of words) between the generated and reference texts. Your scores are all very high, with ROUGE-1, ROUGE-2, and ROUGE-L hovering around 0.85, 0.71, and 0.84, respectively. These scores confirm that the model is effectively capturing the key content and structure from the original sentences.Loss: The validation loss consistently decreases, indicating that the model is generalizing well to unseen data and not overfitting.Overall, these metrics suggest that your fine-tuned T5-small model performs very well at grammar correction.How to Run the CodeTo run this project, follow these simple steps:Clone the Repository:git clone [https://huggingface.co/spaces/itserphan/grammar-correction-t5-lang8](https://huggingface.co/spaces/itserphan/grammar-correction-t5-lang8)
cd grammar-correction-t5-lang8
Install Dependencies:Make sure you have Python installed. You will need to install the required libraries, which are typically listed in a requirements.txt file (you can create one if it doesn't exist by listing the packages you used, e.g., torch, transformers, datasets, sacrebleu, evaluate, etc.).pip install -r requirements.txt
Run the Training Script:This command will start the data loading, preprocessing, training, and evaluation process. It will also handle pushing the model to the Hugging Face Hub.python train.py
How to Use the Model for InferenceYou can easily use your fine-tuned model for grammar correction directly from the Hugging Face Hub.First, install the transformers library:pip install transformers
Then, you can load and use the model with a simple Python script:from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model from your Hugging Face Hub repository
model_id = "itserphan/grammar-correction-t5-lang8"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

def correct_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Example usage
incorrect_sentence = "He is go to school."
corrected_sentence = correct_text(incorrect_sentence)
print(f"Original: {incorrect_sentence}")
print(f"Corrected: {corrected_sentence}")
Try the Live DemoYou can interact with the model directly on its Hugging Face Space:itserphan/grammar-correction-t5-lang8
