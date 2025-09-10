# Fine-Tuning a T5-small Model for Grammar Correction

This project provides a clean and modular training pipeline for fine-tuning a **T5-small** model on the **Lang-8 dataset** for high-quality grammatical error correction.

---

## 📂 Project Structure

- **`train.py`** – Orchestrates the entire training process (data loading, model setup, training).  
- **`data.py`** – Handles dataset loading (`MohamedAshraf701/lang-8`), cleaning null/empty values, and tokenization.  
- **`utils.py`** – Helper functions (e.g., device detection, model/data collator setup).  
- **`config.py`** – Centralizes hyperparameters (batch size, learning rate, epochs, etc.).  
- **`metrics.py`** – Defines `compute_metrics`, calculating **SacreBLEU** and **ROUGE** scores.
- **`app.py`** – Gradio demo / inference interface to test the model interactively.  
- **`requirements.txt`** – Python dependencies for training, evaluation, and inference.

---

## 📊 Achieved Results

| Epoch | Training Loss | Validation Loss | SacreBLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------------|-----------------|-----------|---------|---------|---------|
| 1     | 0.7358        | 0.6963          | 67.0067   | 0.8550  | 0.7105  | 0.8436  |
| 2     | 0.7441        | 0.6810          | 67.2301   | 0.8560  | 0.7127  | 0.8446  |
| 3     | 0.7191        | 0.6766          | 67.3703   | 0.8566  | 0.7139  | 0.8452  |
| 4     | 0.7033        | 0.6755          | 67.4235   | 0.8570  | 0.7148  | 0.8457  |

### ✅ Performance Summary
- **SacreBLEU ~67.4** – excellent, very competitive score for grammar correction tasks.  
- **ROUGE-1: 0.857, ROUGE-2: 0.715, ROUGE-L: 0.846** – high overlap with human references.  
- **Validation loss decreasing** – model generalizes well without overfitting.  

---

## 🚀 How to Run

1. **Clone the Repository**  
```bash
git clone https://huggingface.co/spaces/itserphan/grammar-correction-t5-lang8
cd grammar-correction-t5-lang8
```

2. **Install Dependencies**
```
pip install -r requirements.txt
```

3. **Run Training**
```
python train.py
```
This handles dataset preparation, fine-tuning, evaluation, and pushing to the Hugging Face Hub.

## 🤖 Inference (Use the Model)
**Install transformers first:**
```
pip install transformers
```

**Then run inference:**
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load from Hugging Face Hub
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
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
incorrect = "He is go to school."
corrected = correct_text(incorrect)
print("Original:", incorrect)
print("Corrected:", corrected)
```

## 🌐 Live Demo
👉 Try the model directly on [Hugging Face Spaces](https://huggingface.co/spaces/itserphan/grammar-correction-t5-lang8)
