import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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


demo = gr.Interface(
    fn=correct_text,
    inputs=gr.Textbox(lines=5, placeholder="Enter your sentence with grammar mistakes..."),
    outputs=gr.Textbox(label="Corrected Sentence"),
    title="Grammar Correction with T5-small",
    description="This model automatically corrects grammar mistakes in English sentences. Trained on the Lang-8 dataset.",
    examples= [
    ["i went to the park yesterday it was fun"],
    ["She go to school every day by bus."],
    ["The dogs runs in the yard."],
    ["He bought apple from the market."],
    ["Beautiful very the garden is."],
    ["I don’t know where is she."]]
)

if __name__ == "__main__":
    demo.launch()
