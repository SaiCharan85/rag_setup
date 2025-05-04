from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

MODEL_PATH = "google/flan-t5-base"

class FLANQAInferencer:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def answer(self, question, context):
        prompt = f"Context: {context} Question: {question}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            temperature=0.7,
            no_repeat_ngram_size=2
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
