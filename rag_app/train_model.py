from datasets import load_dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import torch
import os
import sys
import unicodedata

# Set UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

MODEL_NAME = "google/flan-t5-base"
SAVE_PATH = "models/flan_qa_model"

def normalize_text(text):
    """Normalize Unicode text to handle special characters"""
    return unicodedata.normalize('NFKC', text)

def load_and_prepare_data():
    try:
        print("📥 Loading SQuAD dataset...")
        # Load specific splits to avoid glob pattern issues
        train_dataset = load_dataset("squad", split="train[:1000]")  # Using first 1000 examples for faster training
        test_dataset = load_dataset("squad", split="validation[:100]")  # Using first 100 examples for validation
        
        def preprocess(example):
            context = normalize_text(example['context'])
            question = normalize_text(example['question'])
            answer = normalize_text(example['answers']['text'][0]) if example['answers']['text'] else ""
            return {
                "input_text": f"Context: {context} Question: {question}",
                "target_text": answer
            }

        print("📝 Preprocessing dataset...")
        train_dataset = train_dataset.map(preprocess)
        test_dataset = test_dataset.map(preprocess)
        
        # Combine into a single dataset dictionary
        dataset = {
            "train": train_dataset,
            "test": test_dataset
        }
        return dataset
    except Exception as e:
        print(f"❌ Error in data preparation: {str(e)}")
        raise

def tokenize_data(dataset, tokenizer):
    try:
        def tokenize(example):
            model_input = tokenizer(
                example["input_text"], max_length=512, truncation=True, padding="max_length"
            )
            label = tokenizer(
                example["target_text"], max_length=64, truncation=True, padding="max_length"
            )
            model_input["labels"] = label["input_ids"]
            return model_input

        print("🔄 Tokenizing dataset...")
        return dataset.map(tokenize, batched=True)
    except Exception as e:
        print(f"❌ Error in tokenization: {str(e)}")
        raise

def train():
    try:
        print("📦 Loading and preprocessing SQuAD dataset...")
        dataset = load_and_prepare_data()
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        tokenized_dataset = tokenize_data(dataset, tokenizer)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

        print("🧠 Starting training with multiple epochs...")

        training_args = Seq2SeqTrainingArguments(
            output_dir=SAVE_PATH,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            weight_decay=0.01,
            save_total_limit=2,
            num_train_epochs=3,
            predict_with_generate=True,
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=torch.cuda.is_available()
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
        )

        print("🚀 Starting training...")
        trainer.train()

        print("\n✅ Training complete. Saving fine-tuned model...")
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print(f"📁 Model saved to: {SAVE_PATH}")

        input("\nPress Enter to close...")  # Keeps terminal open

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train()
