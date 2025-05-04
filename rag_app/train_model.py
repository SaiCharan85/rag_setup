import os
import sys
import unicodedata
import json
from datasets import Dataset, DatasetDict
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import torch

# Set UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

MODEL_NAME = "google/flan-t5-base"  # Using base model for compatibility
SAVE_PATH = "models/flan_qa_model"

def normalize_text(text):
    """Normalize Unicode text to handle special characters"""
    return unicodedata.normalize('NFKC', text)

def load_and_prepare_data():
    try:
        print("📥 Loading SQuAD dataset...")
        import json
        import os
        
        # Download SQuAD dataset files if they don't exist
        squad_dir = "squad_data"
        os.makedirs(squad_dir, exist_ok=True)
        
        train_file = os.path.join(squad_dir, "train-v1.1.json")
        dev_file = os.path.join(squad_dir, "dev-v1.1.json")
        
        if not os.path.exists(train_file) or not os.path.exists(dev_file):
            print("Downloading SQuAD dataset files...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
                train_file
            )
            urllib.request.urlretrieve(
                "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
                dev_file
            )
        
        # Load the JSON files
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)['data']
        with open(dev_file, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)['data']
        
        # Convert to the format we need
        train_dataset = []
        for article in train_data[:20]:  # Limit to 20 articles for ~2k samples
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas'][:10]:  # Limit questions per paragraph
                    train_dataset.append({
                        'context': context,
                        'question': qa['question'],
                        'answers': {'text': [a['text'] for a in qa['answers']]}
                    })
        
        test_dataset = []
        for article in dev_data[:4]:  # Limit to 4 articles for ~400 samples
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas'][:10]:  # Limit questions per paragraph
                    test_dataset.append({
                        'context': context,
                        'question': qa['question'],
                        'answers': {'text': [a['text'] for a in qa['answers']]}
                    })

        # Convert lists to Dataset objects
        train_dataset = Dataset.from_list(train_dataset)
        test_dataset = Dataset.from_list(test_dataset)

        # Process the datasets
        def preprocess(example):
            context = normalize_text(example['context'])
            question = normalize_text(example['question'])
            answer = normalize_text(example['answers']['text'][0]) if example['answers']['text'] else ""
            
            input_text = f"""
            Context: {context}
            Question: {question}
            
            Instructions:
            1. Generate a natural, conversational response
            2. Use your own words instead of directly copying from context
            3. Make the response concise but complete
            4. Maintain proper grammar and punctuation
            5. Provide a clear and helpful answer
            """
            
            target_text = f"Here's what I found: {answer}. This means that..."
            
            return {
                "input_text": input_text,
                "target_text": target_text
            }

        print("📝 Preprocessing dataset...")
        train_dataset = train_dataset.map(preprocess)
        test_dataset = test_dataset.map(preprocess)

        # Combine into a DatasetDict
        dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        return dataset
    except Exception as e:
        print(f"❌ Error in data preparation: {str(e)}")
        raise

def tokenize_data(dataset, tokenizer):
    """Tokenize the dataset."""
    try:
        def tokenize(example):
            # Tokenize input text with padding and truncation
            model_input = tokenizer(
                example['input_text'],
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Tokenize target text with padding and truncation
            target = tokenizer(
                example['target_text'],
                padding='max_length',
                truncation=True,
                max_length=256,  # Increased from 128 to allow for longer answers
                return_tensors='pt'
            )
            
            # Add labels (target_ids) to the model input
            model_input['labels'] = target['input_ids']
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
        
        # Tokenize data
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        tokenized_dataset = tokenize_data(dataset, tokenizer)
        
        # Create model
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        
        print("🧠 Starting training with multiple epochs...")
        
        # Prepare training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=SAVE_PATH,
            evaluation_strategy="steps",
            logging_strategy="steps",
            per_device_train_batch_size=8,  # Reduced from 16 to avoid OOM
            per_device_eval_batch_size=8,   # Reduced from 16 to match train batch size
            num_train_epochs=3,
            learning_rate=5e-4,
            weight_decay=0.01,
            save_steps=500,
            eval_steps=500,
            logging_steps=50,
            warmup_steps=200,
            prediction_loss_only=True,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2,  # Increased from 1 to help with memory
            predict_with_generate=True,
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )
        
        # Create trainer
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
