import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Step 1: Load Preprocessed Data
train_data = pd.read_csv("datasets/preprocessed_train.csv")
test_data = pd.read_csv("datasets/preprocessed_test.csv")
valid_data = pd.read_csv("datasets/preprocessed_valid.csv")

# Step 2: Define Dataset Class
class FakeNewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        statement = self.data.iloc[index]["statement"]
        label = self.data.iloc[index]["label"]

        encoding = self.tokenizer.encode_plus(
            statement,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Step 3: Initialize Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

# Step 4: Create Datasets
train_dataset = FakeNewsDataset(train_data[:1000], tokenizer, max_len=128)
valid_dataset = FakeNewsDataset(valid_data[:500], tokenizer, max_len=128)

# Step 5: Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch"
)

# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Step 7: Train the Model
trainer.train()

# Step 8: Save the Fine-Tuned Model
model.save_pretrained("models/fake-news-detector")
tokenizer.save_pretrained("models/fake-news-detector")

print("Model training complete. Model saved in 'models/fake-news-detector'.")
