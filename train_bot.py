import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Load Dataset

import pandas as pd
import os

# Define the file path
file_path = r"C:\ai\agrismart\agri_chatbot_dataset.csv"

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Print column names
    print("CSV Columns:", df.columns.tolist())

    # Check if 'response' column exists
    if 'response' not in df.columns:
        print("Error: 'response' column not found!")
    else:
        print("'response' column found successfully!")


# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset Preparation
class ChatDataset(Dataset):
    def __init__(self, queries, responses):
        self.queries = queries
        self.responses = responses
        self.encodings = tokenizer(self.queries, truncation=True, padding=True, max_length=128)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.responses[idx])
        }

# Convert Responses to Numerical Labels
label_map = {response: i for i, response in enumerate(df["response"].unique())}
df["label"] = df["response"].map(label_map)

# Train-Test Split
train_texts, val_texts, train_labels, val_labels = train_test_split(df["query"], df["label"], test_size=0.2)

# Create Datasets
train_dataset = ChatDataset(train_texts.tolist(), train_labels.tolist())
val_dataset = ChatDataset(val_texts.tolist(), val_labels.tolist())

# Load Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_map))

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train Model
trainer.train()

# Save Model
model.save_pretrained("./trained_chatbot")
tokenizer.save_pretrained("./trained_chatbot")
