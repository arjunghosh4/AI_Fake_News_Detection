import pandas as pd
import re

# Step 1: Define column names based on dataset description
column_names = [
    "id", "label", "statement", "subject", "speaker", "job_title", 
    "state_info", "party_affiliation", "barely_true_counts", "false_counts", 
    "half_true_counts", "mostly_true_counts", "pants_onfire_counts", "context"
]

# Step 2: Load the datasets
train_data = pd.read_csv("datasets/train.tsv", sep="\t", names=column_names)
test_data = pd.read_csv("datasets/test.tsv", sep="\t", names=column_names)
valid_data = pd.read_csv("datasets/valid.tsv", sep="\t", names=column_names)

# Step 3: Extract relevant columns
train_data = train_data[["statement", "label"]]
test_data = test_data[["statement", "label"]]
valid_data = valid_data[["statement", "label"]]

# Step 4: Map labels to numeric values
label_mapping = {
    "true": 0,
    "mostly-true": 1,
    "half-true": 2,
    "barely-true": 3,
    "false": 4,
    "pants-fire": 5
}

train_data["label"] = train_data["label"].map(label_mapping)
test_data["label"] = test_data["label"].map(label_mapping)
valid_data["label"] = valid_data["label"].map(label_mapping)

# Step 5: Clean the text in the statement column
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)
    return text

train_data["statement"] = train_data["statement"].apply(clean_text)
test_data["statement"] = test_data["statement"].apply(clean_text)
valid_data["statement"] = valid_data["statement"].apply(clean_text)

# Step 6: Save the preprocessed datasets
train_data.to_csv("datasets/preprocessed_train.csv", index=False)
test_data.to_csv("datasets/preprocessed_test.csv", index=False)
valid_data.to_csv("datasets/preprocessed_valid.csv", index=False)

print("Preprocessing complete. Preprocessed datasets saved.")
