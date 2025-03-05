from transformers import BertForSequenceClassification, BertTokenizer

MODEL_PATH = "./models/fake-news-detector"

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

print("Model and tokenizer loaded successfully!")
