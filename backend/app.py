from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertForSequenceClassification, BertTokenizer
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import torch
from datetime import datetime

app = Flask(__name__)
CORS(app)
# Replace with your database credentials
db_user = "postgres"
db_password = "1234"
db_host = "localhost"
db_port = "5432"
db_name = "fakenews"

# Create the SQLAlchemy engine
engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

Base = declarative_base()

# Define the database table
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    statement = Column(String, nullable=False)
    label = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

    # 0: "True",
    # 1: "Mostly True",
    # 2: "Half-True",
    # 3: "Barely True",
    # 4: "False",
    # 5: "Pants on Fire"

Session = sessionmaker(bind=engine)
session = Session()

MODEL_PATH = "./models/fake-news-detector"
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

@app.route('/')
def home():
    return "Welcome to the Fake News Detector API. Use the `/predict` endpoint for predictions."

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "Use POST with JSON data to get predictions.", 405

    data = request.json
    statement = data.get('statement', '')

    if not statement:
        return jsonify({'error': 'No statement provided'}), 400

    # Tokenize and predict
    inputs = tokenizer(statement, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(probs).item()
        confidence = probs[0][predicted_label].item()

    # Save prediction to the database
    new_prediction = Prediction(
        statement=statement,
        label=predicted_label,
        confidence=confidence
    )
    session.add(new_prediction)
    session.commit()

    response = {
        'label': predicted_label,
        'confidence': confidence
    }
    print("Response being sent:", response)  # Debugging log
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
