# AI Fake News Detector

## Overview
The **AI Fake News Detector** is a full-stack web application designed to classify the truthfulness of a given statement. The system is built using **React.js** for the frontend and **Flask** with **BERT (Bidirectional Encoder Representations from Transformers)** for the backend, ensuring accurate text classification.

## Features
- Accepts a user input statement for analysis.
- Uses a **fine-tuned BERT model** for prediction.
- Returns a label and confidence score indicating the likelihood of the statement's truthfulness.
- Stores predictions in a **PostgreSQL** database.
- Provides a simple, intuitive user interface built with **React.js**.

## Technologies Used
### **Frontend** (React.js)
- React.js
- Axios (for API requests)
- CSS for styling

### **Backend** (Flask & BERT)
- Flask (Python)
- PyTorch (for BERT model inference)
- Transformers (Hugging Face Library)
- SQLAlchemy (for database interactions)
- PostgreSQL (Database)

## Installation & Setup
### **1. Clone the repository**
```sh
 git clone https://github.com/your-repo/ai-fake-news-detector.git
 cd ai-fake-news-detector
```

### **2. Setup Backend**
```sh
 cd backend
 python -m venv venv
 source venv/bin/activate   # On Windows: venv\Scripts\activate
 pip install -r requirements.txt
```

#### **Run the backend server**
```sh
 python app.py
```
- This will start the Flask API on **http://127.0.0.1:5000/**.

### **3. Setup Frontend**
```sh
 cd ../frontend
 npm install
 npm start
```
- The React app will start on **http://localhost:3000/**.

## Project Structure
```
ai-fake-news-detector/
│-- backend/
│   ├── app.py                 # Flask API for prediction
│   ├── models/
│   │   ├── fake-news-detector # BERT Model for classification
│   ├── preprocess.py          # Data preprocessing script
│   ├── train_model.py         # Model training script
│   ├── database_setup.py      # PostgreSQL setup
│   ├── requirements.txt       # Python dependencies
│
│-- frontend/
│   ├── src/
│   │   ├── App.js             # React app logic
│   │   ├── style.css          # Styling
│   ├── package.json           # Frontend dependencies
│
│-- datasets/
│   ├── train.tsv              # Raw training dataset
│   ├── test.tsv               # Raw testing dataset
│   ├── valid.tsv              # Validation dataset
```

## API Endpoints
### **1. Home**
- **`GET /`**
- Returns: `"Welcome to the Fake News Detector API."`

### **2. Predict Statement**
- **`POST /predict`**
- Input JSON:
  ```json
  {"statement": "The earth is flat"}
  ```
- Response JSON:
  ```json
  {
      "label": 4,
      "confidence": 0.85
  }
  ```
  - **Label Mapping:**
    - `0`: "True"
    - `1`: "Mostly True"
    - `2`: "Half-True"
    - `3`: "Barely True"
    - `4`: "False"
    - `5`: "Pants on Fire"

## Model Training (Preprocessing & Fine-Tuning BERT)
The **preprocess.py** script handles data preprocessing:
- Loads the dataset (`train.tsv`, `test.tsv`, `valid.tsv`).
- Cleans the text by removing URLs and special characters.
- Maps labels to numerical values.
- Saves the preprocessed data.

### **Run Preprocessing**
```sh
python preprocess.py
```

### **Training the Model**
The **train_model.py** script fine-tunes BERT on the fake news dataset:
- Loads the preprocessed dataset.
- Uses the `FakeNewsDataset` class to tokenize and prepare input data.
- Initializes a BERT model (`bert-base-uncased`) with 6 classification labels.
- Trains the model using **Hugging Face's Trainer API**.
- Saves the fine-tuned model to `models/fake-news-detector`.

### **Run Model Training**
```sh
python train_model.py
```
- After training, the fine-tuned model and tokenizer are saved in `models/fake-news-detector`.

## Database Integration
The Flask backend uses PostgreSQL to store predictions. To set up the database:
```sh
CREATE DATABASE fakenews;
```
Configure database credentials in `app.py`:
```python
db_user = "postgres"
db_password = "1234"
db_host = "localhost"
db_port = "5432"
db_name = "fakenews"
```

## Future Improvements
- Train BERT on a larger dataset for better accuracy.
- Deploy as a cloud-based web application.
- Enhance UI with additional features and analytics.

## License
This project is intended for educational and research purposes.

