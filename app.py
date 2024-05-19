from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

app = Flask(__name__)

# Path to the directory containing tokenizer files (assuming 'vocab.txt' exists)
tokenizer_dir = './models/bert_tokenizer'

# Create the tokenizer object
tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)

# Path to the directory containing the model files (assuming 'pytorch_model.bin' exists)
model_dir = './models/bert_model'

# Load the pre-trained sentiment analysis model
model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=3)
model.eval()

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    if predicted_class == 0:
        sentiment = "Negative"
    elif predicted_class == 1:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"
    return sentiment

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form['sentence']
        sentiment = predict_sentiment(text)
        return render_template('predict.html', text=text, sentiment=sentiment)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2825, debug=True)
