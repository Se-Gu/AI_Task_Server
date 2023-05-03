from flask import Flask, jsonify, request
from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app)
model_name = "opus-mt-tc-big-tr-en"
tokenizer = MarianTokenizer.from_pretrained(model_name, use_fast=False)
model = MarianMTModel.from_pretrained(model_name)
# Replace with the path to your locally downloaded model
model_path = "bart-large-mnli"

# Load the model from the local path
classifier = pipeline("zero-shot-classification", model=model_path, device=0 if torch.cuda.is_available() else -1)
candidate_labels = ["house rental", "phone", "car", "real estate", "clothing"]

@app.route('/classify', methods=['POST'])
def classify():
    data = request.data.decode('utf-8')  # convert bytes to string
    print(data)
    print("Classifying...")

    translated = model.generate(**tokenizer(data, return_tensors="pt", padding=True))

    sequence_to_classify = tokenizer.decode(translated[0], skip_special_tokens=True)
    print(sequence_to_classify)
    # Classify the input sequence based on the candidate labels
    result = classifier(sequence_to_classify, candidate_labels)
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
