from flask import Flask, jsonify
from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch

app = Flask(__name__)
model_name = "opus-mt-tc-big-tr-en"
tokenizer = MarianTokenizer.from_pretrained(model_name, use_fast=False)
model = MarianMTModel.from_pretrained(model_name)
# Replace with the path to your locally downloaded model
model_path = "bart-large-mnli"

# Load the model from the local path
classifier = pipeline("zero-shot-classification", model=model_path, device=0 if torch.cuda.is_available() else -1)
candidate_labels = ["rental", "smartphones", "vehicles"]

@app.route('/classify')
def classify():
    print("Classifying...")
    src_text = "Ev tutmam gerekiyor"


    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))

    sequence_to_classify = tokenizer.decode(translated[0], skip_special_tokens=True)
    print("Still classifying...")
    # Classify the input sequence based on the candidate labels
    result = classifier(sequence_to_classify, candidate_labels)
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
