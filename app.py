from flask import Flask, request, jsonify
import pandas as pd
from transformers import pipeline, DistilBertTokenizerFast, TFDistilBertForSequenceClassification

app = Flask(__name__)

tokenizer = DistilBertTokenizerFast.from_pretrained("saved_model")
model = TFDistilBertForSequenceClassification.from_pretrained("saved_model")
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=512
)

@app.route('/classify', methods=['POST'])
def classifyIssue():
    print("hello")
    data = request.json
    print(data)
    text = data.get('text', '')
    if not text:
        return jsonify(
            {
                "error": "No Input"
            }
        ), 400
    
    classification = classifier(text)[0]['label']
    return jsonify(
        {
            "classification": classification
        }
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='127.0.0.1')