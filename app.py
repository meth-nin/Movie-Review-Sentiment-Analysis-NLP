from flask import Flask, request, jsonify
from model_utils import predict_sentiment

app = Flask(__name__)

@app.route('/')
def home():
    return "Movie Review Sentiment API is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review_text = data.get("review")
    
    if not review_text:
        return jsonify({"error": "No review provided"}), 400
    
    sentiment = predict_sentiment(review_text)
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)