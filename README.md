# Movie Review Sentiment Analysis

A machine learning-powered API that automatically determines whether a movie review is positive or negative using Natural Language Processing techniques.

##  Project Overview

This project uses the IMDB Dataset of 50K movie reviews to develop a sentiment analysis system for movie reviews. The system compares multiple machine learning models and serves the best performing model through a REST API.

### Key Features

- Multiple ML Models: Logistic Regression, Naive Bayes, SVM and LSTM
- Comprehensive Evaluation: Accuracy, Precision, Recall, F1-Score and Confusion Matrix
- Hyperparameter Tuning: GridSearchCV optimization for best performance
- REST API: Simple Flask based API for real time predictions
- Text Preprocessing: Complete NLP pipeline with cleaning, tokenization and vectorization

### Dataset

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data


## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/meth-nin/Movie-Review-Sentiment-Analysis-NLP.git
cd movie-review-sentiment-analysis
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK data:

(it's in model_utils.py)

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

4. Ensure the IMDB Dataset CSV file in the project directory

### How to run this??

1. Train the models if not already trained:

```bash
python notebook.py
```

2. Start the API server:

```bash
python app.py
```

The API will be available at `http://127.0.0.1:5000/`

3. Test the API:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"review": "type your review here"}'
```

OR

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "type your review here"}'
```


## Model Performance Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.8852 | 0.8768 | 0.8984 | 0.8875 |
| Tuned Logistic Regression | 0.8849 | 0.8767 | 0.8978 | 0.8871 |
| SVM | 0.8790 | 0.8731 | 0.8891 | 0.8810 |
| LSTM | 0.8756 | 0.8562 | 0.9051 | 0.8800 |
| Naive Bayes | 0.8490 | 0.8473 | 0.8543 | 0.8508 |

Best Model: Logistic Regression achieved the highest F1-Score of 0.8875.

### Key Insights

- The dataset is perfectly balanced- 25 000 positive and 25 000 negative reviews
- Positive reviews were longer and contain more words than negative reviews
- Traditional ML models (Logistic Regression, SVM) performed competitively with the deep learning LSTM model
- Hyperparameter tuned model perfomed almost same as the base model


## API Usage

### Endpoint

POST `/predict`

### Request Format

```json
{
  "review": "Movie review"
}
```

### Response Format

```json
{
  "sentiment": "positive/negative"
}
```

### Example Usage

#### curl:

- Positive review example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This movie was absolutely fantastic! The acting was superb and the plot was engaging throughout."}'
```
Expected response: {"sentiment": "positive"}

- Negative review example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "Terrible movie with poor acting and confusing plot. Complete waste of time."}'
```

Expected response: {"sentiment": "negative"}


## Technical Implementation

### Data Preprocessing Pipeline

1. HTML Tag Removal: Using BeautifulSoup to clean HTML tags
2. Text Normalization: Converting to lowercase
3. URL/Special Character Removal: Regex-based cleaning
4. Tokenization: NLTK word tokenization
5. Stop Word Removal: Filtering common English stop words
6. Lemmatization: Reducing words to their base forms
7. TF-IDF Vectorization: Converting text to numerical features (5000 features)

### Model Architecture

#### Traditional ML Models

- Logistic Regression: Linear classifier with L2 regularization
- Naive Bayes: Multinomial variant for text classification
- SVM: Linear Support Vector Machine
- Hyperparameter Tuning: GridSearchCV with 3-fold cross-validation

#### Deep Learning Model - LSTM

- Embedding Layer: 10,000 vocabulary, 64-dimensional embeddings
- LSTM Layer: 64 units with dropout for regularization
- Dense Layers: 32 units (ReLU) + 1 unit (Sigmoid) for binary classification
- Training: 5 epochs, batch size 128, Adam optimizer


## 🔍 Data Analysis Highlights

- Dataset Size: 50,000 movie reviews- 25,000 positive, 25,000 negative
- Average Review Length: 500-1500 characters
- Word Count Distribution: Most reviews contain 100-300 words
- Class Balance: Perfect 50-50 split between positive and negative sentiments


### Project Link:

https://github.com/meth-nin/Movie-Review-Sentiment-Analysis-NLP/tree/main

## 📧 Contact

Methupa Ninduwara - meth2468nin@gmail.com