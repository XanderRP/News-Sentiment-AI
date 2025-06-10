# News Sentiment Analysis Stock AI

This project is an end-to-end machine learning pipeline that uses daily news headlines to predict movements in the S&P 500 index using deep learning and natural language processing.

## 📈 Project Overview

The model:
- Collects financial news headlines using the newsapi.org News API
- Collects historical S&P 500 data from Yahoo Finance
- Preprocesses text and encodes using tokenization and padding
- Trains a Bidirectional LSTM model using TensorFlow/Keras
- Predicts if the S&P 500 will go up or down the following day

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- pandas / numpy / scikit-learn
- yfinance
- requests
- dotenv for environment variable management

## 📁 Project Structure

```
News-Sentiment-AI/
├── main.py
├── .env
├── requirements.txt
├── src/
│   ├── fetch_news.py
│   ├── fetch_sp500.py
│   ├── preprocess.py
│   ├── model.py
│   └── features.py
```

## ✅ Future Improvements

- Integrate a Streamlit frontend
- Use financial-domain BERT models
- Expand to multi-index forecasting
- Add real-time prediction mode

## 📜 License

MIT License
