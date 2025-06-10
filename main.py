import os
from src.fetch_news import fetch_news
from src.fetch_sp500 import fetch_sp500
from src.features import prepare_dataset
from src.preprocess import build_tokenizer, encode_text
from src.model import create_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import numpy as np

def main():
    print("Fetching news and stock data...")
    news_df = fetch_news()
    sp500_df = fetch_sp500()

    print("Preparing dataset...")
    dataset = prepare_dataset(news_df, sp500_df)

    if dataset.empty:
        print("No data available after merging. Check your date ranges.")
        return

    texts = dataset['combined_headlines'].tolist()
    labels = dataset['Target'].tolist()

    print("Tokenizing and encoding text...")
    tokenizer = build_tokenizer(texts, num_words=10000)
    X = encode_text(tokenizer, texts, max_len=100)
    y = np.array(labels)

    print("Splitting train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building and training model...")
    model = create_model(vocab_size=10000, max_len=100)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

    print("Evaluating model...")
    y_pred = model.predict(X_val).flatten()
    y_pred_labels = (y_pred > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_labels))
    print(f"Accuracy: {accuracy_score(y_val, y_pred_labels):.2f}")

if __name__ == "__main__":
    main()

