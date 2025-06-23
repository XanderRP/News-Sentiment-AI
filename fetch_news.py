import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NEWSAPI_KEY")
BASE_URL = "https://newsapi.org/v2/everything"

if not API_KEY:
    raise ValueError("NEWSAPI_KEY is missing. Make sure it's set in your .env file.")

def fetch_news(query="stock market", from_date=None, to_date=None, page_size=100):
    # Use last 7 days as default window
    today = datetime.today().date()
    max_lookback = today - timedelta(days=30)

    if to_date is None:
        to_date = today
    else:
        to_date = datetime.strptime(to_date, "%Y-%m-%d").date()

    if from_date is None:
        from_date = to_date - timedelta(days=7)
    else:
        from_date = datetime.strptime(from_date, "%Y-%m-%d").date()

    # Enforce free-tier limits
    if from_date < max_lookback:
        raise ValueError(f"Your plan only supports news from {max_lookback}. Requested from {from_date}.")

    all_articles = []
    for page in range(1, 6):  # Max 500 results
        params = {
            "q": query,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": page_size,
            "page": page,
            "apiKey": API_KEY
        }

        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if "articles" not in data:
            print("Error from NewsAPI:", data)
            return pd.DataFrame()

        all_articles.extend(data["articles"])

    return pd.DataFrame([{
        "publishedAt": a["publishedAt"],
        "title": a["title"],
        "source": a["source"]["name"]
    } for a in all_articles])
