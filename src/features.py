import pandas as pd

def merge_headlines(news_df):
    # Group multiple headlines per date
    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt']).dt.date
    grouped = news_df.groupby('publishedAt')['title'].apply(lambda x: ' '.join(x)).reset_index()
    grouped.columns = ['date', 'combined_headlines']
    return grouped

def label_stock_returns(sp500_df):
    sp500_df['date'] = pd.to_datetime(sp500_df['Date']).dt.date
    sp500_df['Target'] = (sp500_df['Return'].shift(-1) > 0).astype(int)  # Predicting next day
    return sp500_df[['date', 'Target']]

def prepare_dataset(news_df, sp500_df):
    headlines = merge_headlines(news_df)
    labels = label_stock_returns(sp500_df)
    merged = pd.merge(headlines, labels, on='date')
    return merged[['combined_headlines', 'Target']]
