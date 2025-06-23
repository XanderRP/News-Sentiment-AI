import yfinance as yf

def fetch_sp500(start="2020-01-01", end="2024-12-31"):
    sp500 = yf.download("^GSPC", start=start, end=end)
    sp500["Return"] = sp500["Close"].pct_change()
    sp500 = sp500[["Close", "Return"]].dropna().reset_index()
    return sp500