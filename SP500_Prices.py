import os
import pandas as pd
import yfinance as yf

# Setting up the config
DATA_DIR = "data"  # folder where CSV will be saved
os.makedirs(DATA_DIR, exist_ok=True)

# Wikipedia has the S&P constituents circa Oct 26, 2024
WIKI_REV_URL = "https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid=1253614549"

# Picking dates to download price history
START_DATE = "2023-07-01"
END_DATE   = "2025-03-31"

# Creating functions to download
def get_sp500_tickers():
    df = pd.read_html(WIKI_REV_URL, header=0)[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return sorted(set(tickers))

def download_prices(tickers):
    df = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=False
    )
    # Extract only Adjusted Close
    if isinstance(df.columns, pd.MultiIndex):
        prices = df.xs("Adj Close", axis=1, level=1)
    else:
        prices = df[["Adj Close"]]
        prices.columns = [tickers[0]]
    prices.index.name = "Date"
    return prices

# main function, also want all tickers as their own csv so we can upload folder to gitub so our app can access it
if __name__ == "__main__":
    tickers = get_sp500_tickers()
    prices = download_prices(tickers)

    # Save CSV (plain text so Streamlit can read directly from GitHub)
    csv_path = os.path.join(DATA_DIR, "adj_close.csv")
    prices.to_csv(csv_path)

    print(f"Saved {prices.shape} price table to {csv_path}")
