import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import GradientBoostingClassifier
import datetime
import time

# === CONFIG ===
VS_CURRENCY = 'usd'
INTERVAL = '1h'
WINDOW_SIZE = 24
MIN_VOLUME = 1_000_000


# === BADCOINS TO EXCLUDE ===
BADCOINS = { 'USDC', 'BUSD', 'TUSD', 'DAI', 'FDUSD', 'TRY', 'COP', 'ARS'}

# Binance API endpoints
BASE_URL = 'https://api.binance.com'
TICKER_24HR_ENDPOINT = '/api/v3/ticker/24hr'
KLINES_ENDPOINT = '/api/v3/klines'

# === FETCH PRICE DATA FROM BINANCE ===
def fetch_price_data(symbol, interval='1h', limit=100):
    """
    Fetches price data for a given symbol from Binance and returns it as a DataFrame.
    """
    url = BASE_URL + KLINES_ENDPOINT
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    print(f"Fetching data for {symbol}...")
    r = requests.get(url, params=params)
    data = r.json()
    
    if 'code' in data:
        raise Exception(f"Error fetching data: {data['msg']}")
    
    df = pd.DataFrame(data, columns=[ 
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df['price'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df.set_index('timestamp', inplace=True)
    return df


# === GET 5m DATA FOR GREY MODEL ===
def fetch_5min_data(symbol, hours=4):
    limit = hours * 12  # 12 candles per hour for 5m
    return fetch_price_data(symbol, interval='5m', limit=limit)


# === GET TOP COINS ===
def get_top_coins(n=10):
    """
    Fetches the top trading pairs by 24-hour volume from Binance and filters out non-USDT pairs.
    """
    print("Fetching top coins from Binance...")
    url = BASE_URL + TICKER_24HR_ENDPOINT
    r = requests.get(url)
    data = r.json()

    # Log raw response from the API
    #print("API Response:", data[:5])  # Show a small part of the response for inspection

    sorted_data = sorted(data, key=lambda x: float(x['quoteVolume']), reverse=True)

    symbols = []
    for item in sorted_data:
        symbol = item['symbol']
        if 'USDT' not in symbol:  # Ensure we only look for USDT pairs
            continue  
        if any(stable in symbol for stable in BADCOINS):
            continue  # Skip BADCOINS
        symbols.append(symbol)
        if len(symbols) >= n:
            break
    
    print(f"Found {len(symbols)} USDT pairs.")
    return symbols

# === SIMULATE ON-CHAIN METRICS ===
def simulate_onchain_metrics(df):
    df['active_addresses'] = df['volume'] / 10000 + np.random.randn(len(df)) * 10
    df['whale_transfers'] = (np.random.rand(len(df)) > 0.9).astype(int) * np.random.randint(1, 5, size=len(df))
    return df

# === ADD TECHNICAL INDICATORS ===
def add_features(df):
    df['returns'] = df['price'].pct_change()
    df['ma'] = df['price'].rolling(5).mean()
    df['volatility'] = df['returns'].rolling(5).std()
    df['rsi'] = RSIIndicator(df['price']).rsi()
    bb = BollingerBands(df['price'])
    df['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()
    return df.dropna()

# === TRAIN MODEL ===
def train_model(df):
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)
    features = ['returns', 'ma', 'volatility', 'rsi', 'bb_width', 'active_addresses', 'whale_transfers']
    X = df[features]
    y = df['target']
    model = GradientBoostingClassifier()
    model.fit(X[:-1], y[:-1])
    return model, features

# === GREY MODEL ===
def grey_model_predict(series):
    x = np.array(series[-len(series):])
    if len(x) < 4:
        return 0
    return (x[-1] - x[0]) / x[0]

# === SIGNAL GENERATOR ===
def generate_signal(df, model, features, symbol):
    latest = df.iloc[-1:]
    proba = model.predict_proba(latest[features])[0]

    try:
        grey_df = fetch_5min_data(symbol, hours=4)
        grey_pred = grey_model_predict(grey_df['price'])
    except Exception as e:
        print(f"  Grey model fetch failed for {symbol}: {e}")
        grey_pred = 0

    rsi = df['rsi'].iloc[-1]
    volume = df['volume'].iloc[-1]

    if proba[1] > 0.6 and grey_pred > 0.01 and rsi < 70 and volume > MIN_VOLUME:
        signal = 'BUY'
    elif proba[1] < 0.4 and grey_pred < -0.01 and rsi > 30:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    explanation = {
        'model_confidence_up': round(proba[1], 2),
        'grey_slope_estimate': round(grey_pred, 4),
        'rsi': round(rsi, 2),
        'volume': int(volume)
    }

    return signal, explanation


# === SCAN MARKET ===
def run_market_scan():
    signals_generated = 0
    scanned_pairs = 0
    total_pairs_to_scan = 10
    generated_signals = []
    
    top_coins = get_top_coins(10)
    print("Scanning market for USDT pairs...\n")
    if not top_coins:
        print("No USDT pairs found.")
        return
    
    for coin in top_coins:
        symbolForDf = coin
        print(f"Processing {coin}...")
        try:
            df = fetch_price_data(coin)
            if df['volume'].iloc[-1] < MIN_VOLUME:
                print(f"  Skipped {coin}: low volume")
                continue
            df = simulate_onchain_metrics(df)
            df = add_features(df)
            model, features = train_model(df)
            signal, explanation = generate_signal(df, model, features, coin)


            if signal != 'HOLD':
                signals_generated += 1
                generated_signals.append(f"Signal for {coin}: {signal}")
                print(f"  Signal: {signal}")
                for k, v in explanation.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  Signal: HOLD")

            # Stop after 10 pairs and at least 1 signals
            scanned_pairs += 1
            if scanned_pairs >= total_pairs_to_scan and signals_generated >= 1:
                break
        except Exception as e:
            print(f"  Skipped {coin}: {e}")
        time.sleep(3)
    
    # After scanning is done, print the generated signals
    print("\n--- Generated Signals ---")
    for signal in generated_signals:
        print(signal)

    # If signals generated are fewer than 1, keep scanning until the threshold is met
    if signals_generated < 1:
        print("\nNot enough signals generated. Continuing scan...")
        run_market_scan()

if __name__ == "__main__":
    run_market_scan()
