# Intraday Crypto Trading Strategy (Signal Generator) with DEX Liquidity & Contract Check
# Requires: requests, pandas, numpy, ta, scikit-learn

import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import GradientBoostingClassifier
import datetime
import time

# === CONFIG ===
COIN_ID = 'ethereum'  # CoinGecko ID
VS_CURRENCY = 'usd'
INTERVAL = '1h'
WINDOW_SIZE = 24
MIN_VOLUME = 1_000_000

ETHERSCAN_API_KEY = 'YourEtherscanAPIKeyHere'
DEXSCREENER_URL = 'https://api.dexscreener.com/latest/dex/pairs/ethereum/'

# === FETCH PRICE DATA ===
def fetch_price_data(coin_id, vs_currency, interval):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': '2',  # ✅ Increased to bypass paid restriction
        # remove 'interval' to let CoinGecko choose appropriate granularity
    }
    r = requests.get(url, params=params)

    if r.status_code != 200:
        raise Exception(f"Error fetching price data: {r.status_code} — {r.text}")

    data = r.json()

    if 'prices' not in data or 'total_volumes' not in data:
        raise Exception(f"Unexpected response format:\n{data}")

    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['volume'] = [v[1] for v in data['total_volumes']]
    df.set_index('timestamp', inplace=True)

    # Optional: Keep only most recent 24 hours of hourly data
    df = df.loc[df.index > (df.index[-1] - pd.Timedelta(hours=24))]


    return df



# === SIMULATE ON-CHAIN METRICS ===
def simulate_onchain_metrics(df):
    df['active_addresses'] = df['volume'] / 10000 + np.random.randn(len(df)) * 10
    df['whale_transfers'] = (np.random.rand(len(df)) > 0.9).astype(int) * np.random.randint(1, 5, size=len(df))
    return df

# === ADD FEATURES ===
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
    x = np.array(series[-4:])
    if len(x) < 4:
        return 0
    return (x[-1] - x[0]) / x[0]

# === DEX LIQUIDITY & SLIPPAGE ===
def check_dex_liquidity(token_address):
    try:
        r = requests.get(DEXSCREENER_URL + token_address)
        data = r.json()
        pair = data['pairs'][0] if 'pairs' in data and data['pairs'] else {}
        return {
            'dex_liquidity_usd': float(pair.get('liquidity', {}).get('usd', 0)),
            'dex_volume_usd': float(pair.get('volume', {}).get('h1', 0)),
            'dex_price_impact': float(pair.get('priceChange', {}).get('h1', 0))
        }
    except:
        return {'dex_liquidity_usd': 0, 'dex_volume_usd': 0, 'dex_price_impact': 0}

# === ETHERSCAN CONTRACT SAFETY ===
def check_contract_safety(contract_address):
    url = f'https://api.etherscan.io/api?module=contract&action=getsourcecode&address={contract_address}&apikey={ETHERSCAN_API_KEY}'
    try:
        r = requests.get(url)
        data = r.json()
        if data['status'] == '1':
            info = data['result'][0]
            is_verified = info['ABI'] != 'Contract source code not verified'
            creator = info.get('ContractCreator', 'unknown')
            return {'verified': is_verified, 'creator': creator}
        return {'verified': False, 'creator': 'unknown'}
    except:
        return {'verified': False, 'creator': 'unknown'}

# === SIGNAL GENERATOR ===
def generate_signal(df, model, features, token_address, contract_address):
    latest = df.iloc[-1:]
    proba = model.predict_proba(latest[features])[0]
    grey_pred = grey_model_predict(df['price'])
    rsi = df['rsi'].iloc[-1]

    dex_data = check_dex_liquidity(token_address)
    contract_data = check_contract_safety(contract_address)

    if proba[1] > 0.6 and grey_pred > 0.01 and rsi < 70 and dex_data['dex_liquidity_usd'] > 50000 and contract_data['verified']:
        signal = 'BUY'
    elif proba[1] < 0.4 and grey_pred < -0.01 and rsi > 30:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    explanation = {
        'model_confidence_up': round(proba[1], 2),
        'grey_slope_estimate': round(grey_pred, 4),
        'rsi': round(rsi, 2),
        'volume': int(df['volume'].iloc[-1]),
        'dex_liquidity_usd': int(dex_data['dex_liquidity_usd']),
        'contract_verified': contract_data['verified'],
        'whale_transfers': int(df['whale_transfers'].iloc[-1])
    }

    return signal, explanation

# === MAIN ===
def run():
    token_address = '0xC02aaa39b223FE8D0A0E5C4F27eAD9083C756Cc2'  # WETH Example
    contract_address = token_address

    df = fetch_price_data(COIN_ID, VS_CURRENCY, INTERVAL)
    if df['volume'].iloc[-1] < MIN_VOLUME:
        print("Coin liquidity too low — skipping signal.")
        return

    df = simulate_onchain_metrics(df)
    df = add_features(df)
    model, features = train_model(df)
    signal, explanation = generate_signal(df, model, features, token_address, contract_address)

    print(f"Signal: {signal}")
    print("Reasoning:")
    for k, v in explanation.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    run()
