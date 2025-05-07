import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import datetime
import time
import logging
import sys

# === CONFIG ===
VS_CURRENCY = 'usd'
INTERVAL = '1h'  # Hourly data
WINDOW_SIZE = 24
MIN_VOLUME = 1_000_000  # Minimum quote volume in USD
TRADING_FEE = 0.04 / 100  # Binance futures taker fee
SLIPPAGE = 0.001  # 0.1% slippage
LEVERAGE = 10
STOP_LOSS = 0.02  # 2% stop-loss
TAKE_PROFIT = 0.04  # 4% take-profit
POSITION_SIZE = 0.1  # 10% of account per trade
MIN_DATA_ROWS = 20  # Minimum rows after feature engineering
DATA_LIMIT = 1000  # Increased for more robust training

BADCOINS = {'USDC', 'BUSD', 'TUSD', 'DAI', 'FDUSD', 'TRY', 'COP', 'ARS'}

# === SETUP LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# === FUTURES BINANCE ENDPOINTS ===
BASE_URL = 'https://fapi.binance.com'
TICKER_24HR_ENDPOINT = '/fapi/v1/ticker/24hr'
KLINES_ENDPOINT = '/fapi/v1/klines'
FUNDING_RATE_ENDPOINT = '/fapi/v1/premiumIndex'

# === FETCH PRICE DATA ===
def fetch_price_data(symbol, interval='1h', limit=DATA_LIMIT):
    try:
        url = BASE_URL + KLINES_ENDPOINT
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict) and 'code' in data:
            raise Exception(f"API error: {data['msg']}")

        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df['price'] = df['close'].astype(float)
        df['volume'] = df['quote_asset_volume'].astype(float)
        df.set_index('timestamp', inplace=True)
        logger.info(f"Fetched {len(df)} rows for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch price data for {symbol}: {e}")
        return None

# === FETCH FUNDING RATES ===
def fetch_funding_rate(symbol):
    try:
        url = BASE_URL + FUNDING_RATE_ENDPOINT
        params = {'symbol': symbol}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        logger.debug(f"Funding rate response for {symbol}: {data}")
        funding_rate = float(data.get('lastFundingRate', 0.0))
        return funding_rate
    except Exception as e:
        logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")
        return 0.0  # Fallback to zero

# === TOP COINS BY VOLUME ===
def get_top_futures_pairs(n=10):
    try:
        url = BASE_URL + TICKER_24HR_ENDPOINT
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        sorted_data = sorted(data, key=lambda x: float(x['quoteVolume']), reverse=True)

        symbols = []
        for item in sorted_data:
            symbol = item['symbol']
            if 'USDT' not in symbol or any(bad in symbol for bad in BADCOINS):
                continue
            symbols.append(symbol)
            if len(symbols) >= n:
                break
        logger.info(f"Retrieved {len(symbols)} top futures pairs")
        return symbols
    except Exception as e:
        logger.error(f"Failed to fetch top pairs: {e}")
        return []

# === INDICATORS ===
def add_features(df, symbol):
    try:
        df['returns'] = df['price'].pct_change()
        df['ma_5'] = df['price'].rolling(5).mean()
        df['ma_10'] = df['price'].rolling(10).mean()
        df['volatility'] = df['returns'].rolling(10).std()
        df['rsi'] = RSIIndicator(df['price'], window=14).rsi()
        df['ema_fast'] = EMAIndicator(df['price'], window=12).ema_indicator()
        df['ema_slow'] = EMAIndicator(df['price'], window=26).ema_indicator()
        bb = BollingerBands(df['price'], window=10)
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['price']
        df['volume_sma'] = df['volume'].rolling(10).mean()
        df['funding_rate'] = fetch_funding_rate(symbol)
        # Price derivative (dP/dt, USD/hour)
        df['price_derivative'] = (df['price'] - df['price'].shift(1)) / 1  # h=1 hour
        # 5-period SMA derivative
        df['sma_5'] = df['price'].rolling(5).mean()
        df['sma_derivative'] = (df['sma_5'] - df['sma_5'].shift(1)) / 1
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"After feature engineering, {len(df)} rows remain (from {initial_rows}) for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Error in feature engineering for {symbol}: {e}")
        return df

# === TRAIN MODEL ===
def train_model(df):
    try:
        df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
        features = [
            'returns', 'ma_5', 'ma_10', 'volatility', 'rsi', 'ema_fast', 'ema_slow',
            'bb_width', 'volume_sma', 'funding_rate', 'price_derivative', 'sma_derivative'
        ]
        
        X = df[features]
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Grid search for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
        model = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best model params: {grid_search.best_params_}")
        logger.info(f"Test accuracy: {grid_search.score(X_test, y_test):.4f}")
        
        return grid_search.best_estimator_, features
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None, []

# === BACKTESTING ===
# === CONFIG (Updated) ===
DATA_LIMIT = 2000  # ~83 days for robust training and testing

# === BACKTESTING (Updated) ===
def backtest_strategy(df, model, features, symbol, train_size=0.8):
    try:
        df = df.copy()
        df['signal'] = 0
        
        # Split data to match train_test_split in train_model
        train_rows = int(len(df) * train_size)
        test_df = df.iloc[train_rows:].copy()  # Backtest only on test set
        signals = test_df[features].copy()
        test_df.loc[signals.index, 'signal'] = model.predict(signals)
        
        balance = 10000  # Initial balance in USD
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        trades = []
        
        for i in range(1, len(test_df)):
            price = test_df['price'].iloc[i]
            signal = test_df['signal'].iloc[i]
            
            # Close position if stop-loss or take-profit hit
            if position != 0:
                if position == 1 and (price <= entry_price * (1 - STOP_LOSS) or price >= entry_price * (1 + TAKE_PROFIT)):
                    profit = (price - entry_price) * LEVERAGE * POSITION_SIZE * balance / entry_price
                    profit -= 2 * TRADING_FEE * POSITION_SIZE * balance
                    balance += profit
                    trades.append({'symbol': symbol, 'entry': entry_price, 'exit': price, 'profit': profit, 'time': test_df.index[i]})
                    position = 0
                elif position == -1 and (price >= entry_price * (1 + STOP_LOSS) or price <= entry_price * (1 - TAKE_PROFIT)):
                    profit = (entry_price - price) * LEVERAGE * POSITION_SIZE * balance / entry_price
                    profit -= 2 * TRADING_FEE * POSITION_SIZE * balance
                    balance += profit
                    trades.append({'symbol': symbol, 'entry': entry_price, 'exit': price, 'profit': profit, 'time': test_df.index[i]})
                    position = 0
            
            # Open new position
            if position == 0 and signal == 1:
                position = 1
                entry_price = price * (1 + SLIPPAGE)
            elif position == 0 and signal == 0:
                position = -1
                entry_price = price * (1 - SLIPPAGE)
        
        logger.info(f"Backtest period: {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} rows)")
        return balance, trades
    except Exception as e:
        logger.error(f"Error in backtesting for {symbol}: {e}")
        return 10000, []

# === GENERATE SIGNAL ===
def generate_signal(df, model, features, symbol):
    try:
        latest = df.iloc[-1:]
        proba = model.predict_proba(latest[features])[0]
        
        # EMA crossover and derivative for trend confirmation
        ema_fast = df['ema_fast'].iloc[-1]
        ema_slow = df['ema_slow'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        volume = df['volume'].iloc[-1]
        price_deriv = df['price_derivative'].iloc[-1]
        sma_deriv = df['sma_derivative'].iloc[-1]
        
        # Slightly loosened thresholds
        if (proba[1] > 0.6 and ema_fast > ema_slow and rsi < 70 and 
            volume > MIN_VOLUME and price_deriv > 0):
            signal = 'BUY'
        elif (proba[1] < 0.4 and ema_fast < ema_slow and rsi > 30 and 
              volume > MIN_VOLUME and price_deriv < 0):
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        explanation = {
            'model_confidence_up': round(proba[1], 2),
            'ema_fast': round(ema_fast, 2),
            'ema_slow': round(ema_slow, 2),
            'rsi': round(rsi, 2),
            'volume': int(volume),
            'price_derivative': round(price_deriv, 2),
            'sma_derivative': round(sma_deriv, 2)
        }
        
        return signal, explanation
    except Exception as e:
        logger.error(f"Error generating signal for {symbol}: {e}")
        return 'HOLD', {}

# === MARKET SCAN ===
def run_market_scan():
    signals_generated = 0
    scanned_pairs = 0
    total_pairs_to_scan = 10
    generated_signals = []
    backtest_results = []

    top_coins = get_top_futures_pairs(total_pairs_to_scan)
    logger.info(f"Scanning {len(top_coins)} USDT futures pairs...")

    for coin in top_coins:
        logger.info(f"Processing {coin}...")
        try:
            df = fetch_price_data(coin)
            if df is None or len(df) < 50 or df['volume'].iloc[-1] < MIN_VOLUME:
                logger.info(f"Skipped {coin}: low volume or insufficient data")
                continue

            df = add_features(df, coin)
            logger.info(f"Features added, {len(df)} rows remain for {coin}")
            if len(df) < MIN_DATA_ROWS:
                logger.info(f"Skipped {coin}: insufficient data after feature engineering ({len(df)} rows)")
                continue

            model, features = train_model(df)
            if model is None:
                logger.info(f"Skipped {coin}: model training failed")
                continue
            
            # Backtest
            # In run_market_scan
            final_balance, trades = backtest_strategy(df, model, features, coin, train_size=0.8)
            backtest_results.append({
                'symbol': coin,
                'final_balance': final_balance,
                'num_trades': len(trades),
                'profit': final_balance - 10000
            })
            logger.info(f"Backtest for {coin}: Final balance = ${final_balance:.2f}, Trades = {len(trades)}")

            # Generate signal
            signal, explanation = generate_signal(df, model, features, coin)
            logger.info(f"Signal: {signal}")
            for k, v in explanation.items():
                logger.info(f"  {k}: {v}")
            if signal != 'HOLD':
                signals_generated += 1
                signal_info = f"Signal for {coin}: {signal}"
                generated_signals.append(signal_info)

            scanned_pairs += 1
            if scanned_pairs >= total_pairs_to_scan and signals_generated >= 1:
                break

        except Exception as e:
            logger.error(f"Skipped {coin}: {e}")
        time.sleep(2)

    logger.info("\n--- Generated Signals ---")
    for signal in generated_signals:
        logger.info(signal)
    
    logger.info("\n--- Backtest Results ---")
    for result in backtest_results:
        logger.info(f"{result['symbol']}: Final balance = ${result['final_balance']:.2f}, "
                    f"Profit = ${result['profit']:.2f}, Trades = {result['num_trades']}")

if __name__ == "__main__":
    run_market_scan()
    