"""
Multi-Timeframe Triple-Confirmation Crypto Alert Bot (Mirror Only, No python-binance)
- Reads config from crypto_config.json
- Uses Binance Data Mirror for OHLCV
- Calculates EMA50, EMA200, ADX, MACD, RSI, OBV, Supertrend
- Merged Telegram alerts to avoid spam
- Includes Malaysia local time in console + Telegram alerts
"""

import time
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta
import requests

# -------------------------
# Load config from JSON
# -------------------------
CFG_PATH = "crypto_config.json"
if not os.path.exists(CFG_PATH):
    raise FileNotFoundError(f"Config file {CFG_PATH} not found.")

with open(CFG_PATH, "r") as f:
    user_cfg = json.load(f)

CONFIG = {
    "symbols": user_cfg.get("symbols", []),
    "timeframes": user_cfg.get("timeframes", ["1h"]),
    "lookback": user_cfg.get("lookback", 500),
    "min_confirmations": user_cfg.get("min_confirmations", 3),
    "confidence_threshold_percent": user_cfg.get("alert_confidence_threshold", 80),
    "volume_multiplier": user_cfg.get("volume_multiplier", 1.5),
    "telegram_bot_token": user_cfg.get("telegram_bot_token", ""),
    "telegram_chat_id": user_cfg.get("telegram_chat_id", ""),
    "sleep_between_symbols": user_cfg.get("sleep_between_symbols", 1.0),
}

# In-memory state for consecutive confirmations
LAST_SIGNAL = {sym: {tf: {"side": None, "count": 0} for tf in CONFIG["timeframes"]} for sym in CONFIG["symbols"]}

# -------------------------
# Binance REST API helper (mirror only)
# -------------------------
BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision"  # Public Binance Data Mirror
]


def fetch_ohlcv(symbol, interval, limit):
    last_error = None
    for base_url in BINANCE_ENDPOINTS:
        try:
            url = f"{base_url}/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            r = requests.get(url, params=params, timeout=20)  # increased timeout
            r.raise_for_status()
            klines = r.json()

            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df = df[["open_time", "open", "high", "low", "close", "volume"]]
            df.columns = ["datetime", "open", "high", "low", "close", "volume"]
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(
                float)

            print(f"[{symbol} {interval}] Data loaded from {base_url}")
            return df.set_index("datetime")
        except Exception as e:
            print(f"[{symbol} {interval}] Failed from {base_url} — {e}")
            last_error = e
            continue
    raise ConnectionError(f"Binance mirror failed for {symbol} {interval}: {last_error}")


# -------------------------
# Indicators
# -------------------------
def compute_indicators(df):
    df["EMA50"] = ta.ema(df["close"], length=50)
    df["EMA200"] = ta.ema(df["close"], length=200)
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["ADX"] = adx["ADX_14"]
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDs_12_26_9"]
    df["MACDh"] = macd["MACDh_12_26_9"]
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["OBV"] = ta.obv(df["close"], df["volume"])
    df["OBV_diff_3"] = df["OBV"].diff(3)
    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["Supertrend"] = st["SUPERT_10_3.0"]
    return df


# -------------------------
# Signal evaluation
# -------------------------
def evaluate_signal(df):
    last = df.iloc[-1]
    conds = {
        "trend_up": last["EMA50"] > last["EMA200"],
        "trend_down": last["EMA50"] < last["EMA200"],
        "adx_strong": last["ADX"] > 25,
        "macd_bull": last["MACD"] > last["MACD_signal"],
        "macd_bear": last["MACD"] < last["MACD_signal"],
        "rsi_bull": last["RSI"] > 50,
        "rsi_bear": last["RSI"] < 50,
        "obv_up": last["OBV_diff_3"] > 0,
        "obv_down": last["OBV_diff_3"] < 0,
        "supertrend_bull": last["close"] > last["Supertrend"],
        "supertrend_bear": last["close"] < last["Supertrend"],
    }

    long_pass = sum([conds["trend_up"], conds["adx_strong"], conds["macd_bull"], conds["rsi_bull"], conds["obv_up"],
                     conds["supertrend_bull"]])
    short_pass = sum(
        [conds["trend_down"], conds["adx_strong"], conds["macd_bear"], conds["rsi_bear"], conds["obv_down"],
         conds["supertrend_bear"]])

    long_conf = (long_pass / 6) * 100
    short_conf = (short_pass / 6) * 100

    if long_conf >= short_conf and long_conf > 0:
        side = "LONG" if long_pass >= 4 else "HOLD"
        conf = long_conf
    elif short_conf > long_conf and short_conf > 0:
        side = "SHORT" if short_pass >= 4 else "HOLD"
        conf = short_conf
    else:
        side = "HOLD"
        conf = 0

    strength = "STRONG" if conf >= CONFIG["confidence_threshold_percent"] else ""
    return side, round(conf, 1), strength, last["RSI"], last["volume"], df["volume"].iloc[-20:].mean()


# -------------------------
# Telegram alert
# -------------------------
def send_telegram(message):
    if not CONFIG["telegram_bot_token"] or not CONFIG["telegram_chat_id"]:
        print("[telegram] Not configured.")
        return False
    url = f"https://api.telegram.org/bot{CONFIG['telegram_bot_token']}/sendMessage"
    payload = {"chat_id": CONFIG["telegram_chat_id"], "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"[telegram] Error: {e}")
        return False


# -------------------------
# Main runner
# -------------------------
def run_all():
    symbol_results = {}
    for sym in CONFIG["symbols"]:
        symbol_results[sym] = []
        for tf in CONFIG["timeframes"]:
            try:
                df = fetch_ohlcv(sym, tf, CONFIG["lookback"])
                df = compute_indicators(df)
                side, conf, strength, rsi, vol_now, vol_avg = evaluate_signal(df)

                # Volume filter
                if vol_now < CONFIG["volume_multiplier"] * vol_avg:
                    side = "HOLD"
                # RSI filter
                if side == "LONG" and rsi > 75:
                    side = "HOLD"
                if side == "SHORT" and rsi < 25:
                    side = "HOLD"

                malaysia_time = datetime.utcnow() + timedelta(hours=8)
                time_str = malaysia_time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{time_str}] [{tf}] {sym} — {side} ({conf}% {strength}) Vol:{vol_now:.2f} RSI:{rsi:.1f}")

                if side in ("LONG", "SHORT"):
                    symbol_results[sym].append((tf, side, conf, strength))

                time.sleep(CONFIG["sleep_between_symbols"])

            except ConnectionError as e:
                print(f"[{tf}] {sym} SKIPPED — Network error: {e}")
                continue
            except Exception as e:
                print(f"[{tf}] {sym} ERROR: {e}")
                continue

    # Multi-timeframe agreement
    alerts_to_send = []
    for sym, tf_signals in symbol_results.items():
        if not tf_signals:
            continue
        sides = [s[1] for s in tf_signals]
        if sides.count("LONG") >= 2:
            final_side = "LONG"
        elif sides.count("SHORT") >= 2:
            final_side = "SHORT"
        else:
            continue

        for tf, side, conf, strength in tf_signals:
            prev_state = LAST_SIGNAL[sym][tf]
            if side == prev_state["side"]:
                prev_state["count"] += 1
            else:
                prev_state["side"] = side
                prev_state["count"] = 1

            if (
                    prev_state["count"] == CONFIG["min_confirmations"]
                    and strength == "STRONG"
                    and side == final_side
            ):
                alerts_to_send.append(f"*{sym}* ({tf}) — {side}\nConfidence: *{conf}%* {strength}")

    if alerts_to_send:
        malaysia_time = datetime.utcnow() + timedelta(hours=8)
        timestamp_str = malaysia_time.strftime("%Y-%m-%d %H:%M:%S")
        header = f"📅 Malaysia Time: {timestamp_str}"

        merged_msg = header + "\n\n" + "\n\n".join(alerts_to_send)
        send_telegram(merged_msg)
        print(f"[telegram] Sent merged alert:\n{merged_msg}")


if __name__ == "__main__":
    run_all()
