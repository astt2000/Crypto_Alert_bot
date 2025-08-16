"""
Multi-Timeframe Triple-Confirmation Crypto Alert Bot (Parallel + Dynamic + Patterns)
- Parallel fetching with ThreadPoolExecutor
- Dynamic thresholds loaded from crypto_config.json
- Adds candlestick pattern confirmation (Doji, Engulfing, Hammer, Shooting Star)
- Multiple endpoints with rotation + retry/backoff
- Merged Telegram alerts with Malaysia local time
"""

import time
import json
import os
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import pandas_ta as ta
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from random import shuffle

# -------------------------
# Load config from JSON
# -------------------------
CFG_PATH = "crypto_config.json"
if not os.path.exists(CFG_PATH):
    raise FileNotFoundError(f"Config file {CFG_PATH} not found.")

with open(CFG_PATH, "r") as f:
    user_cfg = json.load(f)

# Backward-compatible defaults
CONFIG = {
    # Core
    "symbols": user_cfg.get("symbols", []),
    "timeframes": user_cfg.get("timeframes", ["1h"]),
    "lookback": user_cfg.get("lookback", 500),
    "min_confirmations": user_cfg.get("min_confirmations", 3),
    "alert_confidence_threshold": user_cfg.get("alert_confidence_threshold", 80),
    "volume_multiplier": user_cfg.get("volume_multiplier", 1.5),
    "sleep_between_symbols": user_cfg.get("sleep_between_symbols", 0.0),
    # Parallelism
    "max_workers": user_cfg.get("max_workers", 8),
    # Telegram
    "telegram_bot_token": user_cfg.get("telegram_bot_token", ""),
    "telegram_chat_id": user_cfg.get("telegram_chat_id", ""),
    # Indicators & thresholds (dynamic)
    "thresholds": user_cfg.get("thresholds", {
        "rsi_mid": 50,
        "rsi_overbought": 75,
        "rsi_oversold": 25,
        "adx_strong": 25,
        "macd_min_hist": 0.0  # optional extra filter
    }),
    "ema": user_cfg.get("ema", {
        "fast": 50,
        "slow": 200
    }),
    "supertrend": user_cfg.get("supertrend", {
        "length": 10,
        "multiplier": 3.0
    }),
    # Endpoints & networking
    "endpoints": user_cfg.get("endpoints", [
        "https://data-api.binance.vision",  # Public Binance Data Mirror
        "https://api.binance.com",
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com"
    ]),
    "http_timeout": user_cfg.get("http_timeout", 12),
    "max_retries": user_cfg.get("max_retries", 3),
    "backoff_seconds": user_cfg.get("backoff_seconds", 1.5),
}

# In-memory state for consecutive confirmations
LAST_SIGNAL = {sym: {tf: {"side": None, "count": 0} for tf in CONFIG["timeframes"]} for sym in CONFIG["symbols"]}

# -------------------------
# HTTP helper with endpoint rotation + retry/backoff
# -------------------------
def fetch_ohlcv(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    endpoints = CONFIG["endpoints"][:]
    shuffle(endpoints)  # randomize to distribute load
    last_error = None

    for attempt in range(1, CONFIG["max_retries"] + 1):
        for base_url in endpoints:
            try:
                url = f"{base_url}/api/v3/klines"
                params = {"symbol": symbol, "interval": interval, "limit": limit}
                r = requests.get(url, params=params, timeout=CONFIG["http_timeout"])
                r.raise_for_status()
                klines = r.json()

                df = pd.DataFrame(klines, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
                ])
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                df = df[["open_time", "open", "high", "low", "close", "volume"]]
                df.columns = ["datetime", "open", "high", "low", "close", "volume"]
                df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

                print(f"[{symbol} {interval}] Data loaded from {base_url} (attempt {attempt})")
                return df.set_index("datetime")
            except Exception as e:
                last_error = e
                print(f"[{symbol} {interval}] Failed from {base_url} — {e} (attempt {attempt})")
                continue

        # exponential backoff
        sleep_s = CONFIG["backoff_seconds"] * (2 ** (attempt - 1))
        time.sleep(sleep_s)

    raise ConnectionError(f"All endpoints failed for {symbol} {interval}: {last_error}")

# -------------------------
# Candlestick patterns (simple implementations)
# -------------------------
def is_doji(o, h, l, c, tol=0.1) -> bool:
    body = abs(c - o)
    range_ = max(h - l, 1e-12)
    return body <= tol * range_

def is_bullish_engulfing(prev_o, prev_c, o, c) -> bool:
    return (prev_c < prev_o) and (c > o) and (o <= prev_c) and (c >= prev_o)

def is_bearish_engulfing(prev_o, prev_c, o, c) -> bool:
    return (prev_c > prev_o) and (c < o) and (o >= prev_c) and (c <= prev_o)

def is_hammer(o, h, l, c) -> bool:
    body = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return (lower_wick >= 2 * body) and (upper_wick <= body)

def is_shooting_star(o, h, l, c) -> bool:
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return (upper_wick >= 2 * body) and (lower_wick <= body)

def compute_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Evaluate last candle (and previous) patterns.
    Returns dict of boolean pattern signals for LONG/SHORT bias.
    """
    if len(df) < 2:
        return {
            "doji": False,
            "bull_engulf": False,
            "bear_engulf": False,
            "hammer": False,
            "shooting_star": False
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]

    o, h, l, c = last["open"], last["high"], last["low"], last["close"]
    po, ph, pl, pc = prev["open"], prev["high"], prev["low"], prev["close"]

    return {
        "doji": is_doji(o, h, l, c),
        "bull_engulf": is_bullish_engulfing(po, pc, o, c),
        "bear_engulf": is_bearish_engulfing(po, pc, o, c),
        "hammer": is_hammer(o, h, l, c),
        "shooting_star": is_shooting_star(o, h, l, c),
    }

# -------------------------
# Indicators
# -------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    ema_fast = CONFIG["ema"]["fast"]
    ema_slow = CONFIG["ema"]["slow"]
    st_len = CONFIG["supertrend"]["length"]
    st_mult = CONFIG["supertrend"]["multiplier"]

    df["EMA_FAST"] = ta.ema(df["close"], length=ema_fast)
    df["EMA_SLOW"] = ta.ema(df["close"], length=ema_slow)

    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["ADX"] = adx["ADX_14"]

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDs_12_26_9"]
    df["MACDh"] = macd["MACDh_12_26_9"]

    df["RSI"] = ta.rsi(df["close"], length=14)
    df["OBV"] = ta.obv(df["close"], df["volume"])
    df["OBV_diff_3"] = df["OBV"].diff(3)

    st = ta.supertrend(df["high"], df["low"], df["close"], length=st_len, multiplier=st_mult)
    df["Supertrend"] = st[f"SUPERT_{st_len}_{st_mult}"]

    return df

# -------------------------
# Signal evaluation (dynamic thresholds + patterns)
# -------------------------
def evaluate_signal(df: pd.DataFrame) -> Tuple[str, float, str, float, float, float, Dict[str, bool]]:
    th = CONFIG["thresholds"]
    last = df.iloc[-1]

    conds = {
        "trend_up": last["EMA_FAST"] > last["EMA_SLOW"],
        "trend_down": last["EMA_FAST"] < last["EMA_SLOW"],
        "adx_strong": last["ADX"] > th["adx_strong"],
        "macd_bull": last["MACD"] > last["MACD_signal"],
        "macd_bear": last["MACD"] < last["MACD_signal"],
        "rsi_bull": last["RSI"] > th["rsi_mid"],
        "rsi_bear": last["RSI"] < th["rsi_mid"],
        "obv_up": last["OBV_diff_3"] > 0,
        "obv_down": last["OBV_diff_3"] < 0,
        "supertrend_bull": last["close"] > last["Supertrend"],
        "supertrend_bear": last["close"] < last["Supertrend"],
        "macd_hist_ok": last["MACDh"] >= th.get("macd_min_hist", 0.0)
    }

    patterns = compute_patterns(df)
    # Pattern contributions (treated as additional checks)
    bull_checks = [
        conds["trend_up"], conds["adx_strong"], conds["macd_bull"], conds["rsi_bull"],
        conds["obv_up"], conds["supertrend_bull"], conds["macd_hist_ok"],
        patterns["bull_engulf"], patterns["hammer"]
    ]
    bear_checks = [
        conds["trend_down"], conds["adx_strong"], conds["macd_bear"], conds["rsi_bear"],
        conds["obv_down"], conds["supertrend_bear"],
        patterns["bear_engulf"], patterns["shooting_star"]
    ]

    long_pass = sum(bool(x) for x in bull_checks)
    short_pass = sum(bool(x) for x in bear_checks)

    # Normalize to percentage against max possible checks in each side
    long_conf = (long_pass / len(bull_checks)) * 100.0
    short_conf = (short_pass / len(bear_checks)) * 100.0

    if long_conf >= short_conf and long_conf > 0:
        side = "LONG" if long_pass >= math.ceil(0.5 * len(bull_checks)) else "HOLD"
        conf = long_conf
    elif short_conf > long_conf and short_conf > 0:
        side = "SHORT" if short_pass >= math.ceil(0.5 * len(bear_checks)) else "HOLD"
        conf = short_conf
    else:
        side, conf = "HOLD", 0.0

    strength = "STRONG" if conf >= CONFIG["alert_confidence_threshold"] else ""
    return side, round(conf, 1), strength, float(last["RSI"]), float(last["volume"]), float(df["volume"].iloc[-20:].mean()), patterns

# -------------------------
# Telegram alert
# -------------------------
def send_telegram(message: str) -> bool:
    if not CONFIG["telegram_bot_token"] or not CONFIG["telegram_chat_id"]:
        print("[telegram] Not configured.")
        return False
    url = f"https://api.telegram.org/bot{CONFIG['telegram_bot_token']}/sendMessage"
    payload = {"chat_id": CONFIG["telegram_chat_id"], "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=payload, timeout=10)
        ok = r.status_code == 200
        if not ok:
            print(f"[telegram] HTTP {r.status_code}: {r.text[:200]}")
        return ok
    except Exception as e:
        print(f"[telegram] Error: {e}")
        return False

# -------------------------
# Worker for a single (symbol, timeframe)
# -------------------------
def process_pair(sym: str, tf: str) -> Optional[Tuple[str, str, float, str, float, float, float, Dict[str, bool]]]:
    """
    Returns tuple: (timeframe, side, conf, strength, rsi, vol_now, vol_avg, patterns)
    or None on failure.
    """
    try:
        df = fetch_ohlcv(sym, tf, CONFIG["lookback"])
        df = compute_indicators(df)
        side, conf, strength, rsi, vol_now, vol_avg, patterns = evaluate_signal(df)

        # Volume filter
        if vol_now < CONFIG["volume_multiplier"] * vol_avg:
            side = "HOLD"

        # RSI bounding filters from dynamic thresholds
        rsi_overbought = CONFIG["thresholds"]["rsi_overbought"]
        rsi_oversold = CONFIG["thresholds"]["rsi_oversold"]
        if side == "LONG" and rsi > rsi_overbought:
            side = "HOLD"
        if side == "SHORT" and rsi < rsi_oversold:
            side = "HOLD"

        malaysia_time = datetime.utcnow() + timedelta(hours=8)
        time_str = malaysia_time.strftime("%Y-%m-%d %H:%M:%S")
        patt_txt = []
        if patterns["bull_engulf"]: patt_txt.append("BullEngulf")
        if patterns["bear_engulf"]: patt_txt.append("BearEngulf")
        if patterns["hammer"]: patt_txt.append("Hammer")
        if patterns["shooting_star"]: patt_txt.append("ShootingStar")
        if patterns["doji"]: patt_txt.append("Doji")

        if patt_txt:
            patt_str = " | Patterns: " + ",".join(patt_txt)
        else:
            patt_str = ""

        print(f"[{time_str}] [{tf}] {sym} — {side} ({conf}% {strength}) Vol:{vol_now:.2f} RSI:{rsi:.1f}{patt_str}")

        # optional sleep pacing (mostly unnecessary with thread pool, but kept for compatibility)
        if CONFIG["sleep_between_symbols"] > 0:
            time.sleep(CONFIG["sleep_between_symbols"])

        return (tf, side, conf, strength, rsi, vol_now, vol_avg, patterns)
    except ConnectionError as e:
        print(f"[{tf}] {sym} SKIPPED — Network error: {e}")
    except Exception as e:
        print(f"[{tf}] {sym} ERROR: {e}")
    return None

# -------------------------
# Main runner (parallel)
# -------------------------
def run_all():
    symbol_results: Dict[str, List[Tuple]] = {sym: [] for sym in CONFIG["symbols"]}

    tasks = list(product(CONFIG["symbols"], CONFIG["timeframes"]))
    if not tasks:
        print("No symbols/timeframes configured.")
        return

    max_workers = min(CONFIG["max_workers"], len(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(process_pair, sym, tf): (sym, tf) for sym, tf in tasks}
        for fut in as_completed(future_map):
            sym, tf = future_map[fut]
            res = fut.result()
            if res is None:
                continue
            tf_res = res  # (tf, side, conf, strength, rsi, vol_now, vol_avg, patterns)
            if tf_res[1] in ("LONG", "SHORT"):
                symbol_results[sym].append(tf_res)

    # Multi-timeframe agreement + consecutive confirmations + strong filter
    alerts_to_send: List[str] = []

    for sym, tf_signals in symbol_results.items():
        if not tf_signals:
            continue

        sides = [s[1] for s in tf_signals]
        final_side: Optional[str] = None
        if sides.count("LONG") >= 2:
            final_side = "LONG"
        elif sides.count("SHORT") >= 2:
            final_side = "SHORT"

        if not final_side:
            continue

        for tf, side, conf, strength, rsi, vol_now, vol_avg, patterns in tf_signals:
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
                patt_txt = []
                if patterns["bull_engulf"]: patt_txt.append("BullEngulf")
                if patterns["bear_engulf"]: patt_txt.append("BearEngulf")
                if patterns["hammer"]: patt_txt.append("Hammer")
                if patterns["shooting_star"]: patt_txt.append("ShootingStar")
                if patterns["doji"]: patt_txt.append("Doji")
                patt_str = f"\nPatterns: {', '.join(patt_txt)}" if patt_txt else ""
                alerts_to_send.append(
                    f"*{sym}* ({tf}) — {side}\nConfidence: *{conf}%* {strength}{patt_str}"
                )

    if alerts_to_send:
        malaysia_time = datetime.utcnow() + timedelta(hours=8)
        timestamp_str = malaysia_time.strftime("%Y-%m-%d %H:%M:%S")
        header = f"📅 Malaysia Time: {timestamp_str}"

        merged_msg = header + "\n\n" + "\n\n".join(alerts_to_send)
        send_telegram(merged_msg)
        print(f"[telegram] Sent merged alert:\n{merged_msg}")

if __name__ == "__main__":
    run_all()
