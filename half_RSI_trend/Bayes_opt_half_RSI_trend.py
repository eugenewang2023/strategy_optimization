#!/usr/bin/env python3
"""
bayes_opt_half_RSI_trend.py
(NO SHIFT + SINGLE Trend Channel set + PF objective + per-ticker SIGMOID weights)

Matches TradingView Pine strategy (bar-based backtest model):
"half_RSI_hr (HA signals + Real risk + minimal future trend filter)"

What this version matches from the Pine you pasted (the merged-channel version):
- Signals use Heikin Ashi derived from REAL OHLC:
    sigClose = haClose
    sigHL2   = (haHigh + haLow)/2
- RSI/HOT logic (Pine-style):
    fast_window = round(slow_window/2)
    fast_rsi = rsi(sigClose, fast_window) (shift fixed to 0)
    slow_rsi = 100 - rsi(sigClose, slow_window)   <-- inverted like Pine
    hot = fast_rsi - slow_rsi
    hot_sm = sma(hot, smooth_len)
    cross_up = crossover(hot_sm, 0)
- Trend decision uses ONE set of channel inputs ONLY:
    channelLength, channelMulti (channelMulti computed for parity; not required for trend decision)
    midChannel = sma(sigHL2, channelLength)
    uptrend    = sigClose > midChannel
    above_lowerTF uses lowerTF_Length = max(1, channelLength//2),
      midChannel_lowerTF = sma(sigHL2, lowerTF_Length)
- Minimal "future trend filter" is merged into same channel midline:
    futureOK = sigClose > midChannel
  (i.e., no futureLen/futureMulti)

Risk control (SL/TP/Trailing) matches Pine logic:
- Risk series use REAL OHLC
- ATR is Wilder RMA ATR on REAL OHLC
- TP touched by REAL high, TP crossdown by REAL close vs take_profit_low
- Stop breach by REAL low vs trail_stop_level
- Trailing stop only ratchets upward

Optimization objective (per-ticker SIGMOID weighting; then mean across tickers):
- For each ticker:
    if trades < min_trades: score_ticker = 0
    else:
      pf_for_score = min(PF_raw, pf_cap)   (pf_cap is used ONLY inside scoring)
      pf_w         = sigmoid(pf_beta * (pf_for_score - pf_center))
      tr_w         = sigmoid(beta    * (trades      - center))
      score_ticker = pf_w * tr_w
- Overall trial score = mean(score_ticker over tickers)
Notes:
- PF_raw is computed from realized trade PnL NET of commissions.
- pf_cap is NOT applied to stored/printed PF values; it is used only inside scoring.

Other features:
- Optional penalty term (if enabled) based on fraction of tickers with maxdd < 0 and very-low trade counts
- COST FLOOR for gross loss on losing trades when computing PF_raw:
    loss_floor = cost * INITIAL_CAPITAL
    for each non-winning trade: gross_loss += max(-pnl, loss_floor)
- fill_mode: "same_close" or "next_open"
"""

import json, math, time, random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional plotting for equity curves
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

# ----------------- USER CONFIG -----------------
DATA_DIR = Path("data")
OUT_DIR  = Path("output")

INITIAL_CAPITAL = 100000.0
POSITION_SIZE_PCT = 0.10
COMMISSION_PCT = 0.006  # 0.6% of INITIAL_CAPITAL per completed trade (round-trip)
# ------------------------------------------------


# =========================
# Helpers: SMA, Wilder RMA, ATR, RSI (TradingView-style)
# =========================

def sma(x: np.ndarray, length: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0:
        return out
    csum = np.cumsum(np.nan_to_num(x, nan=0.0))
    isn = np.isnan(x).astype(np.int32)
    nan_csum = np.cumsum(isn)
    for i in range(length - 1, n):
        nan_count = nan_csum[i] - (nan_csum[i - length] if i >= length else 0)
        if nan_count > 0:
            out[i] = np.nan
        else:
            s = csum[i] - (csum[i - length] if i >= length else 0.0)
            out[i] = s / length
    return out


def rma_wilder(x: np.ndarray, length: int) -> np.ndarray:
    """
    TradingView ta.rma(): Wilder smoothing.
    Seed = SMA(length) at first full non-na window.
    alpha = 1/length.
    """
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n < length:
        return out

    # seed at first full non-na window
    if np.isnan(x[:length]).any():
        start = None
        for i in range(length - 1, n):
            w = x[i - length + 1:i + 1]
            if not np.isnan(w).any():
                out[i] = float(np.mean(w))
                start = i + 1
                break
        if start is None:
            return out
    else:
        out[length - 1] = float(np.mean(x[:length]))
        start = length

    alpha = 1.0 / float(length)
    for i in range(start, n):
        if np.isnan(x[i]) or np.isnan(out[i - 1]):
            out[i] = np.nan
        else:
            out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    tr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return tr
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i - 1]):
            tr[i] = np.nan
            continue
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    return tr


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """Match Pine ta.atr(length): RMA of true range."""
    tr = true_range(high, low, close)
    return rma_wilder(tr, length)


def rsi_tv(price: np.ndarray, length: int) -> np.ndarray:
    """
    Match TradingView ta.rsi():
    rsi = 100 - 100 / (1 + rma(gain, len) / rma(loss, len))
    """
    n = len(price)
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n < 2:
        return out

    ch = np.diff(price, prepend=np.nan)
    gain = np.where(ch > 0, ch, 0.0)
    loss = np.where(ch < 0, -ch, 0.0)

    avg_gain = rma_wilder(gain, length)
    avg_loss = rma_wilder(loss, length)

    for i in range(n):
        ag = avg_gain[i]
        al = avg_loss[i]
        if np.isnan(ag) or np.isnan(al):
            out[i] = np.nan
        else:
            if al == 0.0:
                out[i] = 100.0 if ag > 0 else 0.0
            else:
                rs = ag / al
                out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


# =========================
# Heikin Ashi from REAL OHLC
# =========================

def heikin_ashi_from_real(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    ha_close = (open_ + high + low + close) / 4.0

    ha_open = np.full(n, np.nan, dtype=np.float64)
    ha_high = np.full(n, np.nan, dtype=np.float64)
    ha_low  = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return ha_open, ha_high, ha_low, ha_close

    ha_open[0] = (open_[0] + close[0]) / 2.0
    ha_high[0] = max(high[0], ha_open[0], ha_close[0])
    ha_low[0]  = min(low[0],  ha_open[0], ha_close[0])

    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high[i] = max(high[i], ha_open[i], ha_close[i])
        ha_low[i]  = min(low[i],  ha_open[i], ha_close[i])

    return ha_open, ha_high, ha_low, ha_close


# =========================
# Pine-like cross helper
# =========================

def crossover(cur_x: float, prev_x: float, cur_y: float, prev_y: float) -> bool:
    return (cur_x > cur_y) and (prev_x <= prev_y)


# =========================
# PF from realized trades (net commissions) + COST FLOOR
# =========================

def profit_factor_from_trades(
    trades: List[Dict[str, Any]],
    cost: float,
    total_capital: float,
    commission_pct: float,     # <-- round-trip fraction of capital
) -> Tuple[float, float, float]:
    """
    PF uses:
      • price PnL
      • cost floor on losing trades
      • + fixed round-trip commission added to GROSS LOSS for every completed trade

    commission per trade = commission_pct * total_capital

    Guarantees:
      • if ≥1 trade: gross_loss > 0 → PF finite
      • if 0 trades: PF = 0
    """
    gp = 0.0
    gl = 0.0
    loss_floor = cost * total_capital
    commission_per_trade = commission_pct * total_capital

    completed = 0

    for t in trades:
        if t.get("ignored", False):
            continue
        if "fill_price" not in t or "exit_price" not in t or "qty" not in t:
            continue

        qty = float(t["qty"])
        entry = float(t["fill_price"])
        exitp = float(t["exit_price"])

        completed += 1

        pnl = qty * (exitp - entry)   # PURE price PnL

        if pnl > 0:
            gp += pnl
        else:
            gl += max(-pnl, loss_floor)

        # always charge the round-trip commission
        gl += commission_per_trade

    if completed == 0:
        return 0.0, 0.0, 0.0

    return gp / gl, gp, gl


# =========================
# Sigmoid
# =========================

def sigmoid(z: float) -> float:
    if z > 60.0:
        return 1.0
    if z < -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


# =========================
# Backtest
# =========================

@dataclass
class HalfRSIParams:
    # SINGLE channel set used for ALL trend decisions (including "futureOK")
    channelLength: int = 12
    channelMulti: float = 3.36  # parity only; channel bounds not required for trend decision

    atrPeriod: int = 57
    slMultiplier: float = 7.95
    tpMultiplier: float = 0.7923
    trailMultiplier: float = 1.457

    slow_window: int = 16
    smooth_len: int = 3

    initial_capital: float = INITIAL_CAPITAL
    position_size_pct: float = POSITION_SIZE_PCT
    commission_pct: float = COMMISSION_PCT

    fill_mode: str = "same_close"


def backtest_half_rsi_hr(df: pd.DataFrame, p: HalfRSIParams, cost: float) -> Dict[str, Any]:
    df = df.copy().reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    real_o = df["open"].astype(float).to_numpy()
    real_h = df["high"].astype(float).to_numpy()
    real_l = df["low"].astype(float).to_numpy()
    real_c = df["close"].astype(float).to_numpy()
    n = len(df)

    if n < 5:
        return {
            "equity_curve": [p.initial_capital],
            "trades": [],
            "sharpe": 0.0,
            "maxdd": 0.0,
            "num_trades": 0,
            "final_equity": p.initial_capital,
            "total_return": 0.0,
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
        }

    # HA signals from REAL OHLC
    _, ha_h, ha_l, ha_c = heikin_ashi_from_real(real_o, real_h, real_l, real_c)
    sigClose = ha_c
    sigHL2 = (ha_h + ha_l) / 2.0

    # RISK series are REAL
    riskClose = real_c
    riskHigh  = real_h
    riskLow   = real_l

    # ATR for exits on REAL
    atr_exit = atr_wilder(real_h, real_l, real_c, int(p.atrPeriod))

    # =========================
    # Trend filters (signals) - SINGLE channel set
    # =========================
    chLen = max(1, int(p.channelLength))
    midChannel = sma(sigHL2, chLen)
    uptrend = sigClose > midChannel

    lowerTF_Length = max(1, int(chLen // 2))
    midChannel_lowerTF = sma(sigHL2, lowerTF_Length)
    above_lowerTF = sigClose > midChannel_lowerTF

    # Minimal future trend filter merged into same midChannel:
    # Pine merged version: futureOK = sigClose > midChannel
    futureOK = sigClose > midChannel

    # =========================
    # RSI signals on HA close (SHIFT fixed to 0) - match Pine pasted
    # =========================
    slow_len = max(2, int(p.slow_window))
    fast_window = max(1, int(round(float(slow_len) / 2.0)))

    fast_rsi = rsi_tv(sigClose, fast_window)
    slow_rsi = 100.0 - rsi_tv(sigClose, slow_len)  # inverted like Pine

    hot = fast_rsi - slow_rsi
    sm = max(1, int(p.smooth_len))
    hot_sm = sma(hot, sm)

    # =========================
    # State
    # =========================
    in_pos = False
    pending_entry = False
    entry_fill_idx = None
    qty = 0.0
    cash = float(p.initial_capital)

    equity_curve = np.zeros(n, dtype=np.float64)
    trades: List[Dict[str, Any]] = []

    # Pine-like vars (persist while in position)
    tp_touched = False
    tp_crossdown = False
    trail_offset = np.nan
    trail_stop_level = np.nan
    take_profit_high = np.nan
    take_profit_low = np.nan

    prev_hot_sm = hot_sm[0] if n > 0 else np.nan

    for i in range(n):
        # Fill pending entry at next open
        if p.fill_mode == "next_open" and pending_entry and entry_fill_idx == i:
            fill_price = real_o[i]
            pos_value = cash * float(p.position_size_pct)
            qty = pos_value / fill_price if fill_price > 0 else 0.0
            cash -= pos_value

            in_pos = True
            pending_entry = False

            trades[-1].update({
                "fill_idx": int(i),
                "fill_price": float(fill_price),
                "commission_entry": 0,
                "qty": float(qty),
            })

        # cross_up = ta.crossover(hot_sm, 0)
        if i > 0 and (not np.isnan(hot_sm[i])) and (not np.isnan(prev_hot_sm)):
            cross_up = crossover(hot_sm[i], prev_hot_sm, 0.0, 0.0)
        else:
            cross_up = False

        # longCondition = cross_up and (uptrend or above_lowerTF) and futureOK
        ok_trend = False
        ok_future = False
        if not np.isnan(uptrend[i]) and not np.isnan(above_lowerTF[i]):
            ok_trend = bool(uptrend[i] or above_lowerTF[i])
        if not np.isnan(futureOK[i]):
            ok_future = bool(futureOK[i])

        longCondition = bool(cross_up and ok_trend and ok_future)

        # ENTRY (levels set using REAL close + REAL atr)
        if longCondition and (not in_pos) and (not pending_entry):
            tp_touched = False
            tp_crossdown = False

            atr_i = atr_exit[i]
            if (not np.isnan(atr_i)) and (not np.isnan(riskClose[i])):
                trail_stop_level = riskClose[i] - (atr_i * float(p.slMultiplier))

                min_trail_offset = atr_i * 0.5
                trail_offset = max(atr_i * float(p.trailMultiplier), min_trail_offset)

                take_profit_low = riskClose[i] + (atr_i * float(p.tpMultiplier))
                take_profit_high = take_profit_low + trail_offset

                trades.append({
                    "signal_idx": int(i),
                    "side": "LONG",
                    "signal_price": float(riskClose[i]),
                    "slow_window": int(slow_len),
                    "fast_window": int(fast_window),
                    "smooth_len": int(sm),
                    "channelLength": int(chLen),
                    "channelMulti": float(p.channelMulti),
                })

                if p.fill_mode == "same_close":
                    fill_price = riskClose[i]
                    pos_value = cash * float(p.position_size_pct)
                    qty = pos_value / fill_price if fill_price > 0 else 0.0
                    cash -= pos_value

                    in_pos = True
                    trades[-1].update({
                        "fill_idx": int(i),
                        "fill_price": float(fill_price),
                        "commission_entry": 0,
                        "qty": float(qty),
                    })
                else:
                    if i + 1 < n:
                        pending_entry = True
                        entry_fill_idx = i + 1
                    else:
                        trades[-1].update({"ignored": True})

        # POSITION MANAGEMENT / EXITS (ALL REAL)
        if in_pos:
            # TP touched by REAL high
            if (not tp_touched) and (not np.isnan(take_profit_high)) and (not np.isnan(riskHigh[i])) and (riskHigh[i] > take_profit_high):
                tp_touched = True

            # After TP touched: trail TP-low on REAL close
            if tp_touched and (not np.isnan(take_profit_low)) and (not np.isnan(riskClose[i])) and (not np.isnan(trail_offset)):
                tp_crossdown = bool(riskClose[i] < take_profit_low)
                new_low = riskClose[i] - trail_offset
                take_profit_low = max(take_profit_low, new_low)

            # Exit logic
            if (not np.isnan(riskLow[i])) and (not np.isnan(trail_stop_level)) and (riskLow[i] < trail_stop_level):
                longExit = True
                exit_reason = "SL_breach(realLow < trail_stop_level)"
            elif tp_crossdown:
                longExit = True
                exit_reason = "TP_crossdown(realClose < take_profit_low)"
            else:
                longExit = False
                exit_reason = None

            if longExit:
                # strategy.close model
                if p.fill_mode == "next_open":
                    if i + 1 < n:
                        fill_price = real_o[i + 1]
                        fill_idx = i + 1
                    else:
                        fill_price = riskClose[i]
                        fill_idx = i
                else:
                    fill_price = riskClose[i]
                    fill_idx = i

                exit_value = qty * fill_price
                commission = exit_value * float(p.commission_pct)
                cash += (exit_value - commission)

                trades[-1].update({
                    "exit_idx": int(fill_idx),
                    "exit_price": float(fill_price),
                    "commission_exit": float(commission),
                    "reason": exit_reason
                })

                qty = 0.0
                in_pos = False
                pending_entry = False
                entry_fill_idx = None

                tp_touched = False
                tp_crossdown = False
                trail_offset = np.nan
                trail_stop_level = np.nan
                take_profit_high = np.nan
                take_profit_low = np.nan
            else:
                # Trailing stop moves only upward
                atr_i = atr_exit[i]
                if (not np.isnan(atr_i)) and (not np.isnan(riskHigh[i])) and (not np.isnan(trail_stop_level)):
                    new_stop = riskHigh[i] - (atr_i * float(p.slMultiplier))
                    trail_stop_level = max(trail_stop_level, new_stop)
        else:
            tp_touched = False
            tp_crossdown = False
            trail_offset = np.nan
            trail_stop_level = np.nan
            take_profit_high = np.nan
            take_profit_low = np.nan

        # Mark-to-market equity (REAL close)
        mtm = qty * real_c[i] if in_pos else 0.0
        equity_curve[i] = cash + mtm
        prev_hot_sm = hot_sm[i]

    # Force close at end (REAL close)
    if in_pos and qty > 0:
        fill_price = real_c[-1]
        exit_value = qty * fill_price
        commission = exit_value * float(p.commission_pct)
        cash += (exit_value - commission)
        trades[-1].update({
            "exit_idx": int(n - 1),
            "exit_price": float(fill_price),
            "commission_exit": float(commission),
            "reason": "endClose"
        })
        qty = 0.0
        in_pos = False
        equity_curve[-1] = cash

    eq = equity_curve.copy()
    if len(eq) < 2:
        sharpe = 0.0
        maxdd = 0.0
    else:
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        sharpe = float((np.mean(rets) / (np.std(rets) + 1e-12)) * math.sqrt(252.0)) if len(rets) > 1 else 0.0
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.maximum(peak, 1e-12)
        maxdd = float(np.min(dd))

    final_equity = float(eq[-1])
    total_return = float(final_equity / float(p.initial_capital) - 1.0)

    completed = [t for t in trades if ("exit_price" in t and "fill_price" in t and not t.get("ignored", False))]
    num_trades = len(completed)

    pf, gp, gl = profit_factor_from_trades(
        trades,
        cost=float(cost),
        total_capital=float(p.initial_capital),
        commission_pct=float(COMMISSION_PCT),
    )

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "sharpe": float(sharpe),
        "maxdd": float(maxdd),
        "num_trades": int(num_trades),
        "trades": trades,
        "equity_curve": eq.tolist(),
        "profit_factor": float(pf),
        "gross_profit": float(gp),
        "gross_loss": float(gl),
    }


# =========================
# Optimization (Optuna)
# =========================

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_files() -> Dict[str, pd.DataFrame]:
    files = list(DATA_DIR.glob("*.parquet")) + list(DATA_DIR.glob("*.csv"))
    files.sort()
    data: Dict[str, pd.DataFrame] = {}
    for fp in files:
        try:
            df = pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
            data[fp.stem.upper()] = df
        except Exception as e:
            print(f"Failed to load {fp.name}: {e}")
    return data


def run_optuna_optimization(
    n_trials: int = 200,
    n_files: int = 200,
    fill_mode: str = "same_close",
    study_name: Optional[str] = None,
    seed: int = 42,
    cost: float = 0.006,
    penalty: bool = False,

    # trade-weight (per-ticker)
    min_trades: int = 4,
    center: float = 10.0,
    beta: float = 1.0,

    # PF-weight (per-ticker)
    pf_center: float = 10.0,
    pf_beta: float = 1.0,

    # pf_cap used ONLY in scoring
    pf_cap: float = 100.0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    ensure_dirs()
    data = load_files()
    if not data:
        raise SystemExit(f"No .csv/.parquet found in {DATA_DIR}. Put files there and re-run.")

    tickers_all = sorted(list(data.keys()))

    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
    except Exception as e:
        raise SystemExit("Optuna not installed. pip install optuna") from e

    if study_name is None:
        study_name = f"half_rsi_trend_pf_sig_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    storage_path = OUT_DIR / f"{study_name}.db"
    storage_url = f"sqlite:///{storage_path}"

    sampler = TPESampler(seed=seed, n_startup_trials=30, multivariate=True)
    pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    rng = random.Random(seed)

    def choose_tickers_for_trial() -> List[str]:
        if n_files is None or n_files <= 0 or n_files >= len(tickers_all):
            return tickers_all
        return rng.sample(tickers_all, k=n_files)

    pf_cap_score = float(pf_cap) if float(pf_cap) > 0 else 1.0
    min_trades_i = max(0, int(min_trades))

    def ticker_score(pf_raw: float, num_trades: int) -> float:
        if int(num_trades) < min_trades_i:
            return 0.0

        pf_for_score = float(pf_raw)
        if math.isfinite(pf_for_score):
            pf_for_score = min(pf_for_score, pf_cap_score)
        else:
            pf_for_score = pf_cap_score

        # NO pf_norm term (no division by pf_cap)
        pf_w = sigmoid(float(pf_beta) * (pf_for_score - float(pf_center)))
        tr_w = sigmoid(float(beta) * (float(num_trades) - float(center)))
        return float(pf_w) * float(tr_w)

    def opt_fn(trial):
        slowW = trial.suggest_int("slow_window", 14, 80)
        smth  = trial.suggest_int("smooth_len", 3, 30)

        atrP  = trial.suggest_int("atrPeriod", 5, 60)
        slM   = trial.suggest_float("slMultiplier", 0.5, 8.0)
        tpM   = trial.suggest_float("tpMultiplier", 0.5, 10.0)
        trM   = trial.suggest_float("trailMultiplier", 0.5, 10.0)

        # SINGLE channel set for trend decisions (optimize this if you want)
        chLen = trial.suggest_int("channelLength", 10, 200)
        chMul = trial.suggest_float("channelMulti", 0.5, 8.0)

        tickers = choose_tickers_for_trial()
        if not tickers:
            return -1e9

        sum_score = 0.0
        sum_pf_raw = 0.0
        sum_trades = 0
        num_neg_dd = 0

        for tk in tickers:
            params = HalfRSIParams(
                channelLength=int(chLen),
                channelMulti=float(chMul),

                slow_window=int(slowW),
                smooth_len=int(smth),

                atrPeriod=int(atrP),
                slMultiplier=float(slM),
                tpMultiplier=float(tpM),
                trailMultiplier=float(trM),

                initial_capital=INITIAL_CAPITAL,
                position_size_pct=POSITION_SIZE_PCT,
                commission_pct=COMMISSION_PCT,
                fill_mode=fill_mode,
            )

            m = backtest_half_rsi_hr(data[tk], params, cost=float(cost))

            pf_raw = float(m.get("profit_factor", 0.0))
            nt = int(m.get("num_trades", 0))
            dd = float(m.get("maxdd", 0.0))

            sum_pf_raw += pf_raw if math.isfinite(pf_raw) else pf_cap_score
            sum_trades += nt
            if dd < 0.0:
                num_neg_dd += 1

            sum_score += ticker_score(pf_raw, nt)

        score = sum_score / float(len(tickers))

        pen = 0.0
        if penalty:
            frac_neg_dd = float(num_neg_dd) / float(len(tickers))
            pen += 0.25 * frac_neg_dd
            if sum_trades < max(1, len(tickers)):
                pen += 0.10
            score = score - pen

        avg_pf_raw = sum_pf_raw / float(len(tickers))
        avg_trades = sum_trades / float(len(tickers))

        trial.set_user_attr("avg_pf_raw", float(avg_pf_raw))
        trial.set_user_attr("avg_trades_per_ticker", float(avg_trades))
        trial.set_user_attr("num_neg_dd", int(num_neg_dd))
        trial.set_user_attr("penalty_enabled", bool(penalty))
        trial.set_user_attr("penalty_value", float(pen))
        trial.set_user_attr("cost_floor", float(cost) * float(INITIAL_CAPITAL))

        trial.set_user_attr("pf_cap_score_only", float(pf_cap_score))
        trial.set_user_attr("pf_center", float(pf_center))
        trial.set_user_attr("pf_beta", float(pf_beta))
        trial.set_user_attr("min_trades", int(min_trades_i))
        trial.set_user_attr("center", float(center))
        trial.set_user_attr("beta", float(beta))

        return float(score)

    study.optimize(opt_fn, n_trials=n_trials, n_jobs=1, show_progress_bar=True, gc_after_trial=True)

    best_params = dict(study.best_params)
    best_params["best_score"] = float(study.best_value)
    best_params["fill_mode"] = fill_mode
    best_params["timestamp"] = time.time()
    best_params["score"] = (
        "mean_over_tickers( "
        "  [trades<min_trades?0 : sigmoid(pf_beta*(min(PF,pf_cap)-pf_center)) * sigmoid(beta*(trades-center))] "
        ")"
    )
    best_params["penalty"] = bool(penalty)
    best_params["penalty_uses"] = "num_neg_dd (count of tickers with maxdd < 0), normalized"
    best_params["cost"] = float(cost)

    best_params["beta"] = float(beta)
    best_params["center"] = float(center)
    best_params["min_trades"] = int(min_trades_i)

    best_params["pf_cap_score_only"] = float(pf_cap_score)
    best_params["pf_beta"] = float(pf_beta)
    best_params["pf_center"] = float(pf_center)

    with open(OUT_DIR / "best_params_half_rsi_trend.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Run best params across ALL files + outputs
    rows: List[Dict[str, Any]] = []

    for tk in tickers_all:
        params = HalfRSIParams(
            channelLength=int(best_params.get("channelLength", 80)),
            channelMulti=float(best_params.get("channelMulti", 4.85)),

            slow_window=int(best_params.get("slow_window", 32)),
            smooth_len=int(best_params.get("smooth_len", 12)),

            atrPeriod=int(best_params["atrPeriod"]),
            slMultiplier=float(best_params["slMultiplier"]),
            tpMultiplier=float(best_params["tpMultiplier"]),
            trailMultiplier=float(best_params["trailMultiplier"]),

            initial_capital=INITIAL_CAPITAL,
            position_size_pct=POSITION_SIZE_PCT,
            commission_pct=COMMISSION_PCT,
            fill_mode=fill_mode,
        )

        m = backtest_half_rsi_hr(data[tk], params, cost=float(cost))

        pf_raw = float(m.get("profit_factor", 0.0))  # NOT capped
        gp = float(m.get("gross_profit", 0.0))
        gl = float(m.get("gross_loss", 0.0))
        nt = int(m.get("num_trades", 0))
        r = float(m.get("total_return", 0.0))

        tscore = ticker_score(pf_raw, nt)

        rows.append({
            "ticker": tk,
            "profit_factor": pf_raw,
            "num_trades": nt,
            "ticker_score": float(tscore),
            "total_return": r,
            "gross_profit": gp,
            "gross_loss": gl,
            "final_equity": m["final_equity"],
            "sharpe": m["sharpe"],
            "maxdd": m["maxdd"],
        })

        tdf = pd.DataFrame(m["trades"])
        if len(tdf):
            tdf["profit_factor"] = pf_raw
            tdf["gross_profit"] = gp
            tdf["gross_loss"] = gl
            tdf["num_trades_ticker"] = nt
            tdf["cost"] = float(cost)
            tdf["loss_floor"] = float(cost) * float(INITIAL_CAPITAL)
        tdf.to_csv(OUT_DIR / f"{tk}_trades_best.csv", index=False)

        if HAS_PLT:
            plt.figure(figsize=(10, 4))
            plt.plot(m["equity_curve"])
            plt.title(
                f"{tk} Equity (Best) PF={pf_raw:.3f} trades={nt} "
                f"slowW={params.slow_window} sm={params.smooth_len} "
                f"chLen={params.channelLength} chMul={params.channelMulti:.3f} "
                f"atr={params.atrPeriod} sl={params.slMultiplier:.3f} tp={params.tpMultiplier:.3f} trail={params.trailMultiplier:.3f}"
            )
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"equity_{tk}_best.png")
            plt.close()

    summary_df = pd.DataFrame(rows).sort_values(["ticker_score", "profit_factor"], ascending=False)
    summary_df.to_csv(OUT_DIR / "optimization_summary_half_rsi_trend.csv", index=False)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUT_DIR / "optuna_trials_half_rsi_trend.csv", index=False)

    avg_pf_all = float(summary_df["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(pf_cap_score).mean()) if len(summary_df) else 0.0
    avg_trades_all = float(summary_df["num_trades"].mean()) if len(summary_df) else 0.0
    overall_mean_ticker_score = float(summary_df["ticker_score"].mean()) if len(summary_df) else 0.0
    num_neg_dd_all = int((summary_df["maxdd"] < 0.0).sum()) if len(summary_df) else 0

    score_all = overall_mean_ticker_score
    if penalty and len(summary_df):
        frac_neg_dd = float(num_neg_dd_all) / float(len(summary_df))
        pen = 0.25 * frac_neg_dd
        if float(summary_df["num_trades"].sum()) < max(1.0, float(len(summary_df))):
            pen += 0.10
        score_all = score_all - pen

    print("\n=== BEST PARAMS ===")
    for k, v in best_params.items():
        if k not in ("timestamp",):
            print(f"{k}: {v}")

    print("\n=== OVERALL (ALL FILES) METRICS (BEST PARAMS) ===")
    print(f"Avg Profit Factor (raw):      {avg_pf_all:.6f}")
    print(f"Avg Trades / Ticker:          {avg_trades_all:.3f}")
    print(f"Mean Ticker Score:            {overall_mean_ticker_score:.6f}")
    print(f"Penalty enabled:              {bool(penalty)}")
    print(f"num_neg_dd (maxdd<0):          {num_neg_dd_all}/{len(summary_df)}")
    print(f"cost:                         {float(cost):.6f}")
    print(f"loss_floor (cost*capital):    {float(cost)*float(INITIAL_CAPITAL):.2f}")
    print(f"Score (overall):              {score_all:.6f}")

    print("\n=== PER-TICKER METRICS (BEST PARAMS) ===")

    summary_df = summary_df.copy()

    def ticker_score_for_row(pf_raw: float, nt: int) -> float:
        if nt < int(min_trades):
            return 0.0
        pf_for_score = min(float(pf_raw), pf_cap_score) if math.isfinite(float(pf_raw)) else pf_cap_score

        # NO pf_norm term (no division by pf_cap)
        pf_w = sigmoid(float(pf_beta) * (pf_for_score - float(pf_center)))
        tr_w = sigmoid(float(beta) * (float(nt) - float(center)))
        return float(pf_w) * float(tr_w)

    summary_df["ticker_score"] = [
        ticker_score_for_row(pf, int(nt))
        for pf, nt in zip(summary_df["profit_factor"].values, summary_df["num_trades"].values)
    ]

    summary_df_print = summary_df.sort_values(
        by=["profit_factor", "total_return", "num_trades", "ticker_score"],
        ascending=[False, False, False, False],
    )

    headers = ["ticker", "profit_factor", "num_trades", "ticker_score", "total_return", "gross_profit", "gross_loss"]
    print("{:<10s} {:>14s} {:>14s} {:>10s} {:>13s} {:>14s} {:>14s}".format(*headers))

    for _, row in summary_df_print[headers].iterrows():
        print("{:<10s} {:>14.4f} {:>14d} {:>10.6f} {:>13.4f} {:>14.2f} {:>14.2f}".format(
            str(row["ticker"]),
            float(row["profit_factor"]),
            int(row["num_trades"]),
            float(row["ticker_score"]),
            float(row["total_return"]),
            float(row["gross_profit"]),
            float(row["gross_loss"]),
        ))

    print("\nSaved outputs to:", str(OUT_DIR))
    print(" - best_params_half_rsi_trend.json")
    print(" - optimization_summary_half_rsi_trend.csv")
    print(" - optuna_trials_half_rsi_trend.csv")
    print(" - {TICKER}_trades_best.csv (+ equity_{TICKER}_best.png if matplotlib installed)")
    return best_params, summary_df


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200, help="Optuna trials")
    ap.add_argument("--files", type=int, default=200, help="Number of files to use per trial (subsample); if >= total -> use all")
    ap.add_argument("--fill", choices=["next_open", "same_close"], default="same_close", help="Execution model")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for file subsampling")
    ap.add_argument("--cost", type=float, default=0.006, help="Minimum gross-loss floor per (non-winning) trade = cost * INITIAL_CAPITAL")

    ap.add_argument("--penalty", action="store_true", help="Enable penalty term (default OFF). Uses num_neg_dd (maxdd<0)")

    ap.add_argument("--min-trades", type=int, default=4, help="Per-ticker hard gate: trades<min_trades => weight=0 (default 4)")
    ap.add_argument("--beta", type=float, default=1.0, help="Trade sigmoid steepness: sigmoid(beta*(trades-center)) (default 1.0)")
    ap.add_argument("--center", type=float, default=3.0, help="Trade sigmoid center trades at ~0.5 weight (default 3.0)")

    ap.add_argument("--pf-cap", type=float, default=30.0, help="PF cap used ONLY inside scoring (default 30)")
    ap.add_argument("--pf-center", type=float, default=10.0, help="PF sigmoid center (default 10)")
    ap.add_argument("--pf-beta", type=float, default=2.0, help="PF sigmoid steepness (default 2.0)")

    args = ap.parse_args()

    run_optuna_optimization(
        n_trials=args.trials,
        n_files=args.files,
        fill_mode=args.fill,
        seed=args.seed,
        cost=float(args.cost),
        penalty=bool(args.penalty),

        min_trades=int(args.min_trades),
        beta=float(args.beta),
        center=float(args.center),

        pf_cap=float(args.pf_cap),
        pf_center=float(args.pf_center),
        pf_beta=float(args.pf_beta),
    )
