#!/usr/bin/env python3
"""
bayes_opt_dynamic_SR.py

Goal: Match TradingView Pine strategy:
"Dynamic_SR_a (HA signals, Real risk)"

Rules (match Pine):
- REAL OHLC is the input data (stable regardless of chart type)
- Heikin Ashi is computed from REAL OHLC for SIGNALS (sigClose = haClose)
- Risk management levels use REAL OHLC + REAL ATR (ta.atr = Wilder RMA of TR)
- Trend/SR logic uses:
    support_resistance updated using REAL high/low + REAL ATR
    crossover/crossunder checks use sigClose vs support_resistance
- Long-only (short side commented in Pine)

Important Pine-specific details reproduced:
- `support_resistance` initialization repeats until ATR is non-na (no fallback)
- ta.crossover/ta.crossunder semantics are replicated with correct series indexing:
  in Pine, y[i] is the value at the start of bar i (carried from bar i-1 updates).
  We therefore use a “last / last_last” scheme for SR/TP/SL levels.
- Exits in Pine check ta.crossunder(sigClose, take_profit_low / trail_stop_level)
  (note: Pine uses sigClose here, not riskClose)

Optimization:
- Uses Optuna
- Optimizes: atrPeriod, slMultiplier, tpMultiplier, trailMultiplier
- Objective (Option A): maximize OVERALL Profit Factor across selected files:
    PF_overall = (sum GrossProfit) / (sum GrossLoss)

CLI:
  --trials N         Optuna trials
  --files N          Number of files to use per trial (subsample); if N >= total -> use all
  --fill MODE        next_open or same_close
  --seed SEED        RNG seed for file subsampling
  --no-penalties     Disable trade/DD penalties (score = PF_overall only)
  --pf-cap X         Cap PF_overall at X (default 10) to avoid infinite PF dominating
"""

import os, json, math, time, random
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
POSITION_SIZE_PCT = 0.10     # percent of equity per trade (TV default_qty_value=10)
COMMISSION_PCT = 0.0006      # 0.06% (Pine: commission_value=0.06 with commission_type=strategy.commission.percent)
# ------------------------------------------------


# =========================
# Helpers: Wilder RMA, ATR
# =========================

def rma_wilder(x: np.ndarray, length: int) -> np.ndarray:
    """
    TradingView ta.rma(): Wilder smoothing.
    Seed = SMA(length) at index length-1.
    alpha = 1/length.
    """
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n < length:
        return out

    s = np.sum(x[:length])
    out[length - 1] = s / length
    alpha = 1.0 / float(length)

    for i in range(length, n):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    return tr


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """
    Match Pine ta.atr(length): RMA of true range.
    """
    tr = true_range(high, low, close)
    return rma_wilder(tr, length)


# =========================
# Heikin Ashi from REAL OHLC
# =========================

def heikin_ashi_from_real(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match Pine:
    haClose = (realO+realH+realL+realC)/4
    var haOpen = na
    haOpen := na(haOpen[1]) ? (realO+realC)/2 : (haOpen[1]+haClose[1])/2
    haHigh = max(realHigh, max(haOpen, haClose))
    haLow  = min(realLow,  min(haOpen, haClose))
    """
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
# Pine-like cross helpers
# =========================

def crossover(cur_x: float, prev_x: float, cur_y: float, prev_y: float) -> bool:
    # ta.crossover(x,y): x>y and x[1] <= y[1]
    return (cur_x > cur_y) and (prev_x <= prev_y)

def crossunder(cur_x: float, prev_x: float, cur_y: float, prev_y: float) -> bool:
    # ta.crossunder(x,y): x<y and x[1] >= y[1]
    return (cur_x < cur_y) and (prev_x >= prev_y)


# =========================
# Backtest: Dynamic_SR_a (HA signals, Real risk)
# =========================

@dataclass
class DynamicSRParams:
    ## for siglRisk
    atrPeriod: int = 32
    slMultiplier: float = 1.1470695356016665
    tpMultiplier: float = 0.5287079013335907
    trailMultiplier: float = 4.30682894724224
    
    ## for realRisk
    # atrPeriod: int = 7
    # slMultiplier: float = 3.777544365622622
    # tpMultiplier: float = 0.5136309391531768
    # trailMultiplier: float = 5.997662923939894

    initial_capital: float = INITIAL_CAPITAL
    position_size_pct: float = POSITION_SIZE_PCT
    commission_pct: float = COMMISSION_PCT

    # execution model: TradingView-style (signals on bar i, fill at next bar open)
    fill_mode: str = "next_open"  # "next_open" or "same_close"


def backtest_dynamic_sr_ha_realrisk(df: pd.DataFrame, p: DynamicSRParams) -> Dict[str, Any]:
    """
    Matches Pine script logic:

    - sigClose = HA close computed from REAL OHLC
    - riskHigh/Low/Close = REAL high/low/close
    - riskATR = ta.atr(atrPeriod) on REAL OHLC
    - support_resistance initialized as:
        if na(sr): isUp := false; sr := riskHigh + riskATR * trailMultiplier
      (no fallback if ATR is na; repeats until ATR available)
    - Trend logic:
        if isUp:
            if crossunder(sigClose, sr): sellSignal, isUp=false, sr := riskHigh + riskATR*trailMultiplier
            else sr := max(sr, riskLow - riskATR*trailMultiplier)
        else:
            if crossover(sigClose, sr): buySignal, isUp=true, sr := riskLow - riskATR*trailMultiplier
            else sr := min(sr, riskHigh + riskATR*trailMultiplier)

    - Entry (long only) on buySignal when flat:
        trail_stop_level := riskLow - riskATR*slMultiplier
        take_profit_low  := riskHigh + riskATR*tpMultiplier
    - Exit checks while long:
        if sellSignal -> close
        if crossunder(sigClose, take_profit_low) -> close
        if crossunder(sigClose, trail_stop_level) -> close
        take_profit_low := max(take_profit_low, riskLow - (riskATR*tpMultiplier))
      (Note: Pine uses sigClose for the crossunder checks even though levels are REAL.)
    """
    df = df.copy().reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    o = df["open"].astype(float).to_numpy()
    h = df["high"].astype(float).to_numpy()
    l = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    n = len(df)

    if n < 5:
        return {
            "equity_curve": [p.initial_capital],
            "trades": [],
            "sharpe": 0.0,
            "maxdd": 0.0,
            "num_trades": 0,
            "final_equity": p.initial_capital,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
        }

    # HA (signals side) computed from REAL OHLC
    ha_o, ha_h, ha_l, ha_c = heikin_ashi_from_real(o, h, l, c)

    # REAL ATR (risk side) = Pine ta.atr()
    # risk_atr = atr_wilder(h, l, c, int(p.atrPeriod))
    risk_atr = atr_wilder(ha_h, ha_l, ha_c, int(p.atrPeriod))

    # Signals use HA close
    sigClose = ha_c

    # Risk management uses REAL series
    # riskHigh  = h
    # riskLow   = l
    # riskClose = c
    # Risk management uses HA series
    riskHigh  = ha_h
    riskLow   = ha_l
    riskClose = ha_c

    # Pine: trail_offset = riskATR * tpMultiplier
    trail_offset = risk_atr * float(p.tpMultiplier)

    # --- Pine var state ---
    isUp = True

    # We must model Pine series behavior for var levels:
    # At bar i start, support_resistance equals value after bar i-1 updates.
    # Thus crossover/crossunder at bar i uses:
    #   y[i]   = sr_last     (computed end of i-1)
    #   y[i-1] = sr_last_last(computed end of i-2)
    sr_last = np.nan
    sr_last_last = np.nan

    # Similarly for TP and SL levels (var float)
    tp_last = np.nan
    tp_last_last = np.nan

    sl_last = np.nan
    sl_last_last = np.nan

    # Position state (long-only)
    in_pos = False
    pending_entry = False
    entry_fill_idx = None
    qty = 0.0

    cash = float(p.initial_capital)
    equity_curve = np.zeros(n, dtype=np.float64)
    trades: List[Dict[str, Any]] = []

    # For cross detection on x series (sigClose)
    prev_sigClose = sigClose[0] if n > 0 else np.nan

    for i in range(n):
        # -------------------------
        # Handle pending entry fill
        # -------------------------
        if p.fill_mode == "next_open":
            if pending_entry and entry_fill_idx == i:
                fill_price = o[i]

                pos_value = cash * float(p.position_size_pct)
                qty = pos_value / fill_price if fill_price > 0 else 0.0

                commission = pos_value * float(p.commission_pct)
                cash -= (pos_value + commission)
                in_pos = True
                pending_entry = False

                trades[-1].update({
                    "fill_idx": int(i),
                    "fill_price": float(fill_price),
                    "commission_entry": float(commission),
                    "qty": float(qty),
                })

        # -------------------------
        # Initialize SR like Pine:
        # if na(support_resistance)
        #   isUp := false
        #   support_resistance := riskHigh + riskATR * trailMultiplier
        # (no fallback when ATR is na; it stays na until ATR exists)
        # -------------------------
        if np.isnan(sr_last):
            isUp = False
            sr_last = riskHigh[i] + risk_atr[i] * float(p.trailMultiplier)  # may remain nan

        # -------------------------
        # Compute signals (Pine ta.crossover/ta.crossunder)
        # Using correct series indexing for SR:
        #   y[i]   -> sr_last
        #   y[i-1] -> sr_last_last
        # -------------------------
        buySignal = False
        sellSignal = False

        if i > 0 and (not np.isnan(sr_last)) and (not np.isnan(sr_last_last)) \
           and (not np.isnan(sigClose[i])) and (not np.isnan(prev_sigClose)):
            if isUp:
                if crossunder(sigClose[i], prev_sigClose, sr_last, sr_last_last):
                    sellSignal = True
                    isUp = False
                    sr_new = riskHigh[i] + risk_atr[i] * float(p.trailMultiplier)
                else:
                    sr_new = max(sr_last, riskLow[i] - risk_atr[i] * float(p.trailMultiplier))
            else:
                if crossover(sigClose[i], prev_sigClose, sr_last, sr_last_last):
                    buySignal = True
                    isUp = True
                    sr_new = riskLow[i] - risk_atr[i] * float(p.trailMultiplier)
                else:
                    sr_new = min(sr_last, riskHigh[i] + risk_atr[i] * float(p.trailMultiplier))
        else:
            sr_new = sr_last  # no update if insufficient history / nans

        # -------------------------
        # Entry logic (long-only)
        # if buySignal and flat:
        #   trail_stop_level := riskLow - riskATR*slMultiplier
        #   take_profit_low  := riskHigh + riskATR*tpMultiplier
        # -------------------------
        if buySignal and (not in_pos) and (not pending_entry):
            sl_last = riskLow[i] - risk_atr[i] * float(p.slMultiplier)
            tp_last = riskHigh[i] + risk_atr[i] * float(p.tpMultiplier)

            trades.append({
                "signal_idx": int(i),
                "side": "LONG",
                "signal_price": float(riskClose[i]),
            })

            if p.fill_mode == "same_close":
                fill_price = riskClose[i]
                pos_value = cash * float(p.position_size_pct)
                qty = pos_value / fill_price if fill_price > 0 else 0.0

                commission = pos_value * float(p.commission_pct)
                cash -= (pos_value + commission)
                in_pos = True

                trades[-1].update({
                    "fill_idx": int(i),
                    "fill_price": float(fill_price),
                    "commission_entry": float(commission),
                    "qty": float(qty),
                })
            else:
                if i + 1 < n:
                    pending_entry = True
                    entry_fill_idx = i + 1
                else:
                    trades[-1].update({"ignored": True})

        # -------------------------
        # Exit logic (match Pine):
        # while long:
        #   if sellSignal -> close
        #   if crossunder(sigClose, tp_last) -> close
        #   if crossunder(sigClose, sl_last) -> close
        #
        # NOTE: Pine uses sigClose for these crosses
        # and uses series y indexing (y[i]=level at bar start).
        # So we must use:
        #   y[i]   -> tp_last / sl_last
        #   y[i-1] -> tp_last_last / sl_last_last
        # -------------------------
        if in_pos:
            exit_reason = None
            do_exit = False

            if sellSignal:
                exit_reason = "sellSignal(HA)"
                do_exit = True
            else:
                if i > 0 and (not np.isnan(tp_last)) and (not np.isnan(tp_last_last)) \
                   and (not np.isnan(sigClose[i])) and (not np.isnan(prev_sigClose)):
                    if crossunder(sigClose[i], prev_sigClose, tp_last, tp_last_last):
                        exit_reason = "trailingTP(RealLevel_HAcross)"
                        do_exit = True

                if (not do_exit) and i > 0 and (not np.isnan(sl_last)) and (not np.isnan(sl_last_last)) \
                   and (not np.isnan(sigClose[i])) and (not np.isnan(prev_sigClose)):
                    if crossunder(sigClose[i], prev_sigClose, sl_last, sl_last_last):
                        exit_reason = "hardSL(RealLevel_HAcross)"
                        do_exit = True

            if do_exit:
                # Fill exit
                if p.fill_mode == "next_open":
                    if i + 1 < n:
                        fill_price = o[i + 1]
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

                # Pine cleanup when flat (later in script):
                tp_last = np.nan
                sl_last = np.nan

            else:
                # Trail TP using REAL low:
                # take_profit_low := max(take_profit_low, riskLow - trail_offset)
                if (not np.isnan(tp_last)) and (not np.isnan(trail_offset[i])):
                    tp_last = max(tp_last, riskLow[i] - trail_offset[i])

        # -------------------------
        # Mark-to-market equity on REAL close (like typical backtests)
        # -------------------------
        mtm = qty * riskClose[i] if in_pos else 0.0
        equity_curve[i] = cash + mtm

        # -------------------------
        # Shift series “history” for next bar:
        # After bar i finishes, Pine y[i] becomes the value used as y at bar i+1 start.
        # So:
        #   sr_last_last <- sr_last
        #   sr_last      <- sr_new
        # and same for tp/sl.
        # -------------------------
        sr_last_last = sr_last
        sr_last = sr_new

        tp_last_last = tp_last
        sl_last_last = sl_last

        prev_sigClose = sigClose[i]

    # If still open at end, close at last REAL close
    if in_pos and qty > 0:
        fill_price = riskClose[-1]
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

    # Metrics: sharpe, maxdd
    eq = equity_curve.copy()
    if len(eq) < 2:
        sharpe = 0.0
        maxdd = 0.0
    else:
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        sharpe = float((np.mean(rets) / (np.std(rets) + 1e-12)) * math.sqrt(252.0)) if len(rets) > 1 else 0.0
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.maximum(peak, 1e-12)
        maxdd = float(np.min(dd))  # negative

    # Completed trades
    completed = [t for t in trades if ("exit_price" in t and "fill_price" in t and not t.get("ignored", False))]
    num_trades = len(completed)

    # Gross profit/loss (net of commissions)
    gross_profit = 0.0
    gross_loss = 0.0
    for t in completed:
        if "qty" not in t:
            continue
        q = float(t["qty"])
        entry = float(t["fill_price"])
        exitp = float(t["exit_price"])
        entry_comm = float(t.get("commission_entry", 0.0))
        exit_comm  = float(t.get("commission_exit", 0.0))
        pnl = (exitp - entry) * q - entry_comm - exit_comm
        if pnl > 0:
            gross_profit += pnl
        elif pnl < 0:
            gross_loss += -pnl

    # Profit factor
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = 10.0 if gross_profit > 0 else 0.0

    return {
        "final_equity": float(eq[-1]),
        "total_return": float(eq[-1] / p.initial_capital - 1.0),
        "sharpe": float(sharpe),
        "maxdd": float(maxdd),
        "num_trades": int(num_trades),
        "trades": trades,
        "equity_curve": eq.tolist(),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "profit_factor": float(profit_factor),
    }


# =========================
# Optimization (Optuna)
# =========================

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_files() -> Dict[str, pd.DataFrame]:
    """
    Loads all .csv and .parquet in DATA_DIR.
    Keys are stem uppercased (e.g., NVDA from NVDA.parquet).
    """
    files = list(DATA_DIR.glob("*.parquet")) + list(DATA_DIR.glob("*.csv"))
    files.sort()
    data: Dict[str, pd.DataFrame] = {}
    for fp in files:
        try:
            if fp.suffix.lower() == ".parquet":
                df = pd.read_parquet(fp)
            else:
                df = pd.read_csv(fp)
            data[fp.stem.upper()] = df
        except Exception as e:
            print(f"Failed to load {fp.name}: {e}")
    return data


def run_optuna_optimization(
    n_trials: int = 200,
    n_files: int = 200,
    fill_mode: str = "next_open",
    study_name: Optional[str] = None,
    seed: int = 42,
    use_penalties: bool = True,
    pf_cap: float = 10.0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    ensure_dirs()
    data = load_files()
    if not data:
        raise SystemExit(f"No .csv/.parquet found in {DATA_DIR}. Put files there and re-run.")

    tickers_all = list(data.keys())
    tickers_all.sort()

    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
    except Exception as e:
        raise SystemExit("Optuna not installed. pip install optuna") from e

    if study_name is None:
        study_name = f"dynamic_sr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

    def opt_fn(trial):
        # Keep optimization scope exactly as your header described
        atrP = trial.suggest_int("atrPeriod", 5, 60)
        slM  = trial.suggest_float("slMultiplier", 0.5, 10.0)
        tpM  = trial.suggest_float("tpMultiplier", 0.5, 12.0)
        trM  = trial.suggest_float("trailMultiplier", 0.5, 10.0)

        tickers = choose_tickers_for_trial()

        total_gp = 0.0
        total_gl = 0.0
        agg_trades = 0
        agg_dd = 0.0

        for tk in tickers:
            params = DynamicSRParams(
                atrPeriod=int(atrP),
                slMultiplier=float(slM),
                tpMultiplier=float(tpM),
                trailMultiplier=float(trM),
                initial_capital=INITIAL_CAPITAL,
                position_size_pct=POSITION_SIZE_PCT,
                commission_pct=COMMISSION_PCT,
                fill_mode=fill_mode,
            )
            m = backtest_dynamic_sr_ha_realrisk(data[tk], params)

            total_gp += float(m.get("gross_profit", 0.0))
            total_gl += float(m.get("gross_loss", 0.0))
            agg_trades += int(m.get("num_trades", 0))
            agg_dd += abs(float(m.get("maxdd", 0.0)))

        # OVERALL PF (Option A): pool profits & losses
        if total_gl > 0:
            pf_overall = total_gp / total_gl
        else:
            pf_overall = 10.0 if total_gp > 0 else 0.0

        pf_overall = float(min(pf_overall, float(pf_cap)))

        score = pf_overall
        penalty = 0.0
        if use_penalties:
            if agg_trades < 3:
                penalty += 3.0
            elif agg_trades < 10:
                penalty += 1.0
            score = pf_overall - (agg_dd * 0.25) - penalty

        trial.set_user_attr("pf_overall", float(pf_overall))
        trial.set_user_attr("total_gross_profit", float(total_gp))
        trial.set_user_attr("total_gross_loss", float(total_gl))
        trial.set_user_attr("total_trades", int(agg_trades))
        trial.set_user_attr("sum_abs_maxdd", float(agg_dd))
        trial.set_user_attr("penalty", float(penalty))
        trial.set_user_attr("n_files_used", int(len(tickers)))

        return float(score)

    study.optimize(opt_fn, n_trials=n_trials, n_jobs=1, show_progress_bar=True, gc_after_trial=True)

    best_params = dict(study.best_params)
    best_params["best_score"] = float(study.best_value)
    best_params["fill_mode"] = fill_mode
    best_params["timestamp"] = time.time()
    best_params["score_is_pf_overall_optionA"] = True
    best_params["pf_cap"] = float(pf_cap)
    best_params["use_penalties"] = bool(use_penalties)

    with open(OUT_DIR / "best_params_dynamic_sr.json", "w") as f:
        json.dump(best_params, f, indent=2)

    rows = []
    for tk in tickers_all:
        params = DynamicSRParams(
            atrPeriod=int(best_params["atrPeriod"]),
            slMultiplier=float(best_params["slMultiplier"]),
            tpMultiplier=float(best_params["tpMultiplier"]),
            trailMultiplier=float(best_params["trailMultiplier"]),
            initial_capital=INITIAL_CAPITAL,
            position_size_pct=POSITION_SIZE_PCT,
            commission_pct=COMMISSION_PCT,
            fill_mode=fill_mode,
        )
        m = backtest_dynamic_sr_ha_realrisk(data[tk], params)
        rows.append({
            "ticker": tk,
            "final_equity": m["final_equity"],
            "total_return": m["total_return"],
            "sharpe": m["sharpe"],
            "maxdd": m["maxdd"],
            "num_trades": m["num_trades"],
            "gross_profit": m["gross_profit"],
            "gross_loss": m["gross_loss"],
            "profit_factor": m["profit_factor"],
        })

        pd.DataFrame(m["trades"]).to_csv(OUT_DIR / f"{tk}_trades_best.csv", index=False)

        if HAS_PLT:
            plt.figure(figsize=(10, 4))
            plt.plot(m["equity_curve"])
            plt.title(
                f"{tk} Equity (Best)  atr={params.atrPeriod} sl={params.slMultiplier:.3f} "
                f"tp={params.tpMultiplier:.3f} trail={params.trailMultiplier:.3f}"
            )
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"equity_{tk}_best.png")
            plt.close()

    summary_df = pd.DataFrame(rows).sort_values("profit_factor", ascending=False)
    summary_df.to_csv(OUT_DIR / "optimization_summary_dynamic_sr.csv", index=False)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUT_DIR / "optuna_trials_dynamic_sr.csv", index=False)

    if HAS_PLT:
        plt.figure(figsize=(10, 6))
        for tk in tickers_all:
            params = DynamicSRParams(
                atrPeriod=int(best_params["atrPeriod"]),
                slMultiplier=float(best_params["slMultiplier"]),
                tpMultiplier=float(best_params["tpMultiplier"]),
                trailMultiplier=float(best_params["trailMultiplier"]),
                initial_capital=INITIAL_CAPITAL,
                position_size_pct=POSITION_SIZE_PCT,
                commission_pct=COMMISSION_PCT,
                fill_mode=fill_mode,
            )
            m = backtest_dynamic_sr_ha_realrisk(data[tk], params)
            plt.plot(m["equity_curve"], label=tk)
        plt.legend()
        plt.title("Equity Curves - Best Params (Dynamic_SR_a HA signals / Real risk)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "equity_all_best_dynamic_sr.png")
        plt.close()

    total_gp = 0.0
    total_gl = 0.0
    total_trades = 0
    for tk in tickers_all:
        params = DynamicSRParams(
            atrPeriod=int(best_params["atrPeriod"]),
            slMultiplier=float(best_params["slMultiplier"]),
            tpMultiplier=float(best_params["tpMultiplier"]),
            trailMultiplier=float(best_params["trailMultiplier"]),
            initial_capital=INITIAL_CAPITAL,
            position_size_pct=POSITION_SIZE_PCT,
            commission_pct=COMMISSION_PCT,
            fill_mode=fill_mode,
        )
        m = backtest_dynamic_sr_ha_realrisk(data[tk], params)
        total_gp += float(m.get("gross_profit", 0.0))
        total_gl += float(m.get("gross_loss", 0.0))
        total_trades += int(m.get("num_trades", 0))

    pf_all = (total_gp / total_gl) if total_gl > 0 else (10.0 if total_gp > 0 else 0.0)
    pf_all = float(min(pf_all, float(pf_cap)))

    print("\n=== BEST PARAMS ===")
    for k, v in best_params.items():
        if k not in ("timestamp",):
            print(f"{k}: {v}")

    print("\n=== OVERALL (ALL FILES) PROFIT FACTOR ===")
    print(f"Gross Profit: {total_gp:,.2f}")
    print(f"Gross Loss:   {total_gl:,.2f}")
    print(f"PF Overall:   {pf_all:.4f}")
    print(f"Total Trades: {total_trades}")

    print("\nSaved outputs to:", str(OUT_DIR))
    print(summary_df.to_string(index=False))
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
    ap.add_argument("--no-penalties", action="store_true", help="Disable trade/DD penalties (score = PF_overall only)")
    ap.add_argument("--pf-cap", type=float, default=10.0, help="Cap PF_overall at this value (default 10.0)")
    args = ap.parse_args()

    run_optuna_optimization(
        n_trials=args.trials,
        n_files=args.files,
        fill_mode=args.fill,
        seed=args.seed,
        use_penalties=(not args.no_penalties),
        pf_cap=args.pf_cap,
    )
