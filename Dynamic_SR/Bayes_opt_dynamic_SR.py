#!/usr/bin/env python3
"""
bayes_opt_dynamic_SR.py

Goal: Match TradingView Pine strategy:
"Dynamic_SR (HA signals, Real risk)"

Rules:
- Signals/trend logic use Heikin Ashi computed from REAL OHLC (so chart type doesn't matter)
- SL/TP/Trailing use REAL OHLC + REAL ATR (Pine ta.atr = Wilder RMA of TR)
- Long-only (like your Pine code where short entries are commented)

Optimization:
- Uses Optuna
- Optimizes: atrPeriod, slMultiplier, tpMultiplier, trailMultiplier
- Objective: maximize Profit Factor (with penalties for too-few trades and large drawdown)

Data:
- Put .csv or .parquet files in ./data/
- Required columns (case-insensitive): open, high, low, close (volume optional)
- Outputs go to ./output/
"""

import os
import json
import math
import time
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
POSITION_SIZE_PCT = 0.10     # percent of equity per trade
COMMISSION_PCT = 0.0006      # 0.06%
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
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match Pine:
    haClose = (realO+realH+realL+realC)/4
    haOpen  = na(prev)? (realO+realC)/2 : (haOpen[1]+haClose[1])/2
    haHigh  = max(realHigh, max(haOpen, haClose))
    haLow   = min(realLow,  min(haOpen, haClose))
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

def crossover(cur: float, prev: float, lvl_cur: float, lvl_prev: float) -> bool:
    # ta.crossover(x,y): x>y and x[1] <= y[1]
    return (cur > lvl_cur) and (prev <= lvl_prev)

def crossunder(cur: float, prev: float, lvl_cur: float, lvl_prev: float) -> bool:
    # ta.crossunder(x,y): x<y and x[1] >= y[1]
    return (cur < lvl_cur) and (prev >= lvl_prev)


# =========================
# Profit Factor from trades
# =========================

def compute_profit_factor_from_trades(trades: List[Dict[str, Any]]) -> Tuple[float, float, float, int]:
    """
    Profit Factor = gross_profit / gross_loss
    Uses net PnL per trade: (exit-entry)*qty - (entry_comm + exit_comm)

    Returns: (profit_factor, gross_profit, gross_loss, num_closed_trades)
    """
    gross_profit = 0.0
    gross_loss = 0.0
    closed = 0

    for t in trades:
        if t.get("ignored", False):
            continue
        if "fill_price" not in t or "exit_price" not in t:
            continue

        closed += 1
        entry = float(t["fill_price"])
        exit_ = float(t["exit_price"])
        qty = float(t.get("qty", 0.0))

        comm_e = float(t.get("commission_entry", 0.0))
        comm_x = float(t.get("commission_exit", 0.0))

        pnl = (exit_ - entry) * qty - (comm_e + comm_x)

        if pnl >= 0:
            gross_profit += pnl
        else:
            gross_loss += -pnl

    if closed == 0:
        return 0.0, 0.0, 0.0, 0

    if gross_loss <= 1e-12:
        return float("inf"), gross_profit, gross_loss, closed

    return gross_profit / gross_loss, gross_profit, gross_loss, closed


# =========================
# Backtest: Dynamic_SR (HA signals, Real risk)
# =========================

@dataclass
class DynamicSRParams:
    atrPeriod: int = 25
    slMultiplier: float = 3.0
    tpMultiplier: float = 2.69
    trailMultiplier: float = 4.0

    initial_capital: float = INITIAL_CAPITAL
    position_size_pct: float = POSITION_SIZE_PCT
    commission_pct: float = COMMISSION_PCT

    # execution model: TradingView-style (signals on bar i, fill at next bar open)
    fill_mode: str = "next_open"  # "next_open" or "same_close"


def backtest_dynamic_sr_ha_realrisk(df: pd.DataFrame, p: DynamicSRParams) -> Dict[str, Any]:
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
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "closed_trades": 0
        }

    # REAL ATR (risk side)
    risk_atr = atr_wilder(h, l, c, int(p.atrPeriod))

    # HA (signals side) computed from REAL OHLC
    ha_o, ha_h, ha_l, ha_c = heikin_ashi_from_real(o, h, l, c)

    # Signals use HA close; SR levels use REAL high/low + REAL ATR
    sigClose = ha_c
    riskHigh = h
    riskLow  = l
    riskClose= c

    # Pine: trail_offset = riskATR * tpMultiplier
    trail_offset = risk_atr * float(p.tpMultiplier)

    # State vars (Pine var)
    isUp = True
    sr = np.nan

    # Position state (long-only)
    in_pos = False
    pending_entry = False
    entry_fill_idx = None
    entry_price = 0.0
    qty = 0.0

    trail_stop_level = np.nan
    take_profit_low  = np.nan

    cash = float(p.initial_capital)
    equity_curve = np.zeros(n, dtype=np.float64)
    trades: List[Dict[str, Any]] = []

    # For cross detection
    prev_sigClose = sigClose[0]
    prev_sr = np.nan
    prev_riskClose = riskClose[0]
    prev_takeprofit = np.nan
    prev_stop = np.nan

    # Initialize SR (matches Pine)
    # if na(support_resistance): isUp := false; support_resistance := riskHigh + riskATR * trailMultiplier
    isUp = False
    if not np.isnan(risk_atr[0]):
        sr = riskHigh[0] + risk_atr[0] * float(p.trailMultiplier)
    else:
        sr = riskHigh[0]

    for i in range(n):
        # Handle pending fill at bar open (next_open mode)
        if p.fill_mode == "next_open":
            if pending_entry and entry_fill_idx == i:
                fill_price = o[i]
                equity_now = cash + (qty * riskClose[i] if in_pos else 0.0)
                pos_value = equity_now * float(p.position_size_pct)
                qty = pos_value / fill_price if fill_price > 0 else 0.0
                commission = pos_value * float(p.commission_pct)
                cash -= (pos_value + commission)
                in_pos = True
                pending_entry = False

                trades[-1].update({
                    "fill_idx": i,
                    "fill_price": float(fill_price),
                    "commission_entry": float(commission),
                    "qty": float(qty),
                })
                entry_price = fill_price

        # Compute signals (Pine uses ta.crossover/ta.crossunder with sr)
        buySignal = False
        sellSignal = False

        if i > 0 and not np.isnan(sr) and not np.isnan(prev_sr) and not np.isnan(sigClose[i]) and not np.isnan(prev_sigClose):
            if isUp:
                if crossunder(sigClose[i], prev_sigClose, sr, prev_sr):
                    sellSignal = True
                    isUp = False
                    if not np.isnan(risk_atr[i]):
                        sr = riskHigh[i] + risk_atr[i] * float(p.trailMultiplier)
                    else:
                        sr = riskHigh[i]
                else:
                    if not np.isnan(risk_atr[i]):
                        sr = max(sr, riskLow[i] - risk_atr[i] * float(p.trailMultiplier))
            else:
                if crossover(sigClose[i], prev_sigClose, sr, prev_sr):
                    buySignal = True
                    isUp = True
                    if not np.isnan(risk_atr[i]):
                        sr = riskLow[i] - risk_atr[i] * float(p.trailMultiplier)
                    else:
                        sr = riskLow[i]
                else:
                    if not np.isnan(risk_atr[i]):
                        sr = min(sr, riskHigh[i] + risk_atr[i] * float(p.trailMultiplier))

        # === Entry logic ===
        if buySignal and (not in_pos) and (not pending_entry):
            # levels based on signal bar
            if not np.isnan(risk_atr[i]):
                trail_stop_level = riskLow[i] - risk_atr[i] * float(p.slMultiplier)
                take_profit_low  = riskHigh[i] + risk_atr[i] * float(p.tpMultiplier)
            else:
                trail_stop_level = riskLow[i]
                take_profit_low  = riskHigh[i]

            trades.append({"signal_idx": i, "side": "LONG", "signal_price": float(riskClose[i])})

            if p.fill_mode == "same_close":
                fill_price = riskClose[i]
                pos_value = cash * float(p.position_size_pct)
                qty = pos_value / fill_price if fill_price > 0 else 0.0
                commission = pos_value * float(p.commission_pct)
                cash -= (pos_value + commission)
                in_pos = True
                entry_price = fill_price
                trades[-1].update({
                    "fill_idx": i,
                    "fill_price": float(fill_price),
                    "commission_entry": float(commission),
                    "qty": float(qty),
                })
            else:
                # next_open fill
                if i + 1 < n:
                    pending_entry = True
                    entry_fill_idx = i + 1
                else:
                    trades[-1].update({"ignored": True})

        # === Exit logic ===
        if in_pos:
            exit_reason = None
            do_exit = False

            if sellSignal:
                exit_reason = "sellSignal(HA)"
                do_exit = True
            else:
                if i > 0 and not np.isnan(take_profit_low) and not np.isnan(prev_takeprofit):
                    if crossunder(riskClose[i], prev_riskClose, take_profit_low, prev_takeprofit):
                        exit_reason = "trailingTP(Real)"
                        do_exit = True
                if (not do_exit) and i > 0 and not np.isnan(trail_stop_level) and not np.isnan(prev_stop):
                    if crossunder(riskClose[i], prev_riskClose, trail_stop_level, prev_stop):
                        exit_reason = "hardSL(Real)"
                        do_exit = True

            if do_exit:
                # fill exit
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
                entry_price = 0.0
                trail_stop_level = np.nan
                take_profit_low = np.nan

            else:
                # trail TP using REAL low
                if not np.isnan(take_profit_low) and not np.isnan(trail_offset[i]):
                    take_profit_low = max(take_profit_low, riskLow[i] - trail_offset[i])

        # Mark-to-market equity on REAL close
        mtm = qty * riskClose[i] if in_pos else 0.0
        equity_curve[i] = cash + mtm

        # update prevs
        prev_sigClose = sigClose[i]
        prev_sr = sr
        prev_riskClose = riskClose[i]
        prev_takeprofit = take_profit_low
        prev_stop = trail_stop_level

    # If still open at end, close at last close
    if in_pos and qty > 0:
        fill_price = riskClose[-1]
        exit_value = qty * fill_price
        commission = exit_value * float(p.commission_pct)
        cash += (exit_value - commission)
        trades[-1].update({
            "exit_idx": n - 1,
            "exit_price": float(fill_price),
            "commission_exit": float(commission),
            "reason": "endClose"
        })
        qty = 0.0
        in_pos = False
        equity_curve[-1] = cash

    # Metrics: Sharpe + MaxDD
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

    # Count completed trades
    completed = [t for t in trades if ("exit_price" in t and "fill_price" in t and not t.get("ignored", False))]
    num_trades = len(completed)

    # Profit Factor (net PnL including commissions)
    profit_factor, gross_profit, gross_loss, closed_trades = compute_profit_factor_from_trades(trades)

    return {
        "final_equity": float(eq[-1]),
        "total_return": float(eq[-1] / p.initial_capital - 1.0),
        "sharpe": sharpe,
        "maxdd": maxdd,
        "num_trades": int(num_trades),
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else float("inf"),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "closed_trades": int(closed_trades),
        "trades": trades,
        "equity_curve": eq.tolist(),
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
    data = {}
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


def objective_score(metrics: Dict[str, Any]) -> float:
    """
    Maximize Profit Factor with penalties:
    - too few trades
    - large drawdown
    PF can be inf if no losses; capped for stability.
    """
    pf = float(metrics.get("profit_factor", 0.0))
    num_trades = int(metrics.get("num_trades", 0))
    maxdd_abs = abs(float(metrics.get("maxdd", 0.0)))

    # Cap infinite/huge PF (avoid Optuna chasing 1-2 trade luck)
    if not np.isfinite(pf):
        pf = 10.0
    pf = min(pf, 10.0)

    penalty = 0.0
    if num_trades < 3:
        penalty += 3.0
    elif num_trades < 10:
        penalty += 1.0

    # drawdown penalty
    score = pf - (maxdd_abs * 1.0) - penalty
    return float(score)


def run_optuna_optimization(
    n_trials: int = 200,
    n_files: int = 3,
    fill_mode: str = "next_open",
    study_name: Optional[str] = None,
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

    sampler = TPESampler(seed=42, n_startup_trials=30, multivariate=True)
    pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    def opt_fn(trial):
        atrP = trial.suggest_int("atrPeriod", 5, 40)
        slM  = trial.suggest_float("slMultiplier", 0.5, 5.0)
        tpM  = trial.suggest_float("tpMultiplier", 0.5, 10.0)
        trM  = trial.suggest_float("trailMultiplier", 0.5, 6.0)

        # --- handle --files like Bayes_opt_adapt_RSI.py ---
        # use only first n_files (sorted) for this trial (stable)
        use_n = min(max(1, int(n_files)), len(tickers_all))
        tickers = tickers_all[:use_n]

        scores = []
        agg_trades = 0
        agg_dd = 0.0
        agg_pf = 0.0
        pf_count = 0

        for tk in tickers:
            params = DynamicSRParams(
                atrPeriod=atrP,
                slMultiplier=slM,
                tpMultiplier=tpM,
                trailMultiplier=trM,
                initial_capital=INITIAL_CAPITAL,
                position_size_pct=POSITION_SIZE_PCT,
                commission_pct=COMMISSION_PCT,
                fill_mode=fill_mode,
            )
            m = backtest_dynamic_sr_ha_realrisk(data[tk], params)

            scores.append(objective_score(m))
            agg_trades += int(m.get("num_trades", 0))
            agg_dd += abs(float(m.get("maxdd", 0.0)))

            pf = float(m.get("profit_factor", 0.0))
            if np.isfinite(pf):
                agg_pf += pf
                pf_count += 1

        # Save attrs for inspection
        trial.set_user_attr("avg_profit_factor", agg_pf / max(1, pf_count))
        trial.set_user_attr("avg_abs_maxdd", agg_dd / max(1, len(tickers)))
        trial.set_user_attr("total_trades", agg_trades)

        return float(np.mean(scores)) if scores else -1e9

    study.optimize(opt_fn, n_trials=n_trials, n_jobs=1, show_progress_bar=True, gc_after_trial=True)

    best_params = dict(study.best_params)
    best_params["best_score"] = float(study.best_value)
    best_params["fill_mode"] = fill_mode
    best_params["files_used_per_trial"] = int(min(max(1, int(n_files)), len(tickers_all)))
    best_params["timestamp"] = time.time()

    # Save best params
    with open(OUT_DIR / "best_params_dynamic_sr.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Evaluate best on ALL tickers and save per-ticker artifacts
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
            "profit_factor": m["profit_factor"],
            "gross_profit": m["gross_profit"],
            "gross_loss": m["gross_loss"],
            "sharpe": m["sharpe"],
            "maxdd": m["maxdd"],
            "num_trades": m["num_trades"],
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

    # Save trials table
    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUT_DIR / "optuna_trials_dynamic_sr.csv", index=False)

    # Combined equity plot
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
        plt.title("Equity Curves - Best Params (Dynamic_SR HA signals / Real risk)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "equity_all_best_dynamic_sr.png")
        plt.close()

    print("\n=== BEST PARAMS ===")
    for k, v in best_params.items():
        if k not in ("timestamp",):
            print(f"{k}: {v}")

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
    ap.add_argument("--files", type=int, default=3, help="Number of files to use per trial (like Bayes_opt_adapt_RSI.py)")
    ap.add_argument("--fill", choices=["next_open", "same_close"], default="same_close", help="Execution model")
    args = ap.parse_args()

    run_optuna_optimization(
        n_trials=args.trials,
        n_files=args.files,
        fill_mode=args.fill
    )
