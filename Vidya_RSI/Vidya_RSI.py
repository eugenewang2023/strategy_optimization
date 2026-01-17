#!/usr/bin/env python3
"""
Vidya_RSI.py - High Density Aggression Edition
With Ticker-by-Ticker Reporting and Terminal Status Line
"""

import math
import random
import argparse
import sys
import datetime

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import optuna

# =================================================================================
# INDICATORS & ENGINE (Optimized)
# =================================================================================

def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    out = df[[cols["open"], cols["high"], cols["low"], cols["close"]]].copy()
    out.columns = ["open", "high", "low", "close"]
    return out

def heikin_ashi_from_real(o, h, l, c):
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(ha_c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_c)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    return ha_o, np.maximum.reduce([h, ha_o, ha_c]), np.minimum.reduce([l, ha_o, ha_c]), ha_c

def atr_wilder(h, l, c, period):
    tr = np.zeros(len(c))
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr = np.full(len(c), np.nan)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(c)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def vidya_ema(price: np.ndarray, length: int, smoothing: int) -> np.ndarray:
    n = len(price)
    vidya = np.full(n, np.nan)
    L = int(length)
    if n < L: return vidya
    alpha_base = 2.0 / (smoothing + 1)
    diff = np.diff(price, prepend=price[0])
    pos = np.where(diff > 0, diff, 0.0)
    neg = np.where(diff < 0, -diff, 0.0)
    s_pos = pd.Series(pos).rolling(L).sum().to_numpy()
    s_neg = pd.Series(neg).rolling(L).sum().to_numpy()
    vi = np.abs((s_pos - s_neg) / (s_pos + s_neg + 1e-12))
    vidya[L-1] = price[L-1]
    for i in range(L, n):
        k = alpha_base * vi[i]
        vidya[i] = price[i] * k + vidya[i-1] * (1.0 - k)
    return vidya

@dataclass
class TradeStats:
    gp: float = 0.0
    gl: float = 0.0
    trades: int = 0
    tot_ret: float = 0.0
    pf_diag: float = 0.0

def backtest_vidya_engine(df, **p) -> TradeStats:
    df = ensure_ohlc(df)
    o, h, l, c = df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    if n < 100: return TradeStats()
    _, _, _, ha_c = heikin_ashi_from_real(o, h, l, c)
    atr = atr_wilder(h, l, c, 25)
    v_main = vidya_ema(ha_c, p['vidya_len'], p['vidya_smooth'])
    hot_sm = pd.Series((pd.Series(v_main).ewm(span=p['fastPeriod']).mean() - 
                        pd.Series(v_main).ewm(span=p['slowPeriod']).mean()) / (atr + 1e-12)).rolling(5).mean().to_numpy()
    atr_ma = pd.Series(atr).rolling(p['vol_floor_len']).mean().to_numpy()
    vol_ok = atr > (atr_ma * p['vol_floor_mult'])
    equity, gp, gl, trades = 1.0, 0.0, 0.0, 0
    in_pos, entry, bars_in_trade, cooldown_left, has_hit_be = False, 0.0, 0, 0, False
    for i in range(p['vol_floor_len'] + 5, n - 1):
        if cooldown_left > 0: cooldown_left -= 1; continue
        buy_sig = (hot_sm[i-1] <= p['threshold']) and (hot_sm[i] > p['threshold'])
        sell_sig = (hot_sm[i-1] >= -p['threshold']) and (hot_sm[i] < -p['threshold'])
        if not in_pos and buy_sig and vol_ok[i]:
            entry = o[i+1] if p['fill_mode'] == "next_open" else c[i]
            in_pos, trades = True, trades + 1
            equity *= (1.0 - p['commission_per_side']); bars_in_trade, has_hit_be = 0, False; continue
        if in_pos:
            bars_in_trade += 1
            if (c[i]-entry)/(atr[i]+1e-12) > 1.5: has_hit_be = True
            stop = entry if has_hit_be else (entry - atr[i] * p['slMult'])
            tgt = entry + atr[i] * p['tpMult']
            ex_p = 0.0
            if l[i] <= stop: ex_p = stop
            elif h[i] >= tgt: ex_p = tgt
            elif p['time_stop_bars'] > 0 and bars_in_trade >= p['time_stop_bars'] and c[i] <= entry: ex_p = c[i]
            elif sell_sig or i == n - 2: ex_p = c[i]
            if ex_p > 0:
                equity *= (1.0 - p['commission_per_side'])
                pnl = (ex_p - entry) / entry
                equity *= (1.0 + pnl)
                if pnl >= 0: gp += pnl
                else: gl += abs(pnl); cooldown_left = p['cooldown_bars']
                in_pos = False
    eff_gl = max(gl, (p['loss_floor'] * gp) if gp > 0 else p['loss_floor'])
    return TradeStats(gp, gl, trades, equity-1.0, min(gp/eff_gl, p['pf_cap_score_only']) if eff_gl > 0 else 0.0)

# =================================================================================
# SCORING & LOGGING
# =================================================================================

def score_trial(st: TradeStats, args) -> float:
    if st.trades < args.min_trades: return 0.0
    pf_w = 1.0 / (1.0 + math.exp(-args.pf_k * (st.pf_diag - args.pf_baseline)))
    tr_w = 1.0 / (1.0 + math.exp(-args.trades_k * (st.trades - args.trades_baseline)))
    return (args.weight_pf * pf_w + (1.0 - args.weight_pf) * tr_w) ** args.score_power

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--files", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fill", type=str, default="next_open")
    ap.add_argument("--min-trades", type=int, default=5)
    ap.add_argument("--min-tp2sl", type=float, default=1.1)
    ap.add_argument("--threshold-fixed", type=float, default=0.04)
    ap.add_argument("--vol-floor-len", type=int, default=50)
    ap.add_argument("--vol-floor-mult-fixed", type=float, default=0.55)
    ap.add_argument("--trades-baseline", type=float, default=10.0)
    ap.add_argument("--trades-k", type=float, default=0.4)
    ap.add_argument("--pf-baseline", type=float, default=1.02)
    ap.add_argument("--pf-k", type=float, default=1.2)
    ap.add_argument("--weight-pf", type=float, default=0.4)
    ap.add_argument("--score-power", type=float, default=1.1)
    ap.add_argument("--commission_rate_per_side", type=float, default=0.0006)
    ap.add_argument("--loss_floor", type=float, default=0.001)
    ap.add_argument("--pf-cap", type=float, default=5.0, dest="pf_cap_score_only")
    ap.add_argument("--coverage-target", type=float, default=0.85)
    ap.add_argument("--coverage-k", type=float, default=8.0)
    ap.add_argument("--opt-vidya", action="store_true")
    ap.add_argument("--opt-fastslow", action="store_true")
    ap.add_argument("--opt-time-stop", action="store_true")
    args = ap.parse_known_args()[0]

    files = sorted(Path(args.data_dir).glob("*.parquet"))
    selected_files = random.sample(files, min(len(files), args.files))
    
    # Pre-load data to speed up Optuna
    data_list = [(f.stem, pd.read_parquet(f)) for f in selected_files]

    def objective(trial):
        p = {
            "vidya_len": trial.suggest_int("vl", 2, 20) if args.opt_vidya else 14,
            "vidya_smooth": trial.suggest_int("vs", 5, 40) if args.opt_vidya else 14,
            "fastPeriod": trial.suggest_int("fp", 2, 20) if args.opt_fastslow else 10,
            "slowPeriod": trial.suggest_int("sp", 21, 100) if args.opt_fastslow else 40,
            "time_stop_bars": trial.suggest_int("ts", 5, 50) if args.opt_time_stop else 15,
            "slMult": 3.0, "tpMult": 3.0, "cooldown_bars": 1, "commission_per_side": args.commission_rate_per_side,
            "fill_mode": args.fill, "threshold": args.threshold_fixed, "vol_floor_len": args.vol_floor_len,
            "vol_floor_mult": args.vol_floor_mult_fixed, "loss_floor": args.loss_floor, "pf_cap_score_only": args.pf_cap_score_only
        }
        
        if p["slMult"] < (args.min_tp2sl * p["tpMult"]): return 0.0
        
        scores = []
        eligible = 0
        for name, df in data_list:
            st = backtest_vidya_engine(df, **p)
            sc = score_trial(st, args)
            scores.append(sc)
            if st.trades >= args.min_trades: eligible += 1
        
        # Coverage penalty logic
        coverage_ratio = (eligible / len(data_list))
        coverage_penalty = 1.0 / (1.0 + math.exp(-args.coverage_k * (coverage_ratio - args.coverage_target)))
        final = np.mean(scores) * coverage_penalty
        
        # SAFE STATUS LINE: Catch error if no trials are finished
        try:
            current_best = trial.study.best_value
        except ValueError:
            current_best = 0.0

        sys.stdout.write(f"\rTrial: {trial.number} | Best: {current_best:.5f} | Last: {final:.5f}    ")
        sys.stdout.flush()
        return final

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)
    
    # 1. Create Output Directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 2. Prepare the log file path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    log_file = output_dir / f"opt_results_{timestamp}.csv"
    print("\n" + "="*60)
    print(f"BEST SCORE: {study.best_value}")
    print(f"BEST PARAMS: {study.best_params}")
    print("="*60)

# 3. Generate Final Report
    report_data = []
    header = f"{'Ticker':<15} | {'Trades':<8} | {'Return':<10} | {'PF Diag':<10} | {'Score'}"
    print(header)
    print("-" * 60) 

    # Using Best Params for final pass
    best_p = {
        "vidya_len": study.best_params.get('vl', 14), 
        "vidya_smooth": study.best_params.get('vs', 14),
        "fastPeriod": study.best_params.get('fp', 10), 
        "slowPeriod": study.best_params.get('sp', 40),
        "time_stop_bars": study.best_params.get('ts', 15),
        "slMult": 3.0, "tpMult": 3.0, "cooldown_bars": 1, 
        "commission_per_side": args.commission_rate_per_side,
        "fill_mode": args.fill, "threshold": args.threshold_fixed, 
        "vol_floor_len": args.vol_floor_len,
        "vol_floor_mult": args.vol_floor_mult_fixed, 
        "loss_floor": args.loss_floor, 
        "pf_cap_score_only": args.pf_cap_score_only    
        }
    
    for name, df in data_list:
            st = backtest_vidya_engine(df, **best_p)
            sc = score_trial(st, args)
            print(f"{name:<15} | {st.trades:<8} | {st.tot_ret*100:>8.2f}% | {st.pf_diag:>10.2f} | {sc:.4f}")
            
            report_data.append({
                "ticker": name, 
                "trades": st.trades, 
                "return_pct": round(st.tot_ret*100, 2), 
                "pf": round(st.pf_diag, 2), 
                "score": round(sc, 4)
            })

    # 4. Save to CSV and append Best Params metadata
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(log_file, index=False)
    with open(log_file, 'a') as f:
            f.write(f"\n--- OPTIMIZATION METADATA ---\n")
            f.write(f"BEST_SCORE,{study.best_value}\n")
            for k, v in study.best_params.items():
                f.write(f"{k},{v}\n")   
    print(f"\n[DONE] Detailed results archived in: {log_file}")
    
    print(f"{'Ticker':<15} | {'Trades':<8} | {'Return':<10} | {'PF Diag':<10} | {'Score'}")
    print("-" * 60)
    for name, df in data_list:
        st = backtest_vidya_engine(df, **best_p)
        sc = score_trial(st, args)
        print(f"{name:<15} | {st.trades:<8} | {st.tot_ret*100:>8.2f}% | {st.pf_diag:>10.2f} | {sc:.4f}")

if __name__ == "__main__": main()