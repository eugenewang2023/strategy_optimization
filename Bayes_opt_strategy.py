#!/usr/bin/env python3
"""
bayes_opt_strategy.py

Bayesian/random optimization runner for the user's "Dynamic S/R" strategy.

Place CSVs in ./data/ (NVDA.csv, TSLA.csv, ACWI.csv). Outputs saved to ./output/.
"""
import os, json, time, math, random
from datetime import datetime
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ----------------- USER CONFIG -----------------
DATA_DIR = "data"     # CSVs live here
OUT_DIR  = "output"   # results saved here
INITIAL_CAPITAL = 100000.0
PERCENT_RISK = 0.10   # percent of equity to risk per trade
OBJECTIVE = "sharpe"  # used for comments; bayes objective implemented as -Sharpe + penalties
# ------------------------------------------------

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

# --------- Backtester (fixed: no same-bar exits) ----------
def backtest_df_fixed(df, atrPeriod, slMultiplier, tpMultiplier, trailMultiplier,
                      initial_capital=INITIAL_CAPITAL, percent_risk=PERCENT_RISK):
    df = df.copy().reset_index(drop=True)
    # normalize columns to proper case
    df.columns = [c.capitalize() for c in df.columns]

    for c in ['Open','High','Low','Close','Volume']:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    high = df['High'].astype(float).values
    low  = df['Low'].astype(float).values
    close= df['Close'].astype(float).values
    n = len(df)
    # Wilder-style ATR approximation using simple rolling mean of TR
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.concatenate(([close[0]], close[:-1])))),
                    np.abs(low  - np.concatenate(([close[0]], close[:-1]))))
    atr = pd.Series(tr).rolling(window=max(1,int(atrPeriod)), min_periods=1).mean().values

    # state variables
    isUp = False
    support_resistance = np.nan
    position = 0
    entry_price = 0.0
    trail_stop_level = np.nan
    take_profit_low = np.nan
    cash = float(initial_capital)
    equity = float(initial_capital)
    shares = 0.0
    eq_curve = []
    trades = []
    entry_idx = None

    for i in range(n):
        dist = max(high[i] - low[i], atr[i] if not math.isnan(atr[i]) else (high[i]-low[i]))
        trail_offset = dist * tpMultiplier

        if np.isnan(support_resistance):
            isUp = False
            support_resistance = high[i] + dist * trailMultiplier

        buySignal = False
        sellSignal = False

        if isUp:
            if close[i] < support_resistance:
                sellSignal = True
                isUp = False
                support_resistance = high[i] + dist * trailMultiplier
            else:
                support_resistance = max(support_resistance, low[i] - dist * trailMultiplier)
        else:
            if close[i] > support_resistance:
                buySignal = True
                isUp = True
                support_resistance = low[i] - dist * trailMultiplier
            else:
                support_resistance = min(support_resistance, high[i] + dist * trailMultiplier)

        # Entry
        if buySignal and position == 0:
            entry_price = close[i]
            position_value = equity * percent_risk
            shares = position_value / entry_price if entry_price > 0 else 0.0
            cash -= shares * entry_price
            position = 1
            trail_stop_level = low[i] - dist * slMultiplier
            take_profit_low = high[i] + dist * tpMultiplier
            entry_idx = i
            trades.append({"entry_idx": i, "entry_price": float(entry_price)})

        # Manage position; skip exits on entry-bar
        if position == 1:
            if i != entry_idx:
                if sellSignal:
                    exit_price = close[i]
                    cash += shares * exit_price
                    trades[-1].update({"exit_idx": i, "exit_price": float(exit_price), "reason": "sellSignal"})
                    shares = 0; position = 0; trail_stop_level = np.nan; take_profit_low = np.nan; entry_idx = None
                elif close[i] < take_profit_low:
                    exit_price = close[i]
                    cash += shares * exit_price
                    trades[-1].update({"exit_idx": i, "exit_price": float(exit_price), "reason": "trailingTP"})
                    shares = 0; position = 0; trail_stop_level = np.nan; take_profit_low = np.nan; entry_idx = None
                elif close[i] < trail_stop_level:
                    exit_price = close[i]
                    cash += shares * exit_price
                    trades[-1].update({"exit_idx": i, "exit_price": float(exit_price), "reason": "hardSL"})
                    shares = 0; position = 0; trail_stop_level = np.nan; take_profit_low = np.nan; entry_idx = None
                else:
                    # adjust trailing take_profit_low as allowed
                    take_profit_low = max(take_profit_low, low[i] - trail_offset)

        mtm = shares * close[i]
        equity = cash + mtm
        eq_curve.append(equity)

    # close at end if still open
    if position == 1 and shares > 0:
        exit_price = close[-1]
        cash += shares * exit_price
        trades[-1].update({"exit_idx": n-1, "exit_price": float(exit_price), "reason": "endClose"})
        shares = 0; position = 0; equity = cash

    eq = np.array(eq_curve) if len(eq_curve)>0 else np.array([initial_capital])
    returns = np.diff(eq) / eq[:-1] if len(eq)>1 else np.array([])
    total_return = (equity / initial_capital) - 1.0
    days = len(df)
    years = days / 252.0 if days>0 else 1.0
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * math.sqrt(252) if len(returns) > 1 else 0.0
    peak = np.maximum.accumulate(eq) if len(eq)>0 else np.array([equity])
    drawdown = (eq - peak) / peak if len(eq)>0 else np.array([0.0])
    maxdd = float(drawdown.min()) if len(drawdown)>0 else 0.0

    wins = 0; losses = 0; profits = []
    for t in trades:
        if "exit_price" in t:
            pnl = (t["exit_price"] - t["entry_price"]) * ((initial_capital * percent_risk) / t["entry_price"])
            profits.append(pnl)
            if pnl > 0: wins += 1
            else: losses += 1
    num_trades = len([p for p in profits])
    win_rate = (wins / num_trades) if num_trades > 0 else 0.0
    avg_win = float(np.mean([p for p in profits if p>0])) if any([p>0 for p in profits]) else 0.0
    avg_loss = float(np.mean([p for p in profits if p<=0])) if any([p<=0 for p in profits]) else 0.0

    return {
        "total_return": float(total_return),
        "equity": float(equity),
        "cagr": float((equity / initial_capital) ** (1.0 / max(years,1e-9)) - 1.0) if years > 0 else 0.0,
        "sharpe": float(sharpe),
        "maxdd": float(maxdd),
        "num_trades": int(num_trades),
        "win_rate": float(win_rate),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "trades": trades,
        "equity_curve": eq.tolist()
    }

# ------- Objective wrapper (minimize) -------
def objective(params, df):
    atrP, slM, tpM, trailM = params
    atrP = int(round(atrP))
    slM = float(slM); tpM = float(tpM); trailM = float(trailM)
    metrics = backtest_df_fixed(df, atrPeriod=atrP, slMultiplier=slM, tpMultiplier=tpM, trailMultiplier=trailM)
    sharpe = metrics['sharpe']
    num_trades = metrics['num_trades']
    maxdd = abs(metrics['maxdd'])
    penalty = 0.0
    if num_trades < 3:
        penalty += 1.0
    # objective: minimize negative sharpe + penalties + drawdown penalty (smaller is better)
    obj = -sharpe + penalty + maxdd * 0.5
    return float(obj), metrics

# ------- Main optimization routine -------
def run_optimization():
    ensure_dirs()
    # load CSVs
    csvs = {}
    for fn in os.listdir(DATA_DIR):
        if fn.lower().endswith(".csv"):
            path = os.path.join(DATA_DIR, fn)
            try:
                df = pd.read_csv(path)
                csvs[os.path.splitext(fn)[0].upper()] = df
            except Exception as e:
                print(f"Failed to load {fn}: {e}")
    if len(csvs) == 0:
        raise SystemExit(f"No CSVs found in {DATA_DIR}. Place NVDA.csv etc. there and re-run.")

    # bounds
    bounds = {
        "atr": (5, 30),
        "sl": (1.0, 6.0),
        "tp": (1.0, 12.0),
        "trail": (0.5, 4.0)
    }

    # try GP-based Bayes; otherwise hyperopt; else random
    use_method = None
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
        from sklearn.preprocessing import MinMaxScaler
        from scipy.stats import norm
        use_method = "gp"
    except Exception:
        try:
            from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
            use_method = "hyperopt"
        except Exception:
            use_method = "random"

    print(f"[{datetime.now().isoformat()}] Optimization method: {use_method}")

    best_params = None
    best_metrics = None

    if use_method == "gp":
        # GP Bayes (custom simple loop)
        n_init = 20
        n_iter = 60
        random.seed(42); np.random.seed(42)
        X = []; y = []; y_metrics = []

        def sample_random():
            return np.array([random.randint(bounds['atr'][0], bounds['atr'][1]),
                             random.uniform(bounds['sl'][0], bounds['sl'][1]),
                             random.uniform(bounds['tp'][0], bounds['tp'][1]),
                             random.uniform(bounds['trail'][0], bounds['trail'][1])], dtype=float)

        print("Initial random sampling...")
        for _ in range(n_init):
            p = sample_random()
            val, metrics = objective(p, csvs[list(csvs.keys())[0]])
            X.append(p); y.append(val); y_metrics.append(metrics)

        X = np.array(X); y = np.array(y)
        scaler = MinMaxScaler().fit(X)
        Xs = scaler.transform(X)

        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42, n_restarts_optimizer=3)

        # generate candidate pool
        n_candidates = 3000
        candidates = np.array([sample_random() for _ in range(n_candidates)])

        best_idx = int(np.argmin(y)); best_params = X[best_idx]; best_metrics = y_metrics[best_idx]
        print("Start Bayesian iterations...")
        for it in range(n_iter):
            gp.fit(Xs, y)
            cand_s = scaler.transform(candidates)
            mu, sigma = gp.predict(cand_s, return_std=True)
            y_best = np.min(y)
            # Expected Improvement for minimization
            with np.errstate(divide='warn'):
                imp = y_best - mu - 0.01
                Z = imp / (sigma + 1e-9)
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0] = 0.0
            idx = int(np.nanargmax(ei))
            next_x = candidates[idx]
            val, metrics = objective(next_x, csvs[list(csvs.keys())[0]])
            X = np.vstack([X, next_x]); y = np.append(y, val); y_metrics.append(metrics)
            scaler.fit(X)
            Xs = scaler.transform(X)
            if val < np.min(y):
                pass
            if val < best_metrics.get('sharpe', 1e9) if best_metrics else True:
                # update best if objective improved (we track objective though)
                pass
            # update best by objective value:
            if val < float(np.min(y)):
                best_params = next_x; best_metrics = metrics
            if (it+1) % 10 == 0 or it==0:
                print(f"Iter {it+1}/{n_iter}: tried {next_x.astype(float)} -> obj {val:.6f}; best_obj {np.min(y):.6f}")

        # pick overall best from y
        best_idx = int(np.argmin(y))
        best_params = X[best_idx]
        best_metrics = y_metrics[best_idx]

    elif use_method == "hyperopt":
        # hyperopt TPE search
        from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
        space = {
            'atr': hp.quniform('atr', bounds['atr'][0], bounds['atr'][1], 1),
            'sl': hp.uniform('sl', bounds['sl'][0], bounds['sl'][1]),
            'tp': hp.uniform('tp', bounds['tp'][0], bounds['tp'][1]),
            'trail': hp.uniform('trail', bounds['trail'][0], bounds['trail'][1])
        }
        trials = Trials()
        def fn(x):
            p = [int(x['atr']), x['sl'], x['tp'], x['trail']]
            val, metrics = objective(p, csvs[list(csvs.keys())[0]])
            return {'loss': float(val), 'status': STATUS_OK, 'metrics': metrics}
        best = fmin(fn, space, algo=tpe.suggest, max_evals=200, trials=trials, rstate=np.random.RandomState(42))
        # find best trial
        best_trial = min(trials.results, key=lambda r: r['loss'])
        best_metrics = best_trial.get('metrics', None)
        best_params = [int(best['atr']), float(best['sl']), float(best['tp']), float(best['trail'])]

    else:
        # Random search fallback
        best_val = 1e9
        best_p = None
        best_m = None
        iters = 2000
        print(f"Running random search ({iters} iterations)...")
        for i in range(iters):
            p = [random.randint(bounds['atr'][0], bounds['atr'][1]),
                 random.uniform(bounds['sl'][0], bounds['sl'][1]),
                 random.uniform(bounds['tp'][0], bounds['tp'][1]),
                 random.uniform(bounds['trail'][0], bounds['trail'][1])]
            val, metrics = objective(p, csvs[list(csvs.keys())[0]])
            if val < best_val:
                best_val = val; best_p = p; best_m = metrics
            if (i+1) % 200 == 0:
                print(f"Random iter {i+1}/{iters} best_obj {best_val:.6f}")
        best_params = best_p; best_metrics = best_m

    # finalize best params (round atr)
    best_params = [int(round(best_params[0])), float(best_params[1]), float(best_params[2]), float(best_params[3])]
    print(f"Best params found: atr={best_params[0]}, sl={best_params[1]:.4f}, tp={best_params[2]:.4f}, trail={best_params[3]:.4f}")

    # evaluate best on all loaded CSVs and save
    summary = []
    for name, df in csvs.items():
        m = backtest_df_fixed(df, atrPeriod=best_params[0], slMultiplier=best_params[1], tpMultiplier=best_params[2], trailMultiplier=best_params[3])
        summary.append({
            "ticker": name, "atr": best_params[0], "sl": best_params[1], "tp": best_params[2], "trail": best_params[3],
            "equity": m['equity'], "total_return": m['total_return'], "sharpe": m['sharpe'], "maxdd": m['maxdd'],
            "num_trades": m['num_trades'], "win_rate": m['win_rate']
        })
        # save trades per ticker
        trades_df = pd.DataFrame(m['trades'])
        trades_df.to_csv(os.path.join(OUT_DIR, f"{name}_trades_best.csv"), index=False)
        # save equity plot per ticker
        plt.figure(figsize=(10,4)); plt.plot(m['equity_curve']); plt.title(f"{name} Equity - Best Params"); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"equity_{name}_best.png")); plt.close()

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUT_DIR, "optimization_summary.csv"), index=False)
    with open(os.path.join(OUT_DIR, "best_params.json"), "w") as f:
        json.dump({"params": best_params, "method": use_method, "timestamp": time.time()}, f, indent=2)

    # combined equity plot
    plt.figure(figsize=(10,6))
    for name, df in csvs.items():
        m = backtest_df_fixed(df, atrPeriod=best_params[0], slMultiplier=best_params[1], tpMultiplier=best_params[2], trailMultiplier=best_params[3])
        plt.plot(m['equity_curve'], label=name)
    plt.legend(); plt.title("Equity Curves - Best Params"); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "equity_all_best.png")); plt.close()

    print("Saved outputs to:", OUT_DIR)
    print(summary_df.to_string(index=False))
    return best_params, summary_df

if __name__ == "__main__":
    best, summary = run_optimization()
    best_params = {
            "atrPeriod": best[0],
            "slMultiplier": best[1],
            "tpMultiplier": best[2],
            "trailMultiplier": best[3]
        }

    print("\n=== Best Parameter Set ===")
    for k, v in best_params.items():
        print(f"{k}: {v}")

