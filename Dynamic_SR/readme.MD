Trend & signals: HA close crossing a stateful SR line (SR moves using HA highs/lows and a REAL-ATR band). Risk/exits: entirely driven by REAL OHLC + REAL ATR.

Risk control / exits (REAL-price-based, intrabar checks). Once in a trade, each bar defines three key distances using REAL ATR: 
Hard stop level (uses current bar low)
stop_level = realLow[i] - real_atr[i] * slMult
Take-profit limit level (uses current bar high)
	tgt_level = realHigh[i] + real_atr[i] * tpMult
Trailing distance (also tpMult)
	trail_dist = real_atr[i] * tpMult

Trailing stop behavior (TradingView-like activation)
If use_trailing_exit=True:
Activation condition
Trailing becomes active once price has moved far enough in favor:
activates when realHigh[i] >= entry + trail_dist
Once active
track trail_high_since = max(trail_high_since, realHigh[i])
trail_stop = trail_high_since - trail_dist
So the trailing stop only starts after price reaches entry + ATR*tpMult, then trails the running high by ATR*tpMult.
Exit priority/order (conservative intrabar for long)
Within each bar, it checks exits in this order:
Hard stop (only if enabled)
hard_stop_enabled = (not use_trailing_exit) OR (trail_mode == "trail_plus_hard_sl")
If enabled and realLow[i] <= stop_level → exit at stop_level
Trailing stop (if trailing is active)
If realLow[i] <= trail_stop → exit at trail_stop
TP limit
If realHigh[i] >= tgt_level → exit at tgt_level
Time stop (loss-only)
If bars_in_trade >= time_stop_bars and current fill-based return <= 0
Exit at fill price (same_close or next_open)
Close on sell signal
If close_on_sellSignal and sell_sig
Exit at fill price
On exit:
commission applied again (per-side)
equity updated by trade return: equity *= (1 + (exit-entry)/entry)
gp/gl accumulate using trade return fractions, not dollars
if the trade is a loser, set cooldown_left = cooldown_bars

End-of-data handling
If still in position at the final bar:
force exit at realClose[-1] (commission applied)
counts into gp/gl the same way.

Big takeaways / “mental model”
Trend & signals: HA close crossing a stateful SR line (SR moves using HA highs/lows and a REAL-ATR band).
Risk/exits: entirely driven by REAL OHLC + REAL ATR, with:
optional delayed activation trailing stop (only after entry + ATR*tpMult),
optional hard SL (always, or only if trail_plus_hard_sl),
breakout-style TP at high + ATR*tpMult,
optional sell-signal exit and loss-only time stop,
cooldown only after losing trades.
If you want, I can also summarize the “why this might behave oddly” parts (stops/targets being anchored to current bar highs/lows instead of entry) and what that implies for fills and PF.
