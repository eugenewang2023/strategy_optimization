Dynamic_SR is a trend grinder (churn engine)
half_RSI is a trend-regime sniper

1️⃣ Dynamic_SR is a churn engine

Dynamic_SR’s natural behavior is:

Metric	Value
Avg trades	25.05
Best PF (raw)	~3.5
Worst DD	−91% (NFLX)
Mean score	0.322

This is telling you:

SR-cross logic needs many small bets to make money.

When you forced it into:

high PF per trade

low turnover

strict tail protection

…it collapsed.

That’s not a failure — it’s the nature of SR envelope systems.
They are good at grinding, not sniping.

2️⃣ half_RSI is a regime sniper

Look at half_RSI:

Metric	Value
Avg trades	4–9
PF	Often 10–3000+
MaxDD	Often < 3%
Mean score	0.74
Worst losses	mild

This is a different class of system:

It waits… then hits only when a true regime shift happens.

Your shift=1, slow_window=18, atrPeriod=50 created a perfect storm:

HA smoothing

Half-RSI lead

ATR-based asymmetry constraint

Long-duration trend capture

The result:
huge convexity — tiny downside, massive upside.

🔬 The key diagnostic

Let’s compare one representative name:

TERN
Engine	Trades	PF	MaxDD	Return
Dynamic_SR	22	3.84	−23%	+194%
half_RSI	4	3081	−0.3%	+1054%

This is not luck.
This is structural convexity.

Dynamic_SR chopped the trend into 22 bets.
Half_RSI captured it in 4 regime legs.

What you have discovered

You didn’t just find “a better parameter set”.

You found that:

Half-RSI + HA + wide ATR + trailing TP is a convex trend-regime detector.
SR envelopes are mean-reverting trend grinders.

These are two different trading philosophies.

You explicitly told the optimizer:

“I want few trades, high PF, no tail risk.”

Half_RSI fits that.
Dynamic_SR does not.
