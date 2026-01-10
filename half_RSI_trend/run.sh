nTrails=2000
nFiles=1000
fillOpt=same_close
pf_cap=30
alpha=0.5
## params of the S-curve, the weighting function on the number of trades
center=3
beta=3
## smaller the beta, more number of trades, but harder to converge/optimize
#β=0.5 implies:
#~5 trades/ticker → ~0.92 weight
#~3 trades/ticker → ~0.78 weight
#~1 trades/ticker → ~0.39 weight
#β=0.4 implies:
#~5 trades/ticker → weight ≈ 0.865
#~3 trades/ticker → weight ≈ 0.699
#~1 trade/ticker → weight ≈ 0.330
#β=0.3 implies:
#~5 trades/ticker → weight ≈ 0.777
#~3 trades/ticker → weight ≈ 0.593
#~1 trade/ticker → weight ≈ 0.259
#β=0.2 implies:
#~5 trades/ticker → weight ≈ 0.632
#~3 trades/ticker → weight ≈ 0.451
#~1 trade/ticker → weight ≈ 0.181
#β=0.1 implies:
#~5 trades/ticker → weight ≈ 0.393
#~3 trades/ticker → weight ≈ 0.259
#~1 trade/ticker → weight ≈ 0.095

echo python Bayes_opt_half_RSI_trend.py --trials $nTrails --files $nFiles --pf-cap $pf_cap --alpha $alpha --beta $beta --center $center --fill $fillOpt
python Bayes_opt_half_RSI_trend.py --trials $nTrails --files $nFiles --pf-cap $pf_cap --alpha $alpha --beta $beta --center $center --fill $fillOpt
