nTrails=2000
nFiles=200
#fillOpt=next_open
fillOpt=same_close
min_trades=2
center=3
beta=1
pf_center=2
pf_beta=1
pf_cap=50

## use --penalty to turn penalty ON

#alpha=0.5
## params of the S-curve, the weighting function on the number of trades
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


echo python Bayes_opt_half_RSI_trend.py --trials $nTrails --files $nFiles --min-trades $min_trades --center $center --beta $beta --pf-center $pf_center --pf-beta $pf_beta --pf-cap $pf_cap --fill $fillOpt
python Bayes_opt_half_RSI_trend.py --trials $nTrails --files $nFiles --min-trades $min_trades --center $center --beta $beta --pf-center $pf_center --pf-beta $pf_beta --pf-cap $pf_cap --fill $fillOpt

echo Done: Bayes_opt_half_RSI_trend.py --trials $nTrails --files $nFiles --min-trades $min_trades --center $center --beta $beta --pf-center $pf_center --pf-beta $pf_beta --pf-cap $pf_cap --fill $fillOpt
