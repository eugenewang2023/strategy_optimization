nTrails=2000
nFiles=200
#fillOpt=next_open
fillOpt=same_close

###=================================================================================
### Goal: Find parameters that produce many high-PF tickers.
###=================================================================================
min_trades=8
trades_baseline=25
trades_k=0.2
max_trades=45 
max_trades_k=0.2

pf_cap=6

## pf_baseline = 1.2 or 1.5 → require stronger PF before you give meaningful weight.
pf_baseline=2.2
## Smaller pf_k ⇒ faster ramp (PF gets rewarded more aggressively).
pf_k=2.0

weight_pf=0.95
## Your PF distribution is already extreme.
## Using score_power > 1 will amplify noise in infinite PF names.
score_power=1.0

commission_per_side=0.003
loss_floor=0.006
penalty_center=-0.02
penalty_k=8

echo python Bayes_opt_dynamic_SR.py --trials $nTrails --files $nFiles --penalty --penalty-ret-center $penalty_center --penalty-ret-k $penalty_k --min-trades $min_trades --trades-baseline $trades_baseline --trades-k $trades_k --max-trades $max_trades --max-trades-k $max_trades_k --pf-cap $pf_cap --pf-baseline $pf_baseline --pf-k $pf_k --weight-pf $weight_pf --score-power $score_power --commission_rate_per_side $commission_per_side --loss_floor $loss_floor --fill $fillOpt
python Bayes_opt_dynamic_SR.py --trials $nTrails --files $nFiles --penalty --penalty-ret-center $penalty_center --penalty-ret-k $penalty_k --min-trades $min_trades --trades-baseline $trades_baseline --trades-k $trades_k --max-trades $max_trades --max-trades-k $max_trades_k --pf-cap $pf_cap --pf-baseline $pf_baseline --pf-k $pf_k --weight-pf $weight_pf --score-power $score_power --commission_rate_per_side $commission_per_side --loss_floor $loss_floor --fill $fillOpt

echo Done: Bayes_opt_dynamic_SR.py --trials $nTrails --files $nFiles --penalty --penalty-ret-center $penalty_center --penalty-ret-k $penalty_k --min-trades $min_trades --trades-baseline $trades_baseline --trades-k $trades_k --max-trades $max_trades --max-trades-k $max_trades_k --pf-cap $pf_cap --pf-baseline $pf_baseline --pf-k $pf_k --weight-pf $weight_pf --score-power $score_power --commission_rate_per_side $commission_per_side --loss_floor $loss_floor --fill $fillOpt
