nTrails=2000
nFiles=200
#fillOpt=next_open
fillOpt=same_close

###=================================================================================
### Goal: Find parameters that produce many high-PF tickers.
###=================================================================================
## smalleer k less harsh
min_trades=8
trades_baseline=18
trades_k=0.15
max_trades=55
max_trades_k=0.08

ret_floor=0.005
ret_floor_k=14

pf_cap=10

## pf_baseline = 1.2 or 1.5 → require stronger PF before you give meaningful weight.
pf_baseline=2.2
## Smaller pf_k ⇒ faster ramp (PF gets rewarded more aggressively).
pf_k=2.0
pf_floor=2.0
pf_floor_k=1.5
weight_pf=0.9
## Your PF distribution is already extreme.
## Using score_power > 1 will amplify noise in infinite PF names.
score_power=1.0

commission_per_side=0.0006
loss_floor=0.001
penalty_center=-0.02
penalty_k=8

cooldown=3
time_stop=6

#echo python Bayes_opt_dynamic_SR.py --trials $nTrails --files $nFiles --penalty --penalty-ret-center $penalty_center --penalty-ret-k $penalty_k --min-trades $min_trades --trades-baseline $trades_baseline --trades-k $trades_k --max-trades $max_trades --max-trades-k $max_trades_k --ret-floor $ret_floor --ret-floor-k $ret_floor_k --pf-cap $pf_cap --pf-baseline $pf_baseline --pf-k $pf_k --pf-floor $pf_floor --pf-floor-k $pf_floor_k --weight-pf $weight_pf --score-power $score_power --commission_rate_per_side $commission_per_side --loss_floor $loss_floor --fill $fillOpt --cooldown $cooldown --time-stop $time_stop 
#python Bayes_opt_dynamic_SR.py --trials $nTrails --files $nFiles --penalty --penalty-ret-center $penalty_center --penalty-ret-k $penalty_k --min-trades $min_trades --trades-baseline $trades_baseline --trades-k $trades_k --max-trades $max_trades --max-trades-k $max_trades_k --ret-floor $ret_floor --ret-floor-k $ret_floor_k --pf-cap $pf_cap --pf-baseline $pf_baseline --pf-k $pf_k --pf-floor $pf_floor --pf-floor-k $pf_floor_k --weight-pf $weight_pf --score-power $score_power --commission_rate_per_side $commission_per_side --loss_floor $loss_floor --fill $fillOpt --cooldown $cooldown --time-stop $time_stop 
#echo Done: Bayes_opt_dynamic_SR.py --trials $nTrails --files $nFiles --penalty --penalty-ret-center $penalty_center --penalty-ret-k $penalty_k --min-trades $min_trades --trades-baseline $trades_baseline --trades-k $trades_k --max-trades $max_trades --max-trades-k $max_trades_k --ret-floor $ret_floor --ret-floor-k $ret_floor_k --pf-cap $pf_cap --pf-baseline $pf_baseline --pf-k $pf_k --pf-floor $pf_floor --pf-floor-k $pf_floor_k --weight-pf $weight_pf --score-power $score_power --commission_rate_per_side $commission_per_side --loss_floor $loss_floor --fill $fillOpt --cooldown $cooldown --time-stop $time_stop 

echo python Bayes_opt_dynamic_SR.py --trials $nTrails --files $nFiles --penalty --penalty-ret-center $penalty_center --penalty-ret-k $penalty_k --min-trades $min_trades --trades-baseline $trades_baseline --trades-k $trades_k --max-trades $max_trades --max-trades-k $max_trades_k --ret-floor $ret_floor --ret-floor-k $ret_floor_k --pf-cap $pf_cap --pf-baseline $pf_baseline --pf-k $pf_k --pf-floor $pf_floor --pf-floor-k $pf_floor_k --weight-pf $weight_pf --score-power $score_power --commission_rate_per_side $commission_per_side --loss_floor $loss_floor --fill $fillOpt --cooldown $cooldown --time-stop $time_stop
python Bayes_opt_dynamic_SR.py --trials $nTrails --files $nFiles --penalty --penalty-ret-center $penalty_center --penalty-ret-k $penalty_k --min-trades $min_trades --trades-baseline $trades_baseline --trades-k $trades_k --max-trades $max_trades --max-trades-k $max_trades_k --ret-floor $ret_floor --ret-floor-k $ret_floor_k --pf-cap $pf_cap --pf-baseline $pf_baseline --pf-k $pf_k --pf-floor $pf_floor --pf-floor-k $pf_floor_k --weight-pf $weight_pf --score-power $score_power --commission_rate_per_side $commission_per_side --loss_floor $loss_floor --fill $fillOpt --cooldown $cooldown --time-stop $time_stop 
#echo Done: Bayes_opt_dynamic_SR.py --trials $nTrails --files $nFiles --penalty --penalty-ret-center $penalty_center --penalty-ret-k $penalty_k --min-trades $min_trades --trades-baseline $trades_baseline --trades-k $trades_k --max-trades $max_trades --max-trades-k $max_trades_k --ret-floor $ret_floor --ret-floor-k $ret_floor_k --pf-cap $pf_cap --pf-baseline $pf_baseline --pf-k $pf_k --pf-floor $pf_floor --pf-floor-k $pf_floor_k --weight-pf $weight_pf --score-power $score_power --commission_rate_per_side $commission_per_side --loss_floor $loss_floor --fill $fillOpt --cooldown $cooldown --time-stop $time_stop 
