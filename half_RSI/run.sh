nTrails=1500
nFiles=300
fillOpt=next_open
#fillOpt=same_close

###=================================================================================
### Goal: Find parameters that produce many high-PF tickers.
###=================================================================================
## smalleer k less harsh
min_trades=4
trades_baseline=7
trades_k=0.15
max_trades=25
max_trades_k=0.15

ret_floor=0.02
ret_floor_k=14

pf_cap=10

## pf_baseline = 1.2 or 1.5 → require stronger PF before you give meaningful weight.
pf_baseline=2.2
## Smaller pf_k ⇒ faster ramp (PF gets rewarded more aggressively).
pf_k=2.0
pf_floor=2.0
pf_floor_k=2.0
weight_pf=1.0
## Your PF distribution is already extreme.
## Using score_power > 1 will amplify noise in infinite PF names.
score_power=1.0

commission_per_side=0.0006
loss_floor=0.001
penalty_center=-0.02
penalty_k=8

cooldown=3
time_stop=6

#min_tp2sl=1.3

#echo python Bayes_opt_half_RSI.py --trials $nTrails --files $nFiles --penalty --penalty-ret-center $penalty_center --penalty-ret-k $penalty_k --min-trades $min_trades --trades-baseline $trades_baseline --trades-k $trades_k --max-trades $max_trades --max-trades-k $max_trades_k --ret-floor $ret_floor --ret-floor-k $ret_floor_k --pf-cap $pf_cap --pf-baseline $pf_baseline --pf-k $pf_k --pf-floor $pf_floor --pf-floor-k $pf_floor_k --weight-pf $weight_pf --score-power $score_power --commission_rate_per_side $commission_per_side --loss_floor $loss_floor --fill $fillOpt --cooldown $cooldown --time-stop $time_stop --min-tp2sl=$min_tp2sl
#python Bayes_opt_half_RSI.py --trials $nTrails --files $nFiles --penalty --penalty-ret-center $penalty_center --penalty-ret-k $penalty_k --min-trades $min_trades --trades-baseline $trades_baseline --trades-k $trades_k --max-trades $max_trades --max-trades-k $max_trades_k --ret-floor $ret_floor --ret-floor-k $ret_floor_k --pf-cap $pf_cap --pf-baseline $pf_baseline --pf-k $pf_k --pf-floor $pf_floor --pf-floor-k $pf_floor_k --weight-pf $weight_pf --score-power $score_power --commission_rate_per_side $commission_per_side --loss_floor $loss_floor --fill $fillOpt --cooldown $cooldown --time-stop $time_stop --min-tp2sl=$min_tp2sl
#echo Done: Bayes_opt_half_RSI.py --trials $nTrails --files $nFiles --penalty --penalty-ret-center $penalty_center --penalty-ret-k $penalty_k --min-trades $min_trades --trades-baseline $trades_baseline --trades-k $trades_k --max-trades $max_trades --max-trades-k $max_trades_k --ret-floor $ret_floor --ret-floor-k $ret_floor_k --pf-cap $pf_cap --pf-baseline $pf_baseline --pf-k $pf_k --pf-floor $pf_floor --pf-floor-k $pf_floor_k --weight-pf $weight_pf --score-power $score_power --commission_rate_per_side $commission_per_side --loss_floor $loss_floor --fill $fillOpt --cooldown $cooldown --time-stop $time_stop --min-tp2sl=$min_tp2sl 

echo python Bayes_opt_half_RSI.py --trials $nTrails --files $nFiles \
  --penalty --penalty-ret-center $penalty_center --penalty-ret-k $penalty_k \
  --min-trades $min_trades --trades-baseline $trades_baseline --trades-k $trades_k \
  --max-trades $max_trades --max-trades-k $max_trades_k \
  --ret-floor $ret_floor --ret-floor-k $ret_floor_k \
  --pf-cap $pf_cap --pf-baseline $pf_baseline --pf-k $pf_k \
  --pf-floor $pf_floor --pf-floor-k $pf_floor_k \
  --weight-pf $weight_pf --score-power $score_power \
  --commission_rate_per_side $commission_per_side --loss-floor $loss_floor \
  --fill $fillOpt --cooldown $cooldown --time-stop $time_stop \
  --tp2sl-auto \
  --tp2sl-base 1.20 \
  --tp2sl-sr0 30 \
  --tp2sl-k 0.01 \
  --tp2sl-min 1.10 \
  --tp2sl-max 1.80


python Bayes_opt_half_RSI.py --trials $nTrails --files $nFiles \
  --penalty --penalty-ret-center $penalty_center --penalty-ret-k $penalty_k \
  --min-trades $min_trades --trades-baseline $trades_baseline --trades-k $trades_k \
  --max-trades $max_trades --max-trades-k $max_trades_k \
  --ret-floor $ret_floor --ret-floor-k $ret_floor_k \
  --pf-cap $pf_cap --pf-baseline $pf_baseline --pf-k $pf_k \
  --pf-floor $pf_floor --pf-floor-k $pf_floor_k \
  --weight-pf $weight_pf --score-power $score_power \
  --commission_rate_per_side $commission_per_side --loss_floor $loss_floor \
  --fill $fillOpt --cooldown $cooldown --time-stop $time_stop \
  --tp2sl-auto \
  --tp2sl-base 1.20 \
  --tp2sl-sr0 30 \
  --tp2sl-k 0.01 \
  --tp2sl-min 1.10 \
  --tp2sl-max 1.80
