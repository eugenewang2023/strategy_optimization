## default nTrails=200
nTrails=2000
nFiles=200
alpha=0.25
fillOpt=same_close
#fillOpt=next_open
#fillOpt=intrabar

#echo python Bayes_opt_adapt_RSI.py --mode optimize --trials $nTrails --sample 5000 --files 5 --workers -1
#python Bayes_opt_adapt_RSI.py --mode optimize --trials $nTrails --sample 5000 --files 5 --workers -1

echo python Bayes_opt_adapt_RSI.py --mode optimize --trials $nTrails --files $nFiles --alpha $alpha --fill $fillOpt
python Bayes_opt_adapt_RSI.py --mode optimize --trials $nTrails --files $nFiles --alpha $alpha --fill $fillOpt

## smoke test
#echo python Bayes_opt_adapt_RSI.py --mode optimize --trials 3 
#python Bayes_opt_adapt_RSI.py --mode optimize --trials 3 
