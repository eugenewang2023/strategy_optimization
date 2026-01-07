## default nTrails=200
nTrails=2000
nFiles=200
#echo python Bayes_opt_adapt_RSI.py --mode optimize --trials $nTrails --sample 5000 --files 5 --workers -1
#python Bayes_opt_adapt_RSI.py --mode optimize --trials $nTrails --sample 5000 --files 5 --workers -1

echo python Bayes_opt_adapt_RSI.py --mode optimize --trials $nTrails --files $nFiles --workers 1
python Bayes_opt_adapt_RSI.py --mode optimize --trials $nTrails --files $nFiles --workers 1

## smoke test
#echo python Bayes_opt_adapt_RSI.py --mode optimize --trials 3 
#python Bayes_opt_adapt_RSI.py --mode optimize --trials 3 
