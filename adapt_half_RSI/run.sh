nTrails=2000
nFiles=1000
alpha=0.5
fillOpt=same_close
#fillOpt=next_open
#fillOpt=intrabar

echo python Bayes_opt_adapt_half_RSI.py --trials $nTrails --files $nFiles --alpha $alpha --fill $fillOpt
python Bayes_opt_adapt_half_RSI.py --trials $nTrails --files $nFiles --alpha $alpha --fill $fillOpt 
