nTrails=2000
nFiles=1000
#fillOpt=next_open
#fillOpt=intrabar
fillOpt=same_close

echo python Bayes_opt_adapt_half_RSI.py --trials $nTrails --files $nFiles --fill $fillOpt
python Bayes_opt_adapt_half_RSI.py --trials $nTrails --files $nFiles --fill $fillOpt 
