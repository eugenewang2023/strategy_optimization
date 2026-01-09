nTrails=2000
nFiles=1000
fillOpt=same_close

echo python Bayes_opt_dynamic_SR.py --trials $nTrails --files $nFiles --fill $fillOpt 
python Bayes_opt_dynamic_SR.py --trials $nTrails --files $nFiles --fill $fillOpt 
