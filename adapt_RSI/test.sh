dataFile="data\AAPL.parquet"

if [ "$#" -gt 0 ]; then
    dataFile=$1
    shift
fi

echo python Bayes_opt_adapt_RSI.py --mode  test --test-file $dataFile $*
python Bayes_opt_adapt_RSI.py --mode  test --test-file  $dataFile $*

## smoke test
#echo python Bayes_opt_adapt_RSI.py --mode optimize --trials 3 
#python Bayes_opt_adapt_RSI.py --mode optimize --trials 3 
