1/25/2026
Vidya_RSI_v0.2.2
The Vidya_RSI strategy has been optimized into an ultra-sensitive, short-term trading system with a best score of 0.3877. Key parameters show extreme sensitivity (fastPeriod=6, threshold=0.0016) and short holding periods (9 bars). While risk controls have tightened (cooldown=3, regime_ratio=3.5), the strategy now faces overfitting risks due to its highly tuned nature. Optimization progress has plateaued after 900 trials, indicating the current parameter set represents a local optimum. Next steps should prioritize out-of-sample validation with more data before further optimization.

=== OPTIMIZER HEALTH REPORT ===
Trials completed: 300
Score mean:       0.253143
Score median:     0.271978
Score std:        0.094747
Degenerate trials (0 or None): 0
Best-so-far monotonic: False

Parameter usage counts:
  regime_ratio: 300
  vidya_len: 300
  vidya_smooth: 300
  fastPeriod: 300
  slowPeriod: 300
  time_stop_bars: 300
  threshold: 300
  vol_floor_mult: 300
  cooldown_bars: 300

Parameter entropy:
  regime_ratio: std=0.8790
  vidya_len: std=4.5990
  vidya_smooth: std=13.2229
  fastPeriod: std=3.9666
  slowPeriod: std=4.7455
  time_stop_bars: std=3.3698
  threshold: std=0.0021
  vol_floor_mult: std=0.0541
  cooldown_bars: std=0.8455

=== END REPORT ===

Vidya_RSI Hybrid Adaptive ▒ done.
next_open CSV: output\vidya_RSI_per_ticker_next_open_20260125_210339.csv
Summary TXT: output\vidya_RSI_best_20260125_210339.txt

Done with phase F

═══════════════════════════════════════════════════════════════
Multi-phase optimization complete.
Logs saved in logs/
Objective used: Enhanced
═══════════════════════════════════════════════════════════════


vl – VIDYA EMA length (efficiency lookback window)
vs – VIDYA EMA smoothing factor (base EMA responsiveness)
fp – Fast ZLEMA period applied to VIDYA
sp – Slow ZLEMA period applied to VIDYA
ts – Time-stop in bars (max hold if trade does not move favorably)

1/21/2026
Vidya_RSI_v0.2 with many improvements suggested by CoPilot
1. Your VIDYA/ZLEMA pipeline is good — but the regime filter is too weak
2. Your exit logic is the biggest bottleneck
3. Your PF_unification is excellent — but you’re not using PF_diag in scoring
4. Your objective_penalty_multiplier is powerful — but missing a key penalty
5. Your backtest loop is Python‑slow
6. Your Optuna search space is too narrow
7. Coverage penalty is good — but should be asymmetric
8. Your CSV output is excellent — but missing key diagnostics
9. Shell script + Python script integration
10. Add a “stability score”

1/19/2026
Vidya_RSI_v0.2
Using GMM low-vol data:

Quick Summary – Current Strengths & Weak Points
Strengths,Weaknesses / Bottlenecks
Very good median PF (~2.75),Many very strong individual PFs are heavily capped at 4.0
Excellent coverage (100%),Still quite a lot of zero/near-zero contribution tickers at the bottom
Very high mean ticker score (~0.848),Large tail of negative/very low score tickers dragging average down
Quite clean number of negative trades (only 15),Many tickers show large drawdowns (>20–30%)
Very strong top tickers (many >10–50+ PF),"Edge per trade is good (median ~3bp), but not exceptional"

Most Realistic Improvement Levers (ordered roughly by expected impact/ease)
Priority,Area,Concrete suggestions,Expected difficulty,Realistic upside potential
1,Better separation of good vs mediocre,"score_power → 1.8–2.2
weight_pf → 0.75–0.82",easy,★★★★☆
2,Stronger punishment of bad tails,"Increase tail-protection ret floor to -0.12 / -0.10 (instead of -0.15)
Stronger PF-floor penalty (k=7–9 instead of 5)",easy–medium,★★★★
3,Time-stop tuning + possible dynamic,"Try range ts=12–40
Consider adding very soft time-stop penalty instead of hard cut",medium,★★★☆
4,Raise quality bar at bottom,"min_trades gate → 3–5 (instead of 1)
Or introduce soft min_trades penalty",easy,★★★
5,Reduce extreme capping effect,"Increase cap from 4.0 → 5.0–6.0 (diagnostic cap)
OR make capping softer/non-linear",easy,★★–★★★
6,Edge per trade vs frequency trade-off,"Try to push GL/trade target a bit higher (0.0035–0.0045 instead of 0.003)
Or make k steeper",medium,★★–★★★
7,Parameter space refinement,"fp: try 4–12, sp: 18–28, vl/vs asymmetry (vl>vs more often good in trending low-vol)",medium–hard,★★
8,Regime filter / volatility conditioning,Add simple regime detection (e.g. 20-bar ATR ratio) as additional feature or soft gate,hard,★★★–★★★★ (if succeeds)

Quick Practical Prioritization Table (2025–2026 view)
If you can run only 3 more serious optimizations in the near future, most experienced quant traders would probably choose this order:

A or B (stronger non-linear quality emphasis + higher pf weight)
→ usually gives the biggest single jump when you're already at ~0.72–0.73
C (raise min_trades to 3–5)
→ removes many statistically unreliable results, usually improves robustness
D (tighter tail protection)
→ very often helps when many tickers still have -15% to -40% maxdd

A/B/C/D results: 1138/1143/1149/1154: Stronger quality emphasis/Aggressive quality push/Min trades safety net/Tighter tail protection focus
Overall performance comparison (aggregate)
Run	Avg Total Return	Median Return	Avg Profit Factor (eff)	Median PF	Avg Max DD	Eligible %	Avg # Trades
1143	0.1473	0.1316	34.81	2.91	-0.0766	100%	6.72
1149	0.1448	0.1272	37.97	2.48	-0.0828	99.3%	6.93
1154	0.1146	0.0979	45.79	2.67	-0.0767	100%	6.22
1138	0.1104	0.1162	44.64	2.54	-0.0807	100%	6.35
Key takeaways
1138 is the weakest overall on returns.
1143 has the best overall balance → highest average & median return, best median PF, and shallow drawdown.
1149 is a close second, slightly higher risk (worse DD, lower median PF).
1154 looks over-optimized → very high PF but noticeably lower returns.

# run_phase "B - Aggressive quality push v2" 1 0.82 2.2 -0.10 2.5 8 0.003 12
# run_phase "E - Balanced Aggressive + Safety" 3           0.80        2.0       -0.105    2.4     7     0.003
# run_phase "F - Higher Edge Target"   2           0.80        2.1       -0.11     2.4     7     0.0035  14

Phase E vs. Phase B: Phase E optimized the returns slightly (+0.9% avg return) without changing the overall ticker universe or quality scores significantly.
The Phase F Shift: There is a distinct change in Phase F. The total number of trades dropped by ~7%, and the average return fell by over 4%. This suggests that the conditions or filters in Phase F are more restrictive or are capturing a different market window where fewer high-performing opportunities met the criteria.
	
Option B (Aggressive quality push) result:
=== OBJECTIVE DEGENERACY DIAGNOSTICS ===
penalty_mult: 1.000000
zero_loss_mult: 1.000000
cap_mult: 1.000000
glpt_mult: 1.000000
=== BEST PARAMS ===
fp: 6
reg_ratio: 2.519244903242621
sp: 21
ts: 5
vl: 9
vs: 15
objective_score: 0.680926

Using GMM high-vol data:
You’re seeing classic objective degeneracy: the optimizer is “winning” by finding parameter sets that create tiny (or zero) losses, which explodes effective PF and pushes ticker scores toward 1.0, even though the strategy isn’t truly better.
Here’s what your diagnostics are telling us, and what to change right now.
What the diagnostics say (in plain English)
Median PF (diag) is sane: ~2.75 (capped at 4).
Avg PF (eff) is insane: 54 (and still 5.50 even excluding zero-loss) → that’s the degeneracy.
PF capped tickers: 34/123 (27.6%) → too many names hitting the PF cap.
Zero-loss tickers: 6/123 (4.9%) → not huge, but enough to distort averages and gradients.
min_trades gate = 1 → this is the biggest enabler. Single-trade / few-trade names are easy to “game.”
Avg trades/ticker ~6.8 is low enough that PF becomes noisy.
score_power = 1.5 amplifies extremes (PF spikes) and worsens degeneracy.
loss_floor=0.001 is very small; any low-loss regime looks “godlike.”
Also: Tail-protection ret floor/k: -0.15 / 2.0 is basically letting negative returns slide with minimal penalty (depending on your exact scoring), which can allow PF gaming without enough return pressure.

The fastest fixes (highest ROI)
1) Raise the min_trades gate
Set min_trades to at least 6 (better: 8–12).
2) Increase your loss_floor used for PF scoring
3) Lower score_power back to 1.0
4) Tighten PF influence
5) Add / strengthen a GL-per-trade floor gate (you already have it)

Secondary fixes (after you apply the above)
A) Reduce PF cap pressure
B) Raise trades_baseline
C) Make ret_floor >=0 (currently < 0)

=== Objective Degeneracy Diagnostics ===
Min GL/trade target/k:           0.003000 / 12.000
Median GL per trade (eligible):  0.029687
GLPT mult (best eval):           1.000000
Median Profit Factor (diag):      2.753271   (cap=4.0)
Avg Profit Factor (diag cap):     2.666096   (cap=4.0)
Median Profit Factor (eff):       2.753271   (loss_floor=0.001)
Avg Profit Factor (eff):          54.008552   (loss_floor=0.001)
Avg PF (eff, excl zero-loss):     5.496170
Zero-loss tickers (gl==0,gp>0):   6/123 = 0.049
PF capped (eff>cap):              34/123 = 0.276
Median PF (raw, finite only):     2.753271
Trimmed Mean PF (raw, 5% trim):   5.767775
Avg Trades / Ticker:              6.805
Mean Ticker Score (eligible):     0.848430
Coverage (eligible/total):        123/123 = 1.000
Penalty enabled:                  enabled (soft)
Soft penalty center/k:            0.01 / 10.0
Tail-protection ret floor/k:      -0.15 / 2.0
Max-trades penalty:               max_trades=100, k=0.1
PF-floor penalty:                 pf_floor=1.0, k=5.0
cooldown_bars:                    1 (opt=False) [loss-only]
time_stop_bars:                   24 (0=disabled) (opt=True)
num_neg (return<=0):              15
commission_rate_per_side:         0.000600
PF ROC baseline/k:                1.150 / 2.500
Trades ROC baseline/k:            3.000 / 1.200
weight_pf:                        0.700
score_power:                      1.500
min_trades gate:                  1
loss_floor (scoring):             0.001
threshold_mode:                   fixed
Score (objective):                0.726109

=== BEST PARAMS ===
fp: 5
reg_ratio: 2.61928626415429
sp: 21
ts: 24
vl: 6
vs: 5
best_score (OPTUNA objective): 0.7261093621875501
fill_mode (final report): next_open

=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===
ticker profit_factor_raw_print  profit_factor_eff  profit_factor_diag  pf_capped  zero_loss  num_trades  ticker_score  total_return  gross_profit  gross_loss  gl_per_trade  atr_pct_med         maxdd  eligible
  MKSI                3.980660           3.980660            3.980660          0          0           9      0.998778      0.262653      0.328833    0.082608      0.009179     0.039798 -8.095752e-02         1
  AXTI               16.612523          16.612523            4.000000          1          0           8      0.998044      2.718853      1.608072    0.096799      0.012100     0.080127 -9.679877e-02         1
  GLUE              158.627427         158.627427            4.000000          1          0           8      0.998044      1.711232      1.106896    0.006978      0.000872     0.061259 -6.977961e-03         1
  APLD                4.641083           4.641083            4.000000          1          0           8      0.998044      1.613247      1.508982    0.325136      0.040642     0.095765 -1.920612e-01         1
  VICR                8.237725           8.237725            4.000000          1          0           8      0.998044      0.710310      0.665013    0.080728      0.010091     0.046601 -8.072779e-02         1
  PASG               61.271736          61.271736            4.000000          1          0           7      0.995486      4.379466      2.901316    0.047352      0.006765     0.057175 -2.404946e-02         1
    NG       1000000000.000000        1000.000000            4.000000          1          1           7      0.995486      1.184419      0.845735    0.000000      0.000000     0.048211 -1.000089e-12         1
  EOSE                4.972629           4.972629            4.000000          1          0           7      0.995486      1.066401      1.074323    0.216047      0.030864     0.084444 -2.060859e-01         1
  ZURA                6.701865           6.701865            4.000000          1          0           7      0.995486      1.009449      0.938884    0.140093      0.020013     0.072542 -8.431464e-02         1
    PL                5.739148           5.739148            4.000000          1          0           7      0.995486      0.836393      0.854940    0.148966      0.021281     0.067776 -1.489663e-01         1
    MU                9.995086           9.995086            4.000000          1          0           7      0.995486      0.813893      0.713291    0.071364      0.010195     0.043189 -7.136420e-02         1
  COHR               19.832414          19.832414            4.000000          1          0           7      0.995486      0.786871      0.659220    0.033240      0.004749     0.055058 -2.619185e-02         1
  TERN                5.558115           5.558115            4.000000          1          0           7      0.995486      0.751444      0.794262    0.142901      0.020414     0.061826 -1.429013e-01         1
  SPHR                6.357075           6.357075            4.000000          1          0           7      0.995486      0.624128      0.641279    0.100876      0.014411     0.041048 -1.008764e-01         1
    AG                4.116912           4.116912            4.000000          1          0           7      0.995486      0.526112      0.642664    0.156104      0.022301     0.051359 -1.091944e-01         1
  OLMA                3.345802           3.345802            3.345802          0          0           9      0.995349      0.862669      1.196138    0.357504      0.039723     0.072360 -2.238872e-01         1
  ASTS                3.932949           3.932949            3.932949          0          0           7      0.995332      0.999090      1.141758    0.290306      0.041472     0.080732 -2.725603e-01         1
   ERO                3.909484           3.909484            3.909484          0          0           7      0.995272      0.406211      0.502023    0.128412      0.018345     0.041433 -1.270300e-01         1
   AMD                3.893461           3.893461            3.893461          0          0           7      0.995229      0.359425      0.459569    0.118036      0.016862     0.041689 -1.156915e-01         1
   HLF                3.691423           3.691423            3.691423          0          0           7      0.994507      0.410073      0.535362    0.145029      0.020718     0.046047 -1.182422e-01         1
  RCUS                3.487678           3.487678            3.487678          0          0           7      0.993302      0.344619      0.463779    0.132976      0.018997     0.055183 -7.647840e-02         1
  INTC                3.236077           3.236077            3.236077          0          0           7      0.990667      0.500932      0.673989    0.208273      0.029753     0.046284 -1.982880e-01         1
    BE                3.034721           3.034721            3.034721          0          0          11      0.990630      1.069682      1.428711    0.470788      0.042799     0.073465 -3.659800e-01         1
   NEO                3.022262           3.022262            3.022262          0          0           8      0.989260      0.311607      0.474233    0.156913      0.019614     0.051330 -1.463746e-01         1
  CRSP                3.088118           3.088118            3.088118          0          0           7      0.988156      0.400814      0.592880    0.191987      0.027427     0.057046 -1.744703e-01         1
  RKLB                2.956400           2.956400            2.956400          0          0           8      0.987558      0.471049      0.702134    0.237496      0.029687     0.070309 -2.191820e-01         1
  QBTS               24.184449          24.184449            4.000000          1          0           6      0.987214      1.543773      1.131009    0.046766      0.007794     0.092050 -4.676597e-02         1
  GOSS                7.727772           7.727772            4.000000          1          0           6      0.987214      1.020622      0.924404    0.119621      0.019937     0.076667 -1.196211e-01         1
  PSNL                7.942369           7.942369            4.000000          1          0           6      0.987214      0.780857      0.731689    0.092125      0.015354     0.073062 -5.981366e-02         1
  LABU       1000000000.000000        1000.000000            4.000000          1          1           6      0.987214      0.770590      0.613153    0.000000      0.000000     0.057585 -1.000089e-12         1
     U               11.008308          11.008308            4.000000          1          0           6      0.987214      0.700905      0.637948    0.057951      0.009659     0.056672 -5.795147e-02         1
  XERS                4.667782           4.667782            4.000000          1          0           6      0.987214      0.625521      0.706445    0.151345      0.025224     0.044408 -1.513449e-01         1
  PVLA                4.174939           4.174939            4.000000          1          0           6      0.987214      0.552310      0.650611    0.155837      0.025973     0.052025 -8.100600e-02         1
  ANAB                5.288729           5.288729            4.000000          1          0           6      0.987214      0.546137      0.594790    0.112464      0.018744     0.053579 -1.124637e-01         1
    SA               30.440352          30.440352            4.000000          1          0           6      0.987214      0.474141      0.431033    0.014160      0.002360     0.042689 -1.041376e-02         1
  SNDL                4.856912           4.856912            4.000000          1          0           6      0.987214      0.444943      0.519052    0.106869      0.017811     0.046883 -6.608550e-02         1
  ABUS                9.181643           9.181643            4.000000          1          0           6      0.987214      0.351243      0.357081    0.038891      0.006482     0.042311 -3.855420e-02         1
  IONQ                5.278344           5.278344            4.000000          1          0           6      0.987214      0.350577      0.408920    0.077471      0.012912     0.084476 -7.747119e-02         1
  TTMI                8.571075           8.571075            4.000000          1          0           6      0.987214      0.311860      0.328968    0.038381      0.006397     0.046088 -3.815380e-02         1
  DBRG                4.037648           4.037648            4.000000          1          0           6      0.987214      0.168484      0.239614    0.059345      0.009891     0.046211 -5.817168e-02         1
   PBF                3.872055           3.872055            3.872055          0          0           6      0.986898      0.459713      0.585084    0.151104      0.025184     0.054975 -7.958161e-02         1
  BITF                3.652429           3.652429            3.652429          0          0           6      0.986053      0.782755      1.021803    0.279760      0.046627     0.079748 -2.738430e-01         1
   CDE                2.877004           2.877004            2.877004          0          0           8      0.985109      0.422270      0.631347    0.219446      0.027431     0.053784 -1.780240e-01         1
   CLS                3.459368           3.459368            3.459368          0          0           6      0.984816      0.464069      0.631642    0.182589      0.030431     0.055521 -1.825889e-01         1
  CENX                2.863192           2.863192            2.863192          0          0           8      0.984633      0.392306      0.578016    0.201878      0.025235     0.048258 -1.362830e-01         1
  SOFI                2.858311           2.858311            2.858311          0          0           8      0.984461      0.372255      0.566114    0.198059      0.024757     0.053769 -1.133884e-01         1
    GH                2.904495           2.904495            2.904495          0          0           7      0.983464      0.405429      0.600254    0.206664      0.029523     0.045151 -1.274977e-01         1
  OMER                3.287870           3.287870            3.287870          0          0           6      0.983091      0.722650      1.028315    0.312760      0.052127     0.073763 -2.575670e-01         1
   DBI                2.892409           2.892409            2.892409          0          0           7      0.983075      0.749532      1.127729    0.389893      0.055699     0.068329 -2.054784e-01         1
  AMRC                3.066753           3.066753            3.066753          0          0           6      0.979462      0.363869      0.549341    0.179128      0.029855     0.057315 -1.060529e-01         1
  DIBS                3.052592           3.052592            3.052592          0          0           6      0.979156      0.261710      0.383640    0.125677      0.020946     0.041066 -8.702375e-02         1
  XPEV                3.022431           3.022431            3.022431          0          0           6      0.978466      0.232031      0.353419    0.116932      0.019489     0.044361 -9.091698e-02         1
  FUTU                2.753271           2.753271            2.753271          0          0           7      0.977676      0.159928      0.258433    0.093864      0.013409     0.048077 -7.135780e-02         1
  FTAI                2.681254           2.681254            2.681254          0          0           8      0.976628      0.331002      0.521885    0.194642      0.024330     0.042653 -1.321485e-01         1
   CLF                2.619054           2.619054            2.619054          0          0           8      0.972992      0.401968      0.650158    0.248242      0.031030     0.060995 -1.340299e-01         1
  NEGG                2.538845           2.538845            2.538845          0          0           7      0.964913      0.744597      1.651285    0.650408      0.092915     0.090473 -4.024639e-01         1
  SNDK       1000000000.000000        1000.000000            4.000000          1          1           5      0.961973      1.611920      1.075683    0.000000      0.000000     0.061541 -1.000089e-12         1
  COGT       1000000000.000000        1000.000000            4.000000          1          1           5      0.961973      1.280193      0.907036    0.000000      0.000000     0.058280 -1.000089e-12         1
  SATS       1000000000.000000        1000.000000            4.000000          1          1           5      0.961973      0.924204      0.715444    0.000000      0.000000     0.049401 -1.000089e-12         1
  SNBR                4.445612           4.445612            4.000000          1          0           5      0.961973      0.714368      0.849569    0.191103      0.038221     0.080120 -1.911029e-01         1
   MDB                6.652471           6.652471            4.000000          1          0           5      0.961973      0.427809      0.454981    0.068393      0.013679     0.040933 -6.839278e-02         1
  ASTL       1000000000.000000        1000.000000            4.000000          1          1           5      0.961973      0.403380      0.355003    0.000000      0.000000     0.055804 -1.000089e-12         1
  RGTI                2.610663           2.610663            2.610663          0          0           6      0.961723      0.284276      0.495323    0.189731      0.031622     0.093788 -1.057317e-01         1
  BBAI                2.466958           2.466958            2.466958          0          0           7      0.958988      0.383506      0.755017    0.306052      0.043722     0.100749 -2.262000e-01         1
   AEO                3.116285           3.116285            3.116285          0          0           5      0.955272      0.208926      0.315261    0.101166      0.020233     0.047613 -7.712735e-02         1
  IAUX                3.103390           3.103390            3.103390          0          0           5      0.955027      0.405424      0.593978    0.191397      0.038279     0.052215 -1.084231e-01         1
  MRNA                2.339164           2.339164            2.339164          0          0           7      0.945732      0.227597      0.430180    0.183903      0.026272     0.054949 -1.424771e-01         1
  DSGN                2.395885           2.395885            2.395885          0          0           6      0.943939      0.170875      0.330027    0.137747      0.022958     0.060832 -1.309492e-01         1
  TNDM                2.387745           2.387745            2.387745          0          0           6      0.943078      0.232815      0.464197    0.194408      0.032401     0.055367 -1.944081e-01         1
  RIGL                2.711019           2.711019            2.711019          0          0           5      0.942360      0.197795      0.327469    0.120792      0.024158     0.052840 -1.207918e-01         1
   OGI                2.303301           2.303301            2.303301          0          0           7      0.941280      0.302831      0.556155    0.241460      0.034494     0.044829 -1.791607e-01         1
  LYFT                2.364877           2.364877            2.364877          0          0           6      0.940574      0.271554      0.506459    0.214159      0.035693     0.045033 -1.624575e-01         1
   CCJ                2.252610           2.252610            2.252610          0          0           8      0.936866      0.290013      0.546292    0.242515      0.030314     0.041217 -1.745292e-01         1
   WDC                2.154573           2.154573            2.154573          0          0           8      0.921148      0.164446      0.320750    0.148869      0.018609     0.048468 -1.166163e-01         1
    ON                2.142903           2.142903            2.142903          0          0           7      0.916565      0.117868      0.238584    0.111337      0.015905     0.043540 -8.252766e-02         1
   GEV                2.372114           2.372114            2.372114          0          0           5      0.916539      0.180863      0.344458    0.145211      0.029042     0.041992 -1.420195e-01         1
  VSCO                2.089589           2.089589            2.089589          0          0           8      0.908827      0.192535      0.394071    0.188588      0.023573     0.048295 -1.164547e-01         1
   RIG                2.084349           2.084349            2.084349          0          0          10      0.908741      0.330309      0.689045    0.330581      0.033058     0.050018 -1.952286e-01         1
  HOOD                2.088115           2.088115            2.088115          0          0           8      0.908529      0.213100      0.462496    0.221489      0.027686     0.053935 -1.950634e-01         1
  AMKR                2.074496           2.074496            2.074496          0          0           8      0.905728      0.209345      0.440972    0.212568      0.026571     0.040938 -1.501621e-01         1
   ZIM                2.047905           2.047905            2.047905          0          0           7      0.897568      0.158062      0.342310    0.167151      0.023879     0.046297 -1.596113e-01         1
  ENTA                2.041262           2.041262            2.041262          0          0           7      0.896101      0.284519      0.624898    0.306133      0.043733     0.052185 -1.723696e-01         1
  CRWV                3.246417           3.246417            3.246417          0          0           4      0.892337      0.519373      0.802113    0.247076      0.061769     0.089219 -2.470764e-01         1
   QSI                2.042904           2.042904            2.042904          0          0           6      0.888479      0.270909      0.712210    0.348626      0.058104     0.084042 -2.829646e-01         1
  TSLA                1.965347           1.965347            1.965347          0          0           8      0.880412      0.194293      0.455033    0.231528      0.028941     0.045348 -1.910541e-01         1
  ADPT                1.942364           1.942364            1.942364          0          0           8      0.874392      0.177233      0.422193    0.217360      0.027170     0.051731 -2.078933e-01         1
   HUT                1.958382           1.958382            1.958382          0          0           6      0.868237      0.243753      0.691737    0.353219      0.058870     0.075929 -2.549559e-01         1
  IREN                1.872301           1.872301            1.872301          0          0           7      0.852029      0.318783      0.940675    0.502417      0.071774     0.081890 -4.256299e-01         1
  PGEN                1.862235           1.862235            1.862235          0          0           7      0.848967      0.208676      0.668231    0.358833      0.051262     0.078831 -2.278984e-01         1
  GSAT                1.827527           1.827527            1.827527          0          0           7      0.838010      0.224465      0.638044    0.349130      0.049876     0.052423 -1.747288e-01         1
  NBIS                1.817255           1.817255            1.817255          0          0           7      0.834649      0.169322      0.483224    0.265909      0.037987     0.072417 -2.353998e-01         1
  IMMX                1.769467           1.769467            1.769467          0          0           7      0.818286      0.176173      0.488255    0.275933      0.039419     0.060723 -1.997195e-01         1
  MLYS                1.764539           1.764539            1.764539          0          0           7      0.816530      0.113206      0.370079    0.209731      0.029962     0.054787 -1.539863e-01         1
  ANVS                1.700650           1.700650            1.700650          0          0           7      0.792619      0.152724      0.534009    0.314003      0.044858     0.081295 -1.592413e-01         1
    HL                1.674413           1.674413            1.674413          0          0           8      0.784544      0.162713      0.486071    0.290293      0.036287     0.052227 -1.902089e-01         1
  AREC                1.637407           1.637407            1.637407          0          0           9      0.769925      0.211227      0.918615    0.561018      0.062335     0.117569 -2.770733e-01         1
  OKLO                1.623978           1.623978            1.623978          0          0           7      0.761141      0.178708      0.868583    0.534849      0.076407     0.096089 -4.000832e-01         1
   HIT                1.628767           1.628767            1.628767          0          0           6      0.755625      0.112510      0.610850    0.375038      0.062506     0.091546 -2.848439e-01         1
   KOD                1.562670           1.562670            1.562670          0          0           7      0.733892      0.164110      0.786554    0.503340      0.071906     0.085432 -1.464381e-01         1
   APP                1.499911           1.499911            1.499911          0          0           5      0.674442      0.093286      0.430271    0.286864      0.057373     0.054356 -1.479794e-01         1
  GRAL                1.366387           1.366387            1.366387          0          0           7      0.636543      0.048567      0.295068    0.215948      0.030850     0.082115 -1.444745e-01         1
  CLSK                1.296755           1.296755            1.296755          0          0           7      0.599579      0.047044      0.494140    0.381059      0.054437     0.082110 -2.540652e-01         1
  PACB                1.280062           1.280062            1.280062          0          0           7      0.590635      0.042461      0.437038    0.341420      0.048774     0.078680 -1.777613e-01         1
    MP                1.230958           1.230958            1.230958          0          0           7      0.564271      0.007057      0.473562    0.384711      0.054959     0.067575 -3.357537e-01         1
  CSIQ                1.220829           1.220829            1.220829          0          0           7      0.558836     -0.000591      0.559547    0.458334      0.065476     0.067482 -2.735925e-01         1
   CEG                1.184302           1.184302            1.184302          0          0           6      0.532558      0.014005      0.180721    0.152597      0.025433     0.042083 -8.489587e-02         1
  OWLT                1.165611           1.165611            1.165611          0          0           9      0.532055      0.006220      0.485418    0.416449      0.046272     0.046162 -2.294359e-01         1
  VERA                1.217376           1.217376            1.217376          0          0           5      0.529444      0.004484      0.439831    0.361294      0.072259     0.054737 -2.550155e-01         1
   VRT                1.174592           1.174592            1.174592          0          0           6      0.527408      0.012549      0.235319    0.200341      0.033390     0.048220 -1.524453e-01         1
   INO                1.077325           1.077325            1.077325          0          0           7      0.483287     -0.057449      0.473845    0.439834      0.062833     0.064068 -3.537881e-01         1
  CBRL                1.067469           1.067469            1.067469          0          0           8      0.480271     -0.024897      0.408037    0.382247      0.047781     0.048921 -3.339462e-01         1
   NFE                0.989725           0.989725            0.989725          0          0           6      0.433543     -0.207173      0.644570    0.651261      0.108544     0.142923 -3.351400e-01         1
  NAIL                0.966859           0.966859            0.966859          0          0          10      0.431677     -0.103296      0.564312    0.583655      0.058365     0.066378 -3.364553e-01         1
  NAMS                0.920944           0.920944            0.920944          0          0           7      0.407880     -0.074518      0.345489    0.375146      0.053592     0.049491 -1.770449e-01         1
   RGC                0.850746           0.850746            0.850746          0          0           5      0.353456     -0.266440      0.529218    0.622063      0.124413     0.072466 -5.408238e-01         1
  NVTS                0.835328           0.835328            0.835328          0          0           5      0.347278     -0.164942      0.418811    0.501373      0.100275     0.098052 -3.531597e-01         1
  JACK                0.731505           0.731505            0.731505          0          0           8      0.333825     -0.162900      0.337374    0.461205      0.057651     0.059531 -2.695658e-01         1
  ANET                0.730625           0.730625            0.730625          0          0           7      0.331742     -0.120261      0.252506    0.345602      0.049372     0.039425 -2.799789e-01         1
  ALAB                0.596956           0.596956            0.596956          0          0           6      0.284367     -0.256189      0.318498    0.533536      0.088923     0.072680 -4.430092e-01         1
   WGS                0.513361           0.513361            0.513361          0          0           7      0.268276     -0.210677      0.207787    0.404757      0.057822     0.053459 -2.265416e-01         1
  WEAV                0.435703           0.435703            0.435703          0          0           7      0.251152     -0.188610      0.144828    0.332400      0.047486     0.042795 -2.499528e-01         1
  PLTR                0.339808           0.339808            0.339808          0          0           6      0.228363     -0.205283      0.107569    0.316559      0.052760     0.046765 -2.052829e-01         1
  TQQQ                0.212460           0.212460            0.212460          0          0           8      0.216493     -0.290064      0.086000    0.404783      0.050598     0.040253 -3.030874e-01         1
Saved per-ticker CSV to: output\per_ticker_20260119_0844.csv
real 498.479    user 0.060      sys 0.045       pcpu 0.02

Using GMM low-vol data:
=== Objective Degeneracy Diagnostics ===
Min GL/trade target/k:           0.003000 / 12.000
Median GL per trade (eligible):  0.015570
GLPT mult (best eval):           1.000000
Median Profit Factor (diag):      2.472220   (cap=4.0)
Avg Profit Factor (diag cap):     2.647496   (cap=4.0)
Median Profit Factor (eff):       2.472220   (loss_floor=0.001)
Avg Profit Factor (eff):          30.958518   (loss_floor=0.001)
Avg PF (eff, excl zero-loss):     10.770153
Zero-loss tickers (gl==0,gp>0):   3/147 = 0.020
PF capped (eff>cap):              55/147 = 0.374
Median PF (raw, finite only):     2.472220
Trimmed Mean PF (raw, 5% trim):   5.625769
Avg Trades / Ticker:              7.925
Mean Ticker Score (eligible):     0.695301
Coverage (eligible/total):        86/147 = 0.585
Penalty enabled:                  enabled (soft)
Soft penalty center/k:            0.01 / 10.0
Tail-protection ret floor/k:      -0.15 / 2.0
Max-trades penalty:               max_trades=100, k=0.1
PF-floor penalty:                 pf_floor=1.0, k=5.0
cooldown_bars:                    1 (opt=False) [loss-only]
time_stop_bars:                   34 (0=disabled) (opt=True)
num_neg (return<=0):              19
commission_rate_per_side:         0.000600
PF ROC baseline/k:                1.150 / 2.500
Trades ROC baseline/k:            8.000 / 0.500
weight_pf:                        0.550
score_power:                      1.000
min_trades gate:                  8
loss_floor (scoring):             0.001
threshold_mode:                   fixed
Score (objective):                0.071357

=== BEST PARAMS ===
fp: 5
reg_ratio: 2.2847696736334853
sp: 22
ts: 34
vl: 5
vs: 5
best_score (OPTUNA objective): 0.07135706358110457
fill_mode (final report): next_open

=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===
ticker profit_factor_raw_print  profit_factor_eff  profit_factor_diag  pf_capped  zero_loss  num_trades  ticker_score  total_return  gross_profit  gross_loss  gl_per_trade  atr_pct_med         maxdd  eligible
   WWD                5.355108           5.355108            4.000000          1          0          12      0.945916      0.321222      0.360117    0.067247      0.005604     0.022466 -5.490106e-02         1
   CAT               10.380036          10.380036            4.000000          1          0          10      0.878534      0.588230      0.537481    0.051780      0.005178     0.023837 -3.689517e-02         1
   MLI                9.877509           9.877509            4.000000          1          0          10      0.878534      0.274499      0.281980    0.028548      0.002855     0.019137 -2.187862e-02         1
    TD                4.701393           4.701393            4.000000          1          0          10      0.878534      0.218860      0.259991    0.055301      0.005530     0.013360 -4.410638e-02         1
  LRCX                4.275946           4.275946            4.000000          1          0           9      0.829664      0.592934      0.670827    0.156884      0.017432     0.036365 -9.945598e-02         1
  DLTR                4.256422           4.256422            4.000000          1          0           9      0.829664      0.455992      0.524380    0.123197      0.013689     0.031629 -6.554569e-02         1
 GOOGL                9.047416           9.047416            4.000000          1          0           9      0.829664      0.445512      0.431818    0.047728      0.005303     0.025997 -3.211109e-02         1
   TSM                7.998250           7.998250            4.000000          1          0           9      0.829664      0.430824      0.439850    0.054993      0.006110     0.028402 -4.957685e-02         1
    GS               16.635218          16.635218            4.000000          1          0           9      0.829664      0.424387      0.395744    0.023790      0.002643     0.021935 -1.361763e-02         1
   FDX                7.197296           7.197296            4.000000          1          0           9      0.829664      0.423618      0.434781    0.060409      0.006712     0.023253 -6.040897e-02         1
  BLOK                9.488187           9.488187            4.000000          1          0           9      0.829664      0.401577      0.399642    0.042120      0.004680     0.030596 -2.588181e-02         1
   XBI              138.166248         138.166248            4.000000          1          0           9      0.829664      0.338868      0.302384    0.002189      0.000243     0.021481 -2.188549e-03         1
   LIT                7.481416           7.481416            4.000000          1          0           9      0.829664      0.337394      0.348818    0.046625      0.005181     0.018498 -4.637943e-02         1
   TEL                9.164649           9.164649            4.000000          1          0           9      0.829664      0.322503      0.331285    0.036148      0.004016     0.020640 -2.012704e-02         1
   XSD                4.298343           4.298343            4.000000          1          0           9      0.829664      0.293011      0.360177    0.083794      0.009310     0.019390 -5.181769e-02         1
  BBIO                4.036350           4.036350            4.000000          1          0           9      0.829664      0.262222      0.329314    0.081587      0.009065     0.036031 -7.995415e-02         1
    SU               13.591587          13.591587            4.000000          1          0           9      0.829664      0.257597      0.256162    0.018847      0.002094     0.020194 -1.332969e-02         1
  SIMO                3.831425           3.831425            3.831425          0          0           9      0.829433      0.490724      0.604579    0.157795      0.017533     0.036148 -1.523162e-01         1
   TAN                3.634340           3.634340            3.634340          0          0           9      0.829005      0.252758      0.334991    0.092174      0.010242     0.029632 -6.383499e-02         1
  ARKK                3.115471           3.115471            3.115471          0          0           9      0.826096      0.315456      0.449534    0.144291      0.016032     0.031170 -7.309073e-02         1
  ASML                2.316365           2.316365            2.316365          0          0           9      0.801852      0.182319      0.328999    0.142033      0.015781     0.025928 -1.358438e-01         1
  EPAM                1.849933           1.849933            1.849933          0          0          10      0.797539      0.174842      0.424982    0.229728      0.022973     0.029929 -1.930621e-01         1
   MAR                2.251569           2.251569            2.251569          0          0           9      0.797181      0.107410      0.207913    0.092342      0.010260     0.020799 -6.302956e-02         1
  META                2.237911           2.237911            2.237911          0          0           9      0.796108      0.161137      0.309336    0.138225      0.015358     0.024979 -1.214012e-01         1
   NEM                1.649444           1.649444            1.649444          0          0          11      0.795291      0.182121      0.528028    0.320125      0.029102     0.030759 -2.229448e-01         1
   APH                2.205946           2.205946            2.205946          0          0           9      0.793468      0.192252      0.369017    0.167283      0.018587     0.027277 -1.010673e-01         1
   IYW                1.769709           1.769709            1.769709          0          0          10      0.782621      0.090983      0.228757    0.129262      0.012926     0.014999 -8.176404e-02         1
   STX                5.259398           5.259398            4.000000          1          0           8      0.774558      0.593049      0.631037    0.119983      0.014998     0.032791 -1.199828e-01         1
   XME              139.767149         139.767149            4.000000          1          0           8      0.774558      0.518079      0.436888    0.003126      0.000391     0.024713 -3.125825e-03         1
  REMX                9.134800           9.134800            4.000000          1          0           8      0.774558      0.487460      0.468226    0.051257      0.006407     0.026646 -5.125744e-02         1
   IMO               16.664969          16.664969            4.000000          1          0           8      0.774558      0.406647      0.374964    0.022500      0.002813     0.021454 -2.250015e-02         1
   CMI                5.570142           5.570142            4.000000          1          0           8      0.774558      0.351992      0.390942    0.070185      0.008773     0.021778 -6.006958e-02         1
   VET                5.530648           5.530648            4.000000          1          0           8      0.774558      0.348478      0.395568    0.071523      0.008940     0.036064 -5.534611e-02         1
  bidu                4.009132           4.009132            4.000000          1          0           8      0.774558      0.325910      0.403117    0.100550      0.012569     0.036231 -4.853581e-02         1
    GE                4.170627           4.170627            4.000000          1          0           8      0.774558      0.273761      0.341850    0.081966      0.010246     0.023105 -8.069653e-02         1
   CRS                3.991279           3.991279            3.991279          0          0           8      0.774548      0.277495      0.367043    0.091961      0.011495     0.034695 -9.043327e-02         1
    MD                3.038450           3.038450            3.038450          0          0           8      0.770145      0.318959      0.458969    0.151054      0.018882     0.031332 -7.605315e-02         1
  FTXL                2.892210           2.892210            2.892210          0          0           8      0.768030      0.122655      0.189670    0.065580      0.008197     0.012906 -6.416577e-02         1
  ILMN                1.697822           1.697822            1.697822          0          0          10      0.767496      0.157347      0.448517    0.264172      0.026417     0.035727 -2.194258e-01         1
  LGND                1.924427           1.924427            1.924427          0          0           9      0.760762      0.147669      0.324962    0.168862      0.018762     0.026045 -1.605918e-01         1
     M                1.910876           1.910876            1.910876          0          0           9      0.758683      0.228512      0.526047    0.275291      0.030588     0.035668 -1.894771e-01         1
  MSFT                2.527046           2.527046            2.527046          0          0           8      0.757956      0.100971      0.175970    0.069635      0.008704     0.017993 -6.856541e-02         1
  NVDA                2.512237           2.512237            2.512237          0          0           8      0.757333      0.261884      0.444405    0.176896      0.022112     0.031073 -1.682595e-01         1
  NDAQ                2.465330           2.465330            2.465330          0          0           8      0.755214      0.120899      0.208440    0.084549      0.010569     0.016870 -8.277222e-02         1
   RCL                2.398824           2.398824            2.398824          0          0           8      0.751786      0.297441      0.519101    0.216398      0.027050     0.031688 -9.156438e-02         1
  AAPL                1.578121           1.578121            1.578121          0          0          10      0.738536      0.057975      0.172561    0.109346      0.010935     0.019993 -6.020057e-02         1
  HTHT                1.737964           1.737964            1.737964          0          0           9      0.727281      0.079760      0.204026    0.117394      0.013044     0.025064 -6.616479e-02         1
  ARKW                1.730639           1.730639            1.730639          0          0           9      0.725741      0.118926      0.322386    0.186282      0.020698     0.023216 -1.183589e-01         1
   ASC                2.034794           2.034794            2.034794          0          0           8      0.720726      0.164427      0.336214    0.165233      0.020654     0.030290 -1.556905e-01         1
   EIS                2.027538           2.027538            2.027538          0          0           8      0.719832      0.089281      0.178479    0.088028      0.011003     0.010707 -6.812573e-02         1
  SNOW                2.019301           2.019301            2.019301          0          0           8      0.718802      0.193096      0.421505    0.208738      0.026092     0.034934 -1.284923e-01         1
   AXP                1.958923           1.958923            1.958923          0          0           8      0.710716      0.125930      0.278119    0.141975      0.017747     0.020916 -8.170463e-02         1
   RTX                1.936623           1.936623            1.936623          0          0           8      0.707483      0.141813      0.304999    0.157490      0.019686     0.018756 -9.447888e-02         1
  XPEL                1.916164           1.916164            1.916164          0          0           8      0.704394      0.154845      0.364815    0.190388      0.023799     0.032176 -1.778761e-01         1
  BABA                1.597762           1.597762            1.597762          0          0           9      0.694740      0.089589      0.273202    0.170991      0.018999     0.034743 -1.065779e-01         1
    GM                1.809929           1.809929            1.809929          0          0           8      0.686377      0.101674      0.242084    0.133753      0.016719     0.024737 -6.439221e-02         1
  DELL                1.532823           1.532823            1.532823          0          0           9      0.677499      0.106270      0.393675    0.256830      0.028537     0.035961 -1.811910e-01         1
   SPY                1.250585           1.250585            1.250585          0          0          11      0.677304      0.019544      0.130525    0.104371      0.009488     0.010809 -8.240632e-02         1
   UHS                1.708289           1.708289            1.708289          0          0           8      0.665827      0.083341      0.219735    0.128629      0.016079     0.027131 -7.805589e-02         1
   FDN                1.663728           1.663728            1.663728          0          0           8      0.655751      0.063441      0.178891    0.107524      0.013441     0.014715 -5.162754e-02         1
   AEM                1.143701           1.143701            1.143701          0          0          11      0.640743      0.016702      0.257117    0.224811      0.020437     0.029322 -1.056629e-01         1
  ARKF                1.315671           1.315671            1.315671          0          0           9      0.611256      0.045122      0.270709    0.205757      0.022862     0.022670 -1.610820e-01         1
  PTCT                1.281538           1.281538            1.281538          0          0           9      0.599920      0.045322      0.457794    0.357222      0.039691     0.034649 -1.948140e-01         1
   AIQ                1.094305           1.094305            1.094305          0          0          10      0.584862      0.003576      0.198902    0.181761      0.018176     0.016188 -1.284124e-01         1
  PTGX                1.229407           1.229407            1.229407          0          0           9      0.582313      0.032500      0.278100    0.226207      0.025134     0.033810 -8.802440e-02         1
   MRK                1.224170           1.224170            1.224170          0          0           9      0.580530      0.021565      0.163766    0.133777      0.014864     0.024025 -1.074101e-01         1
  JAZZ                1.357149           1.357149            1.357149          0          0           8      0.569658      0.038643      0.175811    0.129544      0.016193     0.025302 -7.766732e-02         1
   ICE                1.334944           1.334944            1.334944          0          0           8      0.562466      0.031342      0.158178    0.118490      0.014811     0.014857 -1.134542e-01         1
  DDOG                1.325047           1.325047            1.325047          0          0           8      0.559230      0.063822      0.438590    0.331000      0.041375     0.035849 -2.019241e-01         1
  ACWX                0.763419           0.763419            0.763419          0          0          12      0.547932     -0.031646      0.092583    0.121274      0.010106     0.008607 -1.118367e-01         1
   XLK                0.959182           0.959182            0.959182          0          0          10      0.539599     -0.030546      0.229606    0.239377      0.023938     0.017899 -1.969737e-01         1
   LLY                1.255052           1.255052            1.255052          0          0           8      0.535906      0.029096      0.218928    0.174437      0.021805     0.026255 -8.604340e-02         1
  ABBV                1.181777           1.181777            1.181777          0          0           8      0.510918      0.015519      0.144346    0.122143      0.015268     0.020749 -9.610498e-02         1
   TJX                1.019484           1.019484            1.019484          0          0           9      0.510636     -0.002879      0.120991    0.118679      0.013187     0.015331 -6.417460e-02         1
  ANDE                1.115274           1.115274            1.115274          0          0           8      0.488071      0.007600      0.228964    0.205299      0.025662     0.025776 -1.018333e-01         1
  SCHD                0.899024           0.899024            0.899024          0          0           9      0.471557     -0.010027      0.074162    0.082492      0.009166     0.010465 -6.906918e-02         1
  GLSI                0.873844           0.873844            0.873844          0          0           9      0.463777     -0.074338      0.319483    0.365607      0.040623     0.030716 -2.637760e-01         1
   IQV                0.870639           0.870639            0.870639          0          0           9      0.462798     -0.052044      0.231228    0.265584      0.029509     0.025939 -2.027876e-01         1
   MDT                1.037984           1.037984            1.037984          0          0           8      0.461744     -0.001211      0.146426    0.141068      0.017633     0.017951 -6.018141e-02         1
  ALNY                0.996769           0.996769            0.996769          0          0           8      0.447962     -0.029335      0.279699    0.280605      0.035076     0.032827 -2.286325e-01         1
   NKE                0.746012           0.746012            0.746012          0          0           9      0.426949     -0.100583      0.237132    0.317866      0.035318     0.028514 -2.804483e-01         1
   TMO                0.327916           0.327916            0.327916          0          0          10      0.391416     -0.124956      0.062647    0.191047      0.019105     0.024180 -1.756127e-01         1
  REGN                0.796813           0.796813            0.796813          0          0           8      0.385910     -0.079753      0.226719    0.284532      0.035566     0.028301 -2.150944e-01         1
  ADBE                0.768601           0.768601            0.768601          0          0           8      0.378000     -0.081888      0.218045    0.283691      0.035461     0.024628 -2.533932e-01         1
   CNQ                0.434207           0.434207            0.434207          0          0           9      0.358832     -0.129324      0.099579    0.229336      0.025482     0.022349 -1.293235e-01         1
  NFLX                0.427687           0.427687            0.427687          0          0           9      0.357738     -0.257039      0.191366    0.447444      0.049716     0.025512 -3.596973e-01         1
  UTHR               21.218370          21.218370            4.000000          1          0           7      0.000000      0.806234      0.702934    0.033129      0.004733     0.026636 -3.312858e-02         0
   BFH               20.501627          20.501627            4.000000          1          0           6      0.000000      0.690359      0.589278    0.028743      0.004790     0.028687 -2.874297e-02         0
   NGL              447.345116         447.345116            4.000000          1          0           5      0.000000      0.649886      0.536814    0.001200      0.000240     0.036820 -1.200000e-03         0
  INCY       1000000000.000000        1000.000000            4.000000          1          1           6      0.000000      0.626959      0.509920    0.000000      0.000000     0.027744 -1.000089e-12         0
  AMAT               68.578167          68.578167            4.000000          1          0           7      0.000000      0.605058      0.508206    0.007411      0.001059     0.033983 -7.410607e-03         0
   AMG       1000000000.000000        1000.000000            4.000000          1          1           7      0.000000      0.585386      0.481391    0.000000      0.000000     0.020182 -1.000089e-12         0
   FTI                8.718246           8.718246            4.000000          1          0           7      0.000000      0.538609      0.517663    0.059377      0.008482     0.027995 -5.937694e-02         0
   GLW       1000000000.000000        1000.000000            4.000000          1          1           6      0.000000      0.537761      0.453233    0.000000      0.000000     0.027689 -1.000089e-12         0
    SN                6.201479           6.201479            4.000000          1          0           6      0.000000      0.528766      0.569244    0.091792      0.015299     0.036652 -7.150251e-02         0
    CX               51.471875          51.471875            4.000000          1          0           7      0.000000      0.507090      0.442288    0.008593      0.001228     0.025523 -8.592804e-03         0
  BOOT               52.962901          52.962901            4.000000          1          0           6      0.000000      0.494853      0.443788    0.008379      0.001397     0.034682 -8.373633e-03         0
  RVMD                7.095141           7.095141            4.000000          1          0           6      0.000000      0.408274      0.431487    0.060814      0.010136     0.035405 -5.997746e-02         0
   EGO                6.462997           6.462997            4.000000          1          0           6      0.000000      0.407749      0.434847    0.067283      0.011214     0.033785 -6.728260e-02         0
  PALL               35.515330          35.515330            4.000000          1          0           7      0.000000      0.372179      0.338867    0.009541      0.001363     0.022359 -9.541436e-03         0
  AXSM                5.217349           5.217349            4.000000          1          0           7      0.000000      0.363943      0.411820    0.078933      0.011276     0.030134 -7.893275e-02         0
   ATI                4.075556           4.075556            4.000000          1          0           7      0.000000      0.346063      0.446405    0.109532      0.015647     0.032751 -8.544970e-02         0
  DINO                4.535486           4.535486            4.000000          1          0           7      0.000000      0.315110      0.381353    0.084082      0.012012     0.029272 -6.814357e-02         0
   IBM               31.548515          31.548515            4.000000          1          0           6      0.000000      0.311134      0.289988    0.009192      0.001532     0.023677 -9.191810e-03         0
  BLTE                5.329766           5.329766            4.000000          1          0           5      0.000000      0.277467      0.319055    0.059863      0.011973     0.025869 -5.986290e-02         0
  NRIM               73.547143          73.547143            4.000000          1          0           5      0.000000      0.274375      0.256609    0.003489      0.000698     0.021229 -3.489044e-03         0
   AMX               11.473803          11.473803            4.000000          1          0           7      0.000000      0.245024      0.249008    0.021702      0.003100     0.019234 -1.133685e-02         0
  NVGS                5.585338           5.585338            4.000000          1          0           6      0.000000      0.242156      0.280530    0.050226      0.008371     0.022238 -4.721227e-02         0
   AER               14.292295          14.292295            4.000000          1          0           5      0.000000      0.226831      0.231104    0.016170      0.003234     0.015993 -1.614189e-02         0
   JPM                6.051263           6.051263            4.000000          1          0           7      0.000000      0.217868      0.249637    0.041254      0.005893     0.019529 -3.795881e-02         0
     F                4.001105           4.001105            4.000000          1          0           7      0.000000      0.186933      0.240802    0.060184      0.008598     0.023725 -3.672251e-02         0
   NSC               13.653571          13.653571            4.000000          1          0           7      0.000000      0.181969      0.186211    0.013638      0.001948     0.016857 -8.144927e-03         0
   HCA                4.278343           4.278343            4.000000          1          0           7      0.000000      0.169025      0.216607    0.050629      0.007233     0.023531 -3.959893e-02         0
    RY                4.330321           4.330321            4.000000          1          0           7      0.000000      0.113857      0.144846    0.033449      0.004778     0.011056 -2.076828e-02         0
   GLD                4.199885           4.199885            4.000000          1          0           7      0.000000      0.086577      0.111623    0.026578      0.003797     0.011142 -2.216002e-02         0
  PBBK                7.470899           7.470899            4.000000          1          0           6      0.000000      0.057658      0.065744    0.008800      0.001467     0.003621 -4.880261e-03         0
  RGLD                3.863434           3.863434            3.863434          0          0           7      0.000000      0.174133      0.231649    0.059959      0.008566     0.025302 -5.056964e-02         0
  TEVA                3.537816           3.537816            3.537816          0          0           7      0.000000      0.356106      0.470511    0.132995      0.018999     0.030418 -8.284350e-02         0
   LUV                3.468964           3.468964            3.468964          0          0           7      0.000000      0.311266      0.429516    0.123817      0.017688     0.030822 -1.022129e-01         0
   BAC                3.446431           3.446431            3.446431          0          0           7      0.000000      0.204967      0.282709    0.082030      0.011719     0.019041 -5.819722e-02         0
   NVT                3.349265           3.349265            3.349265          0          0           7      0.000000      0.311426      0.440537    0.131532      0.018790     0.030306 -1.295751e-01         0
  NTRA                3.185685           3.185685            3.185685          0          0           7      0.000000      0.192002      0.283440    0.088973      0.012710     0.032750 -8.216167e-02         0
   LVS                3.147782           3.147782            3.147782          0          0           7      0.000000      0.282889      0.410703    0.130474      0.018639     0.027077 -1.262406e-01         0
    CM                3.083197           3.083197            3.083197          0          0           7      0.000000      0.079572      0.116884    0.037910      0.005416     0.011969 -3.777691e-02         0
    TK                3.037386           3.037386            3.037386          0          0           6      0.000000      0.252983      0.370505    0.121982      0.020330     0.027728 -8.135119e-02         0
  SBGI                2.941176           2.941176            2.941176          0          0           6      0.000000      0.216944      0.327431    0.111327      0.018554     0.035444 -6.454563e-02         0
  TARS                2.472220           2.472220            2.472220          0          0           7      0.000000      0.246028      0.421408    0.170457      0.024351     0.036186 -1.609884e-01         0
  ENVA                2.471193           2.471193            2.471193          0          0           7      0.000000      0.217626      0.367704    0.148796      0.021257     0.026596 -7.959335e-02         0
   BLD                2.243612           2.243612            2.243612          0          0           7      0.000000      0.144377      0.268780    0.119798      0.017114     0.028682 -7.733826e-02         0
  CTSH                2.233265           2.233265            2.233265          0          0           7      0.000000      0.147897      0.279139    0.124992      0.017856     0.020151 -1.203917e-01         0
   PBR                2.112903           2.112903            2.112903          0          0           7      0.000000      0.134781      0.262342    0.124162      0.017737     0.023801 -1.191914e-01         0
  ARGX                1.946568           1.946568            1.946568          0          0           6      0.000000      0.115065      0.251296    0.129097      0.021516     0.023503 -1.249333e-01         0
  ROST                1.932671           1.932671            1.932671          0          0           7      0.000000      0.070984      0.155566    0.080493      0.011499     0.018708 -7.462674e-02         0
   JNJ                1.904830           1.904830            1.904830          0          0           7      0.000000      0.069741      0.152216    0.079911      0.011416     0.015171 -6.688236e-02         0
  PLAB                1.859247           1.859247            1.859247          0          0           7      0.000000      0.194220      0.468153    0.251797      0.035971     0.033141 -1.540468e-01         0
  IMAX                1.752146           1.752146            1.752146          0          0           7      0.000000      0.142843      0.383305    0.218763      0.031252     0.029711 -1.594630e-01         0
  EQNR                1.644193           1.644193            1.644193          0          0           6      0.000000      0.061679      0.174318    0.106020      0.017670     0.022284 -8.745361e-02         0
  TCMD                1.600902           1.600902            1.600902          0          0           7      0.000000      0.064889      0.193797    0.121055      0.017294     0.034699 -8.955917e-02         0
   LMT                1.576329           1.576329            1.576329          0          0           6      0.000000      0.062831      0.191364    0.121399      0.020233     0.019674 -6.829672e-02         0
  RDVT                1.494136           1.494136            1.494136          0          0           7      0.000000      0.066592      0.254898    0.170599      0.024371     0.027743 -1.298167e-01         0
   NYT                1.427099           1.427099            1.427099          0          0           6      0.000000      0.040749      0.161987    0.113508      0.018918     0.017242 -1.103018e-01         0
  JKHY                1.204997           1.204997            1.204997          0          0           7      0.000000      0.017025      0.137797    0.114355      0.016336     0.017553 -1.098589e-01         0
   VLO                1.057219           1.057219            1.057219          0          0           7      0.000000     -0.010269      0.243898    0.230698      0.032957     0.028578 -1.128701e-01         0
  FROG                0.770556           0.770556            0.770556          0          0           7      0.000000     -0.112977      0.280280    0.363737      0.051962     0.036821 -2.457931e-01         0
   VTR                0.715175           0.715175            0.715175          0          0           7      0.000000     -0.053910      0.121805    0.170315      0.024331     0.016171 -1.119574e-01         0
  NTRB                0.624272           0.624272            0.624272          0          0           7      0.000000     -0.149051      0.216309    0.346498      0.049500     0.035439 -2.246345e-01         0
   NOW                0.573890           0.573890            0.573890          0          0           7      0.000000     -0.229640      0.282442    0.492154      0.070308     0.028194 -3.540767e-01         0
Saved per-ticker CSV to: output\per_ticker_20260119_1106.csv
real 327.070    user 0.062      sys 0.062       pcpu 0.03

=== Objective Degeneracy Diagnostics ===
Min GL/trade target/k:           0.003000 / 12.000
Median GL per trade (eligible):  0.014252
GLPT mult (best eval):           1.000000
Median Profit Factor (diag):      2.589050   (cap=4.0)
Avg Profit Factor (diag cap):     2.549757   (cap=4.0)
Median Profit Factor (eff):       2.589050   (loss_floor=0.001)
Avg Profit Factor (eff):          35.749393   (loss_floor=0.001)
Avg PF (eff, excl zero-loss):     4.237936
Zero-loss tickers (gl==0,gp>0):   5/158 = 0.032
PF capped (eff>cap):              46/158 = 0.291
Median PF (raw, finite only):     2.589050
Trimmed Mean PF (raw, 5% trim):   3.896242
Avg Trades / Ticker:              7.570
Mean Ticker Score (eligible):     0.813823
Coverage (eligible/total):        158/158 = 1.000
Penalty enabled:                  enabled (soft)
Soft penalty center/k:            0.01 / 10.0
Tail-protection ret floor/k:      -0.15 / 2.0
Max-trades penalty:               max_trades=100, k=0.1
PF-floor penalty:                 pf_floor=1.0, k=5.0
cooldown_bars:                    1 (opt=False) [loss-only]
time_stop_bars:                   8 (0=disabled) (opt=True)
num_neg (return<=0):              21
commission_rate_per_side:         0.000600
PF ROC baseline/k:                1.150 / 2.500
Trades ROC baseline/k:            3.000 / 1.200
weight_pf:                        0.700
score_power:                      1.500
min_trades gate:                  1
loss_floor (scoring):             0.001
threshold_mode:                   fixed
Score (objective):                0.719409

=== BEST PARAMS ===
fp: 10
reg_ratio: 2.0653036036152104
sp: 22
ts: 8
vl: 5
vs: 6
best_score (OPTUNA objective): 0.7194091523901832
fill_mode (final report): next_open

=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===
ticker profit_factor_raw_print  profit_factor_eff  profit_factor_diag  pf_capped  zero_loss  num_trades  ticker_score  total_return  gross_profit  gross_loss  gl_per_trade  atr_pct_med         maxdd  eligible
   CAT                5.055045           5.055045            4.000000          1          0          10      0.999055      0.423986      0.467424    0.092467      0.009247     0.023837 -5.821754e-02         1
   WWD               10.461847          10.461847            4.000000          1          0          10      0.999055      0.320535      0.322158    0.030794      0.003079     0.022466 -1.847573e-02         1
  LRCX               14.927819          14.927819            4.000000          1          0           9      0.998820      0.755722      0.646859    0.043332      0.004815     0.036365 -3.401586e-02         1
   AMG               17.601804          17.601804            4.000000          1          0           9      0.998820      0.439377      0.401024    0.022783      0.002531     0.020182 -2.269877e-02         1
 GOOGL                7.660949           7.660949            4.000000          1          0           9      0.998820      0.400627      0.405164    0.052887      0.005876     0.025997 -2.909538e-02         1
   LIT                8.515930           8.515930            4.000000          1          0           9      0.998820      0.365745      0.366312    0.043015      0.004779     0.018498 -4.301488e-02         1
   FDX               15.510852          15.510852            4.000000          1          0           9      0.998820      0.320187      0.306510    0.019761      0.002196     0.023253 -1.976103e-02         1
  DLTR                6.057340           6.057340            4.000000          1          0           9      0.998820      0.286455      0.316873    0.052312      0.005812     0.031629 -4.519958e-02         1
   MLI               17.896170          17.896170            4.000000          1          0           9      0.998820      0.283203      0.270131    0.015094      0.001677     0.019137 -1.509436e-02         1
    SU               13.838258          13.838258            4.000000          1          0           9      0.998820      0.244671      0.244844    0.017693      0.001966     0.020194 -1.239913e-02         1
  BLOK                3.772307           3.772307            3.772307          0          0           9      0.998174      0.295289      0.381091    0.101023      0.011225     0.030596 -9.435296e-02         1
  CIEN                4.805898           4.805898            4.000000          1          0           8      0.998044      0.542573      0.612889    0.127529      0.015941     0.038962 -9.371303e-02         1
   XME               41.549236          41.549236            4.000000          1          0           8      0.998044      0.433592      0.383361    0.009227      0.001153     0.024713 -9.226673e-03         1
   TSM               14.984679          14.984679            4.000000          1          0           8      0.998044      0.409675      0.392156    0.026170      0.003271     0.028402 -2.571639e-02         1
  ARKK                6.294619           6.294619            4.000000          1          0           8      0.998044      0.381709      0.418365    0.066464      0.008308     0.031170 -4.349523e-02         1
   STX                4.368614           4.368614            4.000000          1          0           8      0.998044      0.368934      0.450322    0.103081      0.012885     0.032791 -9.931409e-02         1
  REMX                5.056001           5.056001            4.000000          1          0           8      0.998044      0.336358      0.383220    0.075795      0.009474     0.026646 -5.203634e-02         1
  NTRA                4.780534           4.780534            4.000000          1          0           8      0.998044      0.265164      0.319739    0.066883      0.008360     0.032750 -5.365549e-02         1
   TEL                4.862211           4.862211            4.000000          1          0           8      0.998044      0.260274      0.311645    0.064095      0.008012     0.020640 -4.990282e-02         1
   AXP                4.793338           4.793338            4.000000          1          0           8      0.998044      0.246285      0.297746    0.062117      0.007765     0.020916 -3.464428e-02         1
  HTHT                4.349782           4.349782            4.000000          1          0           8      0.998044      0.193571      0.241236    0.055459      0.006932     0.025064 -4.563596e-02         1
   EIS                5.931137           5.931137            4.000000          1          0           8      0.998044      0.165837      0.189476    0.031946      0.003993     0.010707 -1.871213e-02         1
  ASML                3.889637           3.889637            3.889637          0          0           8      0.997776      0.232263      0.302103    0.077669      0.009709     0.025928 -4.138749e-02         1
  PAAS                3.801720           3.801720            3.801720          0          0           8      0.997503      0.382639      0.487376    0.128199      0.016025     0.038832 -1.242714e-01         1
   CMI                3.713559           3.713559            3.713559          0          0           8      0.997162      0.275565      0.359139    0.096710      0.012089     0.021778 -9.425931e-02         1
   XSD                3.540054           3.540054            3.540054          0          0           9      0.997004      0.288342      0.382653    0.108093      0.012010     0.019390 -6.111288e-02         1
   AMX                3.660972           3.660972            3.660972          0          0           8      0.996920      0.170945      0.227723    0.062203      0.007775     0.019234 -4.470220e-02         1
  SNOW                3.641271           3.641271            3.641271          0          0           8      0.996821      0.291644      0.395769    0.108690      0.013586     0.034934 -9.431984e-02         1
  RVMD               11.600455          11.600455            4.000000          1          0           7      0.995486      0.646506      0.585029    0.050432      0.007205     0.035405 -4.996595e-02         1
   FTI               11.439755          11.439755            4.000000          1          0           7      0.995486      0.510305      0.477755    0.041763      0.005966     0.027995 -4.176269e-02         1
  UTHR                6.418652           6.418652            4.000000          1          0           7      0.995486      0.496910      0.554798    0.086435      0.012348     0.026636 -5.498323e-02         1
  TCMD       1000000000.000000        1000.000000            4.000000          1          1           7      0.995486      0.410049      0.359229    0.000000      0.000000     0.034699 -1.000089e-12         1
    GS               13.456952          13.456952            4.000000          1          0           7      0.995486      0.393883      0.378518    0.028128      0.004018     0.021935 -2.673515e-02         1
  NVGS                7.282295           7.282295            4.000000          1          0           7      0.995486      0.331571      0.348546    0.047862      0.006837     0.022238 -4.786213e-02         1
  PALL               17.755612          17.755612            4.000000          1          0           7      0.995486      0.302838      0.291367    0.016410      0.002344     0.022359 -1.636549e-02         1
  DINO                4.683272           4.683272            4.000000          1          0           7      0.995486      0.277435      0.336547    0.071861      0.010266     0.029272 -3.845962e-02         1
   XBI               43.538136          43.538136            4.000000          1          0           7      0.995486      0.238364      0.225428    0.005178      0.000740     0.021481 -2.779280e-03         1
    CM                6.380382           6.380382            4.000000          1          0           7      0.995486      0.194649      0.216666    0.033958      0.004851     0.011969 -3.395820e-02         1
   JNJ               11.469123          11.469123            4.000000          1          0           7      0.995486      0.156630      0.163389    0.014246      0.002035     0.015171 -1.166703e-02         1
   JPM                4.309949           4.309949            4.000000          1          0           7      0.995486      0.113965      0.150661    0.034957      0.004994     0.019529 -3.455365e-02         1
   BLD                4.999649           4.999649            4.000000          1          0           7      0.995486      0.093876      0.115134    0.023028      0.003290     0.028682 -1.235110e-02         1
   VET                3.662142           3.662142            3.662142          0          0           7      0.994369      0.360193      0.465678    0.127160      0.018166     0.036064 -9.061878e-02         1
    CX                3.640039           3.640039            3.640039          0          0           7      0.994258      0.271967      0.362181    0.099499      0.014214     0.025523 -9.745409e-02         1
  AXSM                3.500278           3.500278            3.500278          0          0           7      0.993395      0.292059      0.389641    0.111317      0.015902     0.030134 -1.086900e-01         1
   FDN                3.217301           3.217301            3.217301          0          0           8      0.992950      0.118542      0.176884    0.054979      0.006872     0.014715 -4.602066e-02         1
  BBIO                3.211727           3.211727            3.211727          0          0           8      0.992868      0.162147      0.233839    0.072808      0.009101     0.036031 -2.544866e-02         1
  SIMO                3.158770           3.158770            3.158770          0          0           8      0.992022      0.325500      0.471448    0.149251      0.018656     0.036148 -1.355694e-01         1
   LUV                3.255018           3.255018            3.255018          0          0           7      0.990927      0.293290      0.421348    0.129446      0.018492     0.030822 -1.264353e-01         1
  RGLD                3.162446           3.162446            3.162446          0          0           7      0.989532      0.149580      0.219780    0.069497      0.009928     0.025302 -4.567534e-02         1
   IMO                3.014487           3.014487            3.014487          0          0           8      0.989073      0.140695      0.208093    0.069031      0.008629     0.021454 -5.123914e-02         1
    GE                3.010163           3.010163            3.010163          0          0           8      0.988967      0.209365      0.306687    0.101884      0.012735     0.023105 -6.402925e-02         1
    RY                3.094418           3.094418            3.094418          0          0           7      0.988283      0.096203      0.140882    0.045528      0.006504     0.011056 -2.465150e-02         1
   BAC                3.079823           3.079823            3.079823          0          0           7      0.987987      0.190742      0.279426    0.090728      0.012961     0.019041 -6.588628e-02         1
  INCY       1000000000.000000        1000.000000            4.000000          1          1           6      0.987214      0.755012      0.590161    0.000000      0.000000     0.027744 -1.000089e-12         1
    SN                8.297466           8.297466            4.000000          1          0           6      0.987214      0.599784      0.593856    0.071571      0.011928     0.036652 -7.157077e-02         1
  AMAT       1000000000.000000        1000.000000            4.000000          1          1           6      0.987214      0.597637      0.496055    0.000000      0.000000     0.033983 -1.000089e-12         1
   GLW                9.354545           9.354545            4.000000          1          0           6      0.987214      0.326621      0.333935    0.035698      0.005950     0.027689 -2.591679e-02         1
  RYTM                6.715324           6.715324            4.000000          1          0           6      0.987214      0.322364      0.352116    0.052435      0.008739     0.037623 -2.764888e-02         1
  NDAQ       1000000000.000000        1000.000000            4.000000          1          1           6      0.987214      0.203327      0.191608    0.000000      0.000000     0.016870 -1.000089e-12         1
  ARGX                5.703490           5.703490            4.000000          1          0           6      0.987214      0.180159      0.209806    0.036786      0.006131     0.023503 -3.664512e-02         1
  META                2.867018           2.867018            2.867018          0          0          10      0.985773      0.265009      0.403882    0.140872      0.014087     0.024979 -1.180513e-01         1
   LVS                2.866382           2.866382            2.866382          0          0           7      0.982198      0.254742      0.391782    0.136682      0.019526     0.027077 -1.322381e-01         1
  XPEL                3.213354           3.213354            3.213354          0          0           6      0.982081      0.295580      0.429316    0.133604      0.022267     0.032176 -1.288619e-01         1
  TARS                3.191854           3.191854            3.191854          0          0           6      0.981754      0.218147      0.315026    0.098697      0.016449     0.036186 -9.572092e-02         1
   ATI                3.167154           3.167154            3.167154          0          0           6      0.981355      0.283330      0.414754    0.130955      0.021826     0.032751 -6.505219e-02         1
  BOOT                3.127831           3.127831            3.127831          0          0           6      0.980669      0.244634      0.375412    0.120023      0.020004     0.034682 -8.835927e-02         1
  TEVA                2.819224           2.819224            2.819224          0          0           7      0.980461      0.305716      0.465711    0.165191      0.023599     0.030418 -9.860654e-02         1
  NVDA                2.747901           2.747901            2.747901          0          0           8      0.979972      0.282209      0.444331    0.161698      0.020212     0.031073 -1.547256e-01         1
    TD                2.601009           2.601009            2.601009          0          0           8      0.971833      0.128456      0.205233    0.078905      0.009863     0.013360 -5.680799e-02         1
   CVI                2.634260           2.634260            2.634260          0          0           7      0.971394      0.229054      0.382933    0.145366      0.020767     0.039383 -9.529744e-02         1
  SBGI                2.758105           2.758105            2.758105          0          0           6      0.969673      0.149643      0.242192    0.087811      0.014635     0.035444 -4.981326e-02         1
   NVT                2.589117           2.589117            2.589117          0          0           7      0.968508      0.238159      0.409833    0.158291      0.022613     0.030306 -1.529216e-01         1
  ENVA                2.573289           2.573289            2.573289          0          0           7      0.967421      0.232734      0.377558    0.146722      0.020960     0.026596 -7.997924e-02         1
   XLK                2.516034           2.516034            2.516034          0          0           9      0.966436      0.155276      0.264495    0.105124      0.011680     0.017899 -6.822192e-02         1
  PTGX                2.482548           2.482548            2.482548          0          0           9      0.963652      0.147207      0.251210    0.101191      0.011243     0.033810 -4.846846e-02         1
   BFH       1000000000.000000        1000.000000            4.000000          1          1           5      0.961973      0.741631      0.589994    0.000000      0.000000     0.028687 -1.000089e-12         1
   NGL                6.475314           6.475314            4.000000          1          0           5      0.961973      0.304650      0.337296    0.052089      0.010418     0.036820 -5.202837e-02         1
   AER               19.783044          19.783044            4.000000          1          0           5      0.961973      0.221994      0.220560    0.011149      0.002230     0.015993 -1.111791e-02         1
   IBM                5.656657           5.656657            4.000000          1          0           5      0.961973      0.131347      0.155445    0.027480      0.005496     0.023677 -1.726133e-02         1
   NYT                3.815535           3.815535            3.815535          0          0           5      0.961486      0.135205      0.180609    0.047335      0.009467     0.017242 -4.733516e-02         1
  AVDL                2.588983           2.588983            2.588983          0          0           6      0.960303      0.261228      0.445088    0.171916      0.028653     0.039183 -1.128155e-01         1
   APH                2.391782           2.391782            2.391782          0          0           9      0.954940      0.223072      0.392250    0.163999      0.018222     0.027277 -7.540394e-02         1
  IMAX                2.512939           2.512939            2.512939          0          0           6      0.954710      0.210473      0.359228    0.142951      0.023825     0.029711 -1.080290e-01         1
    MD                2.502569           2.502569            2.502569          0          0           6      0.953868      0.218885      0.372806    0.148969      0.024828     0.031332 -7.437899e-02         1
  FTXL                2.347375           2.347375            2.347375          0          0           8      0.949218      0.088711      0.161017    0.068594      0.008574     0.012906 -6.705498e-02         1
   SQM                2.316866           2.316866            2.316866          0          0           8      0.945517      0.175544      0.327444    0.141331      0.017666     0.037471 -9.568045e-02         1
   EGO                2.686880           2.686880            2.686880          0          0           5      0.941120      0.161325      0.261598    0.097361      0.019472     0.033785 -6.564333e-02         1
    TK                2.239559           2.239559            2.239559          0          0           7      0.932454      0.134385      0.258454    0.115404      0.016486     0.027728 -1.095191e-01         1
   CRS                2.215333           2.215333            2.215333          0          0           7      0.928770      0.153667      0.315809    0.142556      0.020365     0.034695 -1.363605e-01         1
  NRIM                4.920489           4.920489            4.000000          1          0           4      0.896852      0.104579      0.130303    0.026482      0.006620     0.021229 -2.363077e-02         1
   RTX                2.013980           2.013980            2.013980          0          0           8      0.892342      0.136785      0.278833    0.138449      0.017306     0.018756 -1.317730e-01         1
  DELL                2.006155           2.006155            2.006155          0          0           8      0.890495      0.202680      0.441635    0.220140      0.027517     0.035961 -1.525618e-01         1
   RCL                1.975240           1.975240            1.975240          0          0           7      0.880472      0.182438      0.415927    0.210570      0.030081     0.031688 -1.331168e-01         1
  RDVT                1.959982           1.959982            1.959982          0          0           7      0.876578      0.122388      0.274653    0.140130      0.020019     0.027743 -1.148085e-01         1
  ILMN                1.941402           1.941402            1.941402          0          0           9      0.874878      0.203331      0.461180    0.237550      0.026394     0.035727 -2.196173e-01         1
   LMT                1.949712           1.949712            1.949712          0          0           7      0.873895      0.129466      0.279682    0.143448      0.020493     0.019674 -1.191852e-01         1
  PLAB                1.932197           1.932197            1.932197          0          0           7      0.869204      0.200284      0.457229    0.236637      0.033805     0.033141 -1.833455e-01         1
  LGND                1.917364           1.917364            1.917364          0          0           8      0.867557      0.123772      0.276551    0.144235      0.018029     0.026045 -1.320180e-01         1
   NEM                1.883061           1.883061            1.883061          0          0          11      0.858706      0.151707      0.356176    0.189147      0.017195     0.030759 -1.301334e-01         1
  MSFT                1.896034           1.896034            1.896034          0          0           6      0.851174      0.065839      0.156996    0.082802      0.013800     0.017993 -7.452241e-02         1
     M                1.846441           1.846441            1.846441          0          0           9      0.847214      0.172890      0.430031    0.232897      0.025877     0.035668 -1.761155e-01         1
   VLO                1.803552           1.803552            1.803552          0          0           7      0.830078      0.123306      0.306594    0.169994      0.024285     0.028526 -9.583982e-02         1
   TAN                1.762250           1.762250            1.762250          0          0           9      0.818831      0.050182      0.121242    0.068799      0.007644     0.029632 -5.469594e-02         1
   HCA                1.695153           1.695153            1.695153          0          0           7      0.790462      0.062462      0.163974    0.096731      0.013819     0.023531 -7.587505e-02         1
  EQNR                1.693533           1.693533            1.693533          0          0           6      0.782168      0.034486      0.091799    0.054206      0.009034     0.022284 -4.329396e-02         1
  BLTE                1.754289           1.754289            1.754289          0          0           5      0.781541      0.089510      0.230599    0.131449      0.026290     0.025869 -7.705771e-02         1
   PBR                1.650343           1.650343            1.650343          0          0           7      0.772300      0.083272      0.233779    0.141655      0.020236     0.023801 -1.350811e-01         1
   MAR                1.605360           1.605360            1.605360          0          0           9      0.756093      0.065930      0.210166    0.130915      0.014546     0.020799 -9.899721e-02         1
  BABA                1.589909           1.589909            1.589909          0          0          10      0.749459      0.121996      0.390493    0.245607      0.024561     0.034743 -1.918343e-01         1
  JKHY                1.664120           1.664120            1.664120          0          0           5      0.747156      0.050366      0.138494    0.083224      0.016645     0.017553 -8.127738e-02         1
  CTSH                1.567965           1.567965            1.567965          0          0           6      0.728836      0.061160      0.200679    0.127987      0.021331     0.020151 -1.230468e-01         1
   TJX                1.541446           1.541446            1.541446          0          0           9      0.727056      0.044120      0.135076    0.087629      0.009737     0.015331 -5.463028e-02         1
   ICE                1.494227           1.494227            1.494227          0          0           8      0.703773      0.048869      0.170828    0.114325      0.014291     0.014857 -1.100550e-01         1
  FROG                1.459574           1.459574            1.459574          0          0           7      0.684393      0.084801      0.378220    0.259130      0.037019     0.036821 -1.355505e-01         1
   AIQ                1.430094           1.430094            1.430094          0          0          10      0.672648      0.047149      0.191481    0.133894      0.013389     0.016526 -9.090933e-02         1
  DDOG                1.423089           1.423089            1.423089          0          0           7      0.665947      0.058856      0.299923    0.210755      0.030108     0.035849 -1.989031e-01         1
   UHS                1.420074           1.420074            1.420074          0          0           7      0.664405      0.053901      0.217840    0.153400      0.021914     0.027131 -1.031336e-01         1
  REGN                1.415053           1.415053            1.415053          0          0           8      0.664062      0.063764      0.288200    0.203667      0.025458     0.028301 -1.742699e-01         1
  PTCT                1.418426           1.418426            1.418426          0          0           7      0.663560      0.079489      0.414039    0.291900      0.041700     0.034649 -1.862853e-01         1
  AVGO                1.402726           1.402726            1.402726          0          0           9      0.658382      0.057710      0.307145    0.218963      0.024329     0.038222 -2.001836e-01         1
  MDGL                1.459186           1.459186            1.459186          0          0           5      0.654672      0.062257      0.305558    0.209403      0.041881     0.038293 -1.558808e-01         1
     F                1.396422           1.396422            1.396422          0          0           7      0.652218      0.044466      0.194569    0.139334      0.019905     0.023725 -1.196909e-01         1
  ARKW                1.390472           1.390472            1.390472          0          0           9      0.652021      0.065221      0.316107    0.227338      0.025260     0.023216 -1.606287e-01         1
   LLY                1.360059           1.360059            1.360059          0          0           8      0.635416      0.057747      0.310045    0.227964      0.028496     0.026255 -1.293628e-01         1
   MRK                1.352695           1.352695            1.352695          0          0           9      0.632197      0.038931      0.182994    0.135281      0.015031     0.024025 -1.068967e-01         1
   IYW                1.351466           1.351466            1.351466          0          0           9      0.631547      0.045558      0.230726    0.170723      0.018969     0.014999 -1.256317e-01         1
   ASC                1.336314           1.336314            1.336314          0          0           8      0.622852      0.056227      0.305290    0.228457      0.028557     0.030290 -1.904354e-01         1
  EPAM                1.325423           1.325423            1.325423          0          0           7      0.614880      0.045981      0.287310    0.216768      0.030967     0.029929 -2.010927e-01         1
  APPN                1.263205           1.263205            1.263205          0          0           6      0.574678      0.030236      0.291847    0.231037      0.038506     0.037424 -1.453676e-01         1
  ARKF                1.217935           1.217935            1.217935          0          0           9      0.560033      0.027646      0.271114    0.222601      0.024733     0.022670 -1.750602e-01         1
   IQV                1.185100           1.185100            1.185100          0          0           9      0.542441      0.020555      0.245826    0.207431      0.023048     0.025939 -1.842855e-01         1
  ALNY                1.165267           1.165267            1.165267          0          0           8      0.531243      0.012335      0.268005    0.229995      0.028749     0.032827 -1.884450e-01         1
  JAZZ                1.178628           1.178628            1.178628          0          0           6      0.529547      0.015145      0.164591    0.139646      0.023274     0.025302 -6.944197e-02         1
   SPY                1.134600           1.134600            1.134600          0          0          10      0.515840      0.006647      0.112276    0.098956      0.009896     0.010809 -9.466656e-02         1
   GLD                1.103866           1.103866            1.103866          0          0          10      0.499774      0.009086      0.156863    0.142103      0.014210     0.012373 -1.351010e-01         1
   MDT                1.102612           1.102612            1.102612          0          0           8      0.498322      0.005806      0.104687    0.094945      0.011868     0.017951 -6.303993e-02         1
  NTRB                1.115908           1.115908            1.115908          0          0           6      0.496616      0.002860      0.216678    0.194172      0.032362     0.035439 -1.165013e-01         1
  AAPL                1.010734           1.010734            1.010734          0          0          10      0.452740     -0.004889      0.142866    0.141349      0.014135     0.019993 -8.631306e-02         1
  ANDE                1.000131           1.000131            1.000131          0          0           8      0.446804     -0.006722      0.130783    0.130766      0.016346     0.025776 -9.943476e-02         1
  ADBE                0.982570           0.982570            0.982570          0          0           8      0.438359     -0.020356      0.221564    0.225495      0.028187     0.024628 -2.068577e-01         1
  ABBV                0.969239           0.969239            0.969239          0          0           8      0.432034     -0.010500      0.133699    0.137942      0.017243     0.020749 -9.414512e-02         1
    GM                0.897743           0.897743            0.897743          0          0           6      0.391512     -0.016564      0.096617    0.107622      0.017937     0.024737 -8.647518e-02         1
  GLSI                0.854110           0.854110            0.854110          0          0           9      0.381400     -0.050516      0.204851    0.239842      0.026649     0.030716 -1.752857e-01         1
  PBBK                0.800902           0.800902            0.800902          0          0           7      0.357623     -0.006544      0.024444    0.030520      0.004360     0.004293 -2.688939e-02         1
  ROST                0.761130           0.761130            0.761130          0          0           6      0.336859     -0.029607      0.081485    0.107058      0.017843     0.018708 -9.905848e-02         1
  SCHD                0.726523           0.726523            0.726523          0          0           8      0.332086     -0.022694      0.056698    0.078041      0.009755     0.010465 -7.598498e-02         1
  bidu                0.710639           0.710639            0.710639          0          0           8      0.326633     -0.063581      0.138344    0.194676      0.024335     0.036231 -1.295703e-01         1
  ACWX                0.689374           0.689374            0.689374          0          0          11      0.320286     -0.043850      0.091795    0.133157      0.012105     0.008607 -1.241523e-01         1
   NKE                0.632727           0.632727            0.632727          0          0           9      0.302369     -0.140271      0.218852    0.345887      0.038432     0.028514 -3.011751e-01         1
   AEM                0.628869           0.628869            0.628869          0          0          10      0.301382     -0.119720      0.187802    0.298635      0.029864     0.029322 -1.620902e-01         1
   NOW                0.587891           0.587891            0.587891          0          0           7      0.287335     -0.181620      0.235846    0.401173      0.057310     0.028194 -3.144792e-01         1
  RGEN                0.448379           0.448379            0.448379          0          0           9      0.255883     -0.182683      0.146502    0.326738      0.036304     0.038063 -1.826834e-01         1
   VTR                0.398032           0.398032            0.398032          0          0           7      0.243770     -0.097051      0.064432    0.161876      0.023125     0.016171 -1.517082e-01         1
   TMO                0.384760           0.384760            0.384760          0          0           9      0.243386     -0.120932      0.076140    0.197890      0.021988     0.024180 -1.838171e-01         1
  NFLX                0.361659           0.361659            0.361659          0          0           9      0.239244     -0.266966      0.154505    0.427213      0.047468     0.025512 -3.650665e-01         1
   CNQ                0.280152           0.280152            0.280152          0          0           7      0.224140     -0.182419      0.074358    0.265420      0.037917     0.022349 -1.824187e-01         1
  AZTA                0.184915           0.184915            0.184915          0          0           9      0.213594     -0.236101      0.057923    0.313240      0.034804     0.038094 -2.598020e-01         1
   NSC                0.200108           0.200108            0.200108          0          0           7      0.213427     -0.054307      0.013753    0.068727      0.009818     0.016857 -5.430659e-02         1
Saved per-ticker CSV to: output\per_ticker_20260119_0855.csv

Added stand-alone python script to split the data.
Split data with 2 options (gmm or vol/kmeans)
==============================================
Volatility split completed
Method:              gmm
ATR length:          25
Approx threshold:    0.037675
Low-vol count:       158  -> low_vol.txt
High-vol count:      123 -> high_vol.txt
Summary CSV:         split_summary.csv
==============================================

==============================================
Volatility split completed
Method:              kmeans
ATR length:          25
Approx threshold:    0.036928
Low-vol count:       147  -> low_vol.txt
High-vol count:      134 -> high_vol.txt
Summary CSV:         split_summary.csv
==============================================
 
Why split data by volatility works:
Your output clearly shows volatility clustering
PF-capped names dominate high ATR%
Low-vol names want longer VIDYA / slower thresholds
High-vol names want shorter VIDYA / tighter exits
This split reduces objective conflict without needing more data

1/18/2026
Vidya_RSI_v0.1 final with Alpaca_options.csv (281 files) data. 
Results show mixed ticker populations (capped “monsters” + normal mid-PF + low-PF/high-loss names). 
GLPT isn’t binding (median 0.019 > 0.003) and zero-loss is controlled (3.9%), so the main issue is one parameter set can’t fit all regimes—it gets pulled in different directions. 
Since you can’t time-split, the best split is by volatility using your existing atr_pct_med (e.g., top 40% vs bottom 60%), then run two Optuna studies to get two specialized parameter sets.

Using all high-vol + low-vol (281) data files:
=== Objective Degeneracy Diagnostics ===
Min GL/trade target/k:           0.003000 / 12.000
Median GL per trade (eligible):  0.019264
GLPT mult (best eval):           1.000000
Median Profit Factor (diag):      2.420492   (cap=4.0)
Avg Profit Factor (diag cap):     2.447986   (cap=4.0)
Median Profit Factor (eff):       2.420492   (loss_floor=0.001)
Avg Profit Factor (eff):          43.714513   (loss_floor=0.001)
Avg PF (eff, excl zero-loss):     4.754734
Zero-loss tickers (gl==0,gp>0):   11/281 = 0.039
PF capped (eff>cap):              84/281 = 0.299
Median PF (raw, finite only):     2.420492
Trimmed Mean PF (raw, 5% trim):   4.331633
Avg Trades / Ticker:              6.577
Mean Ticker Score (eligible):     0.771350
Coverage (eligible/total):        281/281 = 1.000
Penalty enabled:                  enabled (soft)
Soft penalty center/k:            0.01 / 10.0
Tail-protection ret floor/k:      -0.15 / 2.0
Max-trades penalty:               max_trades=100, k=0.1
PF-floor penalty:                 pf_floor=1.0, k=5.0
cooldown_bars:                    1 (opt=False) [loss-only]
time_stop_bars:                   11 (0=disabled) (opt=True)
num_neg (return<=0):              58
commission_rate_per_side:         0.000600
PF ROC baseline/k:                1.150 / 2.500
Trades ROC baseline/k:            3.000 / 1.200
weight_pf:                        0.700
score_power:                      1.500
min_trades gate:                  1
loss_floor (scoring):             0.001
threshold_mode:                   fixed
Score (objective):                0.670891

=== BEST PARAMS ===
fp: 7
reg_ratio: 2.6053158181842617
sp: 21
ts: 11
vl: 5
vs: 9
best_score (OPTUNA objective): 0.6708913406074537
fill_mode (final report): next_open

=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===
ticker profit_factor_raw_print  profit_factor_eff  profit_factor_diag  pf_capped  zero_loss  num_trades  ticker_score  total_return  gross_profit  gross_loss  gl_per_trade  atr_pct_med         maxdd  eligible
   MLI       1000000000.000000        1000.000000            4.000000          1          1          10      0.999055      0.420951      0.362629    0.000000      0.000000     0.019137 -1.000089e-12         1
    BE                3.940405           3.940405            3.940405          0          0          10      0.998919      1.457360      1.472327    0.373649      0.037365     0.073465 -2.852849e-01         1
   FDX               11.412938          11.412938            4.000000          1          0           9      0.998820      0.428146      0.405604    0.035539      0.003949     0.023253 -3.523053e-02         1
   CAT                7.730979           7.730979            4.000000          1          0           9      0.998820      0.315580      0.326882    0.042282      0.004698     0.023837 -3.452506e-02         1
  DLTR                6.403348           6.403348            4.000000          1          0           9      0.998820      0.289746      0.316825    0.049478      0.005498     0.031629 -4.947805e-02         1
  APLD                5.207618           5.207618            4.000000          1          0           8      0.998044      1.635883      1.439956    0.276509      0.034564     0.095765 -1.753704e-01         1
  AREC                5.131105           5.131105            4.000000          1          0           8      0.998044      1.441874      1.395780    0.272023      0.034003     0.117569 -1.318122e-01         1
  LABU               46.620440          46.620440            4.000000          1          0           8      0.998044      0.902023      0.706767    0.015160      0.001895     0.057585 -1.183624e-02         1
    AG                5.410135           5.410135            4.000000          1          0           8      0.998044      0.734756      0.759989    0.140475      0.017559     0.051359 -1.091944e-01         1
  LRCX               16.076751          16.076751            4.000000          1          0           8      0.998044      0.628118      0.546864    0.034016      0.004252     0.036365 -3.401586e-02         1
   WWD       1000000000.000000        1000.000000            4.000000          1          1           8      0.998044      0.431357      0.373409    0.000000      0.000000     0.022466 -1.000089e-12         1
   AMG               13.107993          13.107993            4.000000          1          0           8      0.998044      0.360955      0.348046    0.026552      0.003319     0.020182 -2.645027e-02         1
  BABA                4.309670           4.309670            4.000000          1          0           8      0.998044      0.309028      0.378197    0.087755      0.010969     0.034743 -5.366913e-02         1
   TAN                4.089459           4.089459            4.000000          1          0           8      0.998044      0.296820      0.369393    0.090328      0.011291     0.029632 -4.901140e-02         1
   LIT                6.776950           6.776950            4.000000          1          0           8      0.998044      0.271498      0.291510    0.043015      0.005377     0.018498 -4.301488e-02         1
  BBIO                5.074433           5.074433            4.000000          1          0           8      0.998044      0.255192      0.298788    0.058881      0.007360     0.036031 -3.341841e-02         1
   XBI                8.889807           8.889807            4.000000          1          0           8      0.998044      0.203106      0.215690    0.024263      0.003033     0.021481 -2.243489e-02         1
  HTHT                4.118972           4.118972            4.000000          1          0           8      0.998044      0.196447      0.248630    0.060362      0.007545     0.025064 -3.892713e-02         1
  SIMO                3.858993           3.858993            3.858993          0          0           8      0.997687      0.354887      0.462676    0.119896      0.014987     0.036148 -1.052472e-01         1
   NEM                3.559118           3.559118            3.559118          0          0           9      0.997128      0.355015      0.463160    0.130133      0.014459     0.030759 -1.301334e-01         1
    GE                3.647595           3.647595            3.647595          0          0           8      0.996854      0.219674      0.292787    0.080269      0.010034     0.023105 -6.423355e-02         1
   MDT                3.572389           3.572389            3.572389          0          0           8      0.996434      0.127234      0.172969    0.048418      0.006052     0.017951 -2.522910e-02         1
  AXTI               53.374957          53.374957            4.000000          1          0           7      0.995486      2.086631      1.317824    0.024690      0.003527     0.080127 -2.468993e-02         1
  GLUE       1000000000.000000        1000.000000            4.000000          1          1           7      0.995486      1.372145      0.947482    0.000000      0.000000     0.061259 -1.000089e-12         1
  VICR               11.022844          11.022844            4.000000          1          0           7      0.995486      1.115524      0.921352    0.083586      0.011941     0.046601 -8.358570e-02         1
     U               12.789760          12.789760            4.000000          1          0           7      0.995486      0.982623      0.811671    0.063463      0.009066     0.056672 -5.560313e-02         1
  SPHR                6.881589           6.881589            4.000000          1          0           7      0.995486      0.754378      0.732323    0.106418      0.015203     0.041048 -1.064178e-01         1
    MU                6.648066           6.648066            4.000000          1          0           7      0.995486      0.719278      0.694333    0.104441      0.014920     0.043189 -1.044413e-01         1
  UTHR               16.787959          16.787959            4.000000          1          0           7      0.995486      0.700700      0.636336    0.037904      0.005415     0.026636 -1.978707e-02         1
   ERO                4.851538           4.851538            4.000000          1          0           7      0.995486      0.536654      0.597437    0.123144      0.017592     0.041433 -1.150020e-01         1
    NG                5.027033           5.027033            4.000000          1          0           7      0.995486      0.474349      0.540334    0.107486      0.015355     0.048211 -1.074857e-01         1
   FTI               10.746776          10.746776            4.000000          1          0           7      0.995486      0.473232      0.448814    0.041763      0.005966     0.027995 -4.176269e-02         1
    SA                6.072916           6.072916            4.000000          1          0           7      0.995486      0.424453      0.459456    0.075657      0.010808     0.042689 -6.876757e-02         1
   IMO               18.916747          18.916747            4.000000          1          0           7      0.995486      0.322422      0.305774    0.016164      0.002309     0.021454 -1.616418e-02         1
    CX                4.375571           4.375571            4.000000          1          0           7      0.995486      0.319039      0.388199    0.088720      0.012674     0.025523 -8.761902e-02         1
  REMX                6.533499           6.533499            4.000000          1          0           7      0.995486      0.308492      0.334323    0.051171      0.007310     0.026646 -2.569549e-02         1
  DINO                6.125598           6.125598            4.000000          1          0           7      0.995486      0.299756      0.336547    0.054941      0.007849     0.029272 -3.975268e-02         1
 GOOGL                6.490853           6.490853            4.000000          1          0           7      0.995486      0.284089      0.309798    0.047728      0.006818     0.025997 -3.211109e-02         1
    SU                7.729037           7.729037            4.000000          1          0           7      0.995486      0.257673      0.273686    0.035410      0.005059     0.020194 -2.208036e-02         1
   TSM                5.831682           5.831682            4.000000          1          0           7      0.995486      0.153698      0.179533    0.030786      0.004398     0.028402 -3.064932e-02         1
    CM                4.448981           4.448981            4.000000          1          0           7      0.995486      0.128147      0.159957    0.035954      0.005136     0.011969 -3.594471e-02         1
   TEL                4.950460           4.950460            4.000000          1          0           7      0.995486      0.121828      0.149138    0.030126      0.004304     0.020640 -1.419255e-02         1
   JNJ                4.088513           4.088513            4.000000          1          0           7      0.995486      0.090985      0.118500    0.028984      0.004141     0.015171 -2.634076e-02         1
    MD                3.986985           3.986985            3.986985          0          0           7      0.995458      0.417700      0.511583    0.128313      0.018330     0.031332 -7.437899e-02         1
  AXSM                3.741614           3.741614            3.741614          0          0           7      0.994722      0.323983      0.416506    0.111317      0.015902     0.030134 -1.086900e-01         1
  PTGX                3.124258           3.124258            3.124258          0          0           9      0.992183      0.161151      0.235703    0.075443      0.008383     0.033810 -3.764347e-02         1
  ANAB                3.088546           3.088546            3.088546          0          0           8      0.990716      0.514302      0.702390    0.227418      0.028427     0.053579 -1.150560e-01         1
   STX                3.209370           3.209370            3.209370          0          0           7      0.990279      0.154726      0.223976    0.069788      0.009970     0.032791 -6.823480e-02         1
   AMD                3.149193           3.149193            3.149193          0          0           7      0.989304      0.326746      0.465691    0.147876      0.021125     0.041689 -1.408628e-01         1
   RIG                2.960797           2.960797            2.960797          0          0           9      0.988454      0.446677      0.656486    0.221726      0.024636     0.050018 -1.108197e-01         1
  PASG               15.469008          15.469008            4.000000          1          0           6      0.987214      3.123621      2.771400    0.179158      0.029860     0.057175 -1.522573e-01         1
  BITF                5.554929           5.554929            4.000000          1          0           6      0.987214      1.696400      1.497907    0.269654      0.044942     0.079748 -2.696536e-01         1
  COGT               23.072392          23.072392            4.000000          1          0           6      0.987214      1.367142      1.008745    0.043721      0.007287     0.058280 -4.372088e-02         1
  NEGG                5.622022           5.622022            4.000000          1          0           6      0.987214      1.236755      1.367248    0.243195      0.040533     0.090473 -2.304997e-01         1
   DBI                4.852370           4.852370            4.000000          1          0           6      0.987214      1.067358      1.110024    0.228759      0.038127     0.068329 -2.001619e-01         1
    PL               13.782047          13.782047            4.000000          1          0           6      0.987214      1.054957      0.871330    0.063222      0.010537     0.067776 -6.322209e-02         1
  TERN                7.683267           7.683267            4.000000          1          0           6      0.987214      0.915310      0.849283    0.110537      0.018423     0.061826 -1.105367e-01         1
  CSIQ                4.191723           4.191723            4.000000          1          0           6      0.987214      0.834443      0.976912    0.233057      0.038843     0.067482 -1.695792e-01         1
  COHR               10.647941          10.647941            4.000000          1          0           6      0.987214      0.752725      0.674586    0.063354      0.010559     0.055058 -6.335365e-02         1
  RGTI                7.247839           7.247839            4.000000          1          0           6      0.987214      0.749049      0.736781    0.101655      0.016943     0.093788 -1.016552e-01         1
  FUTU               15.544578          15.544578            4.000000          1          0           6      0.987214      0.599931      0.549088    0.035323      0.005887     0.048077 -3.532347e-02         1
  PAAS               46.389276          46.389276            4.000000          1          0           6      0.987214      0.573672      0.492759    0.010622      0.001770     0.038832 -1.062227e-02         1
  CIEN                7.791751           7.791751            4.000000          1          0           6      0.987214      0.497589      0.497570    0.063859      0.010643     0.038962 -6.385861e-02         1
  RVMD                6.136167           6.136167            4.000000          1          0           6      0.987214      0.400107      0.432895    0.070548      0.011758     0.035405 -6.931278e-02         1
   LVS                6.157435           6.157435            4.000000          1          0           6      0.987214      0.345549      0.383574    0.062294      0.010382     0.027077 -6.181391e-02         1
  IONQ                4.937449           4.937449            4.000000          1          0           6      0.987214      0.307653      0.372439    0.075431      0.012572     0.084476 -7.543144e-02         1
  SNOW                8.673231           8.673231            4.000000          1          0           6      0.987214      0.219876      0.238162    0.027459      0.004577     0.034934 -1.634127e-02         1
  NTRA                4.167255           4.167255            4.000000          1          0           6      0.987214      0.216117      0.278721    0.066883      0.011147     0.032750 -5.365549e-02         1
  ABUS               18.074252          18.074252            4.000000          1          0           6      0.987214      0.209223      0.208572    0.011540      0.001923     0.042311 -1.153973e-02         1
    GS                7.665242           7.665242            4.000000          1          0           6      0.987214      0.197168      0.215608    0.028128      0.004688     0.021935 -2.673515e-02         1
  NDAQ               13.372280          13.372280            4.000000          1          0           6      0.987214      0.193833      0.198827    0.014869      0.002478     0.016870 -1.486858e-02         1
  TEVA                3.864351           3.864351            3.864351          0          0           6      0.986875      0.391997      0.493526    0.127713      0.021285     0.030418 -8.205714e-02         1
  BOOT                3.722013           3.722013            3.722013          0          0           6      0.986372      0.273755      0.378357    0.101654      0.016942     0.034682 -7.160436e-02         1
  RKLB                3.676557           3.676557            3.676557          0          0           6      0.986170      0.615881      0.796310    0.216591      0.036099     0.070309 -2.017691e-01         1
   APP                3.612140           3.612140            3.612140          0          0           6      0.985841      0.618949      0.790406    0.218819      0.036470     0.054356 -1.388848e-01         1
  MLYS                2.885406           2.885406            2.885406          0          0           8      0.985391      0.336416      0.541990    0.187838      0.023480     0.054787 -1.324298e-01         1
  ENVA                2.930585           2.930585            2.930585          0          0           7      0.984265      0.254923      0.377558    0.128834      0.018405     0.026596 -7.997924e-02         1
   PBR                3.333298           3.333298            3.333298          0          0           6      0.983621      0.177006      0.249029    0.074709      0.012452     0.023801 -7.435678e-02         1
  TSLA                3.315334           3.315334            3.315334          0          0           6      0.983419      0.231025      0.326188    0.098388      0.016398     0.045348 -9.673832e-02         1
   LLY                2.900069           2.900069            2.900069          0          0           7      0.983323      0.138915      0.213307    0.073552      0.010507     0.026255 -4.385759e-02         1
  DBRG                3.190836           3.190836            3.190836          0          0           6      0.981738      0.146434      0.235340    0.073755      0.012293     0.046211 -7.065455e-02         1
   LUV                2.832250           2.832250            2.832250          0          0           7      0.980961      0.266025      0.417947    0.147567      0.021081     0.030822 -1.282481e-01         1
   CLS                3.125717           3.125717            3.125717          0          0           6      0.980630      0.336292      0.492408    0.157534      0.026256     0.055521 -1.157716e-01         1
  EOSE                3.117661           3.117661            3.117661          0          0           6      0.980480      0.519492      0.758519    0.243297      0.040550     0.084444 -1.972057e-01         1
   VET                3.100812           3.100812            3.100812          0          0           6      0.980157      0.219049      0.327011    0.105460      0.017577     0.036064 -9.189494e-02         1
  MRNA                3.085182           3.085182            3.085182          0          0           6      0.979846      0.225528      0.347334    0.112581      0.018764     0.054949 -9.776593e-02         1
    TD                2.793792           2.793792            2.793792          0          0           7      0.979439      0.105308      0.162611    0.058205      0.008315     0.013360 -3.660539e-02         1
   CMI                2.739087           2.739087            2.739087          0          0           7      0.977018      0.138949      0.219641    0.080188      0.011455     0.021778 -7.805019e-02         1
  IREN                2.922109           2.922109            2.922109          0          0           6      0.975771      0.654200      0.992616    0.339692      0.056615     0.081890 -3.110085e-01         1
   SQM                2.647835           2.647835            2.647835          0          0           8      0.974741      0.239181      0.392717    0.148316      0.018540     0.037471 -1.298094e-01         1
  AMRC                2.873493           2.873493            2.873493          0          0           6      0.974208      0.357169      0.567227    0.197400      0.032900     0.057315 -1.357760e-01         1
  ARKK                2.674722           2.674722            2.674722          0          0           7      0.973732      0.120400      0.197954    0.074009      0.010573     0.031170 -5.920820e-02         1
   PBF                2.807106           2.807106            2.807106          0          0           6      0.971754      0.280905      0.446369    0.159014      0.026502     0.054975 -8.194164e-02         1
   CDE                2.747851           2.747851            2.747851          0          0           6      0.969206      0.371069      0.609961    0.221978      0.036996     0.053784 -1.805557e-01         1
  BBAI                2.594574           2.594574            2.594574          0          0           7      0.968873      0.387318      0.744843    0.287077      0.041011     0.100749 -2.793021e-01         1
   AMX                2.546195           2.546195            2.546195          0          0           7      0.965465      0.082441      0.137614    0.054047      0.007721     0.019234 -4.470220e-02         1
   WDC                2.473388           2.473388            2.473388          0          0           8      0.962085      0.252355      0.426304    0.172356      0.021545     0.048468 -1.166163e-01         1
  QBTS               27.262769          27.262769            4.000000          1          0           5      0.961973      1.865669      1.274970    0.046766      0.009353     0.092050 -4.676597e-02         1
  SNDK       1000000000.000000        1000.000000            4.000000          1          1           5      0.961973      1.516215      1.036833    0.000000      0.000000     0.061541 -1.000089e-12         1
  GOSS               13.400198          13.400198            4.000000          1          0           5      0.961973      0.928563      0.805407    0.060104      0.012021     0.076667 -6.010411e-02         1
  INCY       1000000000.000000        1000.000000            4.000000          1          1           5      0.961973      0.562549      0.468737    0.000000      0.000000     0.027744 -1.000089e-12         1
   ATI               84.546127          84.546127            4.000000          1          0           5      0.961973      0.510978      0.449470    0.005316      0.001063     0.032751 -5.316272e-03         1
   EGO       1000000000.000000        1000.000000            4.000000          1          1           5      0.961973      0.494380      0.424245    0.000000      0.000000     0.033785 -1.000089e-12         1
  XPEV                9.792376           9.792376            4.000000          1          0           5      0.961973      0.431632      0.431492    0.044064      0.008813     0.044361 -4.406410e-02         1
  ASTL               41.618534          41.618534            4.000000          1          0           5      0.961973      0.387932      0.359758    0.008644      0.001729     0.055804 -8.644169e-03         1
  AMAT       1000000000.000000        1000.000000            4.000000          1          1           5      0.961973      0.343482      0.306880    0.000000      0.000000     0.033983 -1.000089e-12         1
    GH                4.529045           4.529045            4.000000          1          0           5      0.961973      0.303454      0.373663    0.082504      0.016501     0.045151 -8.080994e-02         1
   XME               28.588886          28.588886            4.000000          1          0           5      0.961973      0.276539      0.263780    0.009227      0.001845     0.024713 -9.226673e-03         1
   AER               39.804087          39.804087            4.000000          1          0           5      0.961973      0.240393      0.230185    0.005783      0.001157     0.015993 -5.782951e-03         1
  PALL                7.071625           7.071625            4.000000          1          0           5      0.961973      0.239917      0.263776    0.037301      0.007460     0.022359 -2.093509e-02         1
  RGLD                9.632705           9.632705            4.000000          1          0           5      0.961973      0.211337      0.225456    0.023405      0.004681     0.025302 -2.340528e-02         1
   IBM                8.565089           8.565089            4.000000          1          0           5      0.961973      0.204512      0.218919    0.025559      0.005112     0.023677 -2.555941e-02         1
   GLW               18.529844          18.529844            4.000000          1          0           5      0.961973      0.203038      0.202013    0.010902      0.002180     0.027689 -1.090205e-02         1
  NVGS                3.875412           3.875412            3.875412          0          0           5      0.961669      0.138790      0.182967    0.047212      0.009442     0.022238 -4.721227e-02         1
  ARGX                3.861840           3.861840            3.861840          0          0           5      0.961630      0.158497      0.209806    0.054328      0.010866     0.023503 -5.432812e-02         1
  PGEN                2.492652           2.492652            2.492652          0          0           7      0.961219      0.411504      0.763906    0.306463      0.043780     0.078831 -1.504177e-01         1
  TTMI                2.594360           2.594360            2.594360          0          0           6      0.960662      0.217279      0.365038    0.140704      0.023451     0.046088 -1.357549e-01         1
  EQNR                3.619132           3.619132            3.619132          0          0           5      0.960650      0.100184      0.138852    0.038366      0.007673     0.022284 -3.791291e-02         1
  XERS                3.507467           3.507467            3.507467          0          0           5      0.959958      0.397575      0.539165    0.153719      0.030744     0.044408 -1.537191e-01         1
   QSI                3.421484           3.421484            3.421484          0          0           5      0.959277      0.317873      0.490696    0.143416      0.028683     0.084042 -1.110763e-01         1
  ASTS                3.412474           3.412474            3.412474          0          0           5      0.959197      0.618878      0.851388    0.249493      0.049899     0.080732 -2.339313e-01         1
   BLD                2.572551           2.572551            2.572551          0          0           6      0.959177      0.108187      0.181551    0.070573      0.011762     0.028682 -4.860216e-02         1
   CLF                2.566310           2.566310            2.566310          0          0           6      0.958738      0.298856      0.501749    0.195514      0.032586     0.060995 -1.088547e-01         1
   GEV                3.224580           3.224580            3.224580          0          0           5      0.957047      0.233342      0.349441    0.108368      0.021674     0.041992 -1.078067e-01         1
   VLO                2.436056           2.436056            2.436056          0          0           7      0.956124      0.166566      0.290992    0.119452      0.017065     0.028526 -6.942052e-02         1
  SOFI                2.420492           2.420492            2.420492          0          0           7      0.954604      0.147108      0.259750    0.107313      0.015330     0.053769 -1.037847e-01         1
   OGI                3.074475           3.074475            3.074475          0          0           5      0.954449      0.259566      0.391176    0.127233      0.025447     0.044829 -1.249154e-01         1
    RY                2.507818           2.507818            2.507818          0          0           6      0.954297      0.053646      0.090643    0.036144      0.006024     0.011056 -2.327394e-02         1
  RCUS                3.044831           3.044831            3.044831          0          0           5      0.953813      0.301558      0.453660    0.148994      0.029799     0.055183 -1.226330e-01         1
  ZURA                2.464706           2.464706            2.464706          0          0           6      0.950618      0.377963      0.683835    0.277451      0.046242     0.072542 -2.253286e-01         1
   CCJ                2.307250           2.307250            2.307250          0          0           8      0.944298      0.274545      0.508352    0.220328      0.027541     0.041217 -2.101820e-01         1
  ENTA                2.268582           2.268582            2.268582          0          0           7      0.936624      0.268019      0.509974    0.224799      0.032114     0.052185 -1.289501e-01         1
  SBGI                2.330575           2.330575            2.330575          0          0           6      0.936570      0.131127      0.242192    0.103919      0.017320     0.035444 -4.170793e-02         1
  CENX                2.260867           2.260867            2.260867          0          0           7      0.935541      0.307435      0.570467    0.252322      0.036046     0.048258 -1.899487e-01         1
  PACB                2.253155           2.253155            2.253155          0          0           7      0.934440      0.316782      0.657921    0.292000      0.041714     0.078680 -1.777613e-01         1
  bidu                2.197220           2.197220            2.197220          0          0           8      0.928385      0.194565      0.367216    0.167128      0.020891     0.036231 -1.061193e-01         1
  OMER                2.150890           2.150890            2.150890          0          0           7      0.918003      0.323629      0.786466    0.365647      0.052235     0.073763 -2.480500e-01         1
   XSD                2.107412           2.107412            2.107412          0          0           8      0.912369      0.096335      0.186400    0.088450      0.011056     0.019390 -8.654222e-02         1
   CVI                2.109550           2.109550            2.109550          0          0           7      0.910303      0.169554      0.356131    0.168819      0.024117     0.039383 -1.081802e-01         1
  HOOD                2.095156           2.095156            2.095156          0          0           8      0.909947      0.191338      0.401364    0.191567      0.023946     0.053935 -1.385103e-01         1
  FTAI                2.092427           2.092427            2.092427          0          0           8      0.909400      0.211089      0.440365    0.210456      0.026307     0.042653 -1.750823e-01         1
  BLTE                2.301086           2.301086            2.301086          0          0           5      0.908111      0.126909      0.231605    0.100650      0.020130     0.025869 -5.439084e-02         1
   ASC                2.070942           2.070942            2.070942          0          0           7      0.902509      0.157113      0.321911    0.155442      0.022206     0.030290 -1.486347e-01         1
  MKSI                2.058098           2.058098            2.058098          0          0           8      0.902255      0.151262      0.306608    0.148976      0.018622     0.039798 -1.235124e-01         1
  TARS                2.107298           2.107298            2.107298          0          0           6      0.901838      0.161177      0.333513    0.158266      0.026378     0.036186 -1.528675e-01         1
  SATS       1000000000.000000        1000.000000            4.000000          1          1           4      0.896852      0.856676      0.678365    0.000000      0.000000     0.049401 -1.000089e-12         1
  SNDL       1000000000.000000        1000.000000            4.000000          1          1           4      0.896852      0.707865      0.580602    0.000000      0.000000     0.046883 -1.000089e-12         1
  IAUX                8.764249           8.764249            4.000000          1          0           4      0.896852      0.604531      0.595952    0.067998      0.017000     0.052215 -6.799803e-02         1
    SN       1000000000.000000        1000.000000            4.000000          1          1           4      0.896852      0.322819      0.299289    0.000000      0.000000     0.036652 -1.000089e-12         1
   BFH                7.224877           7.224877            4.000000          1          0           4      0.896852      0.270491      0.294696    0.040789      0.010197     0.028687 -4.078900e-02         1
   KOD                2.019483           2.019483            2.019483          0          0           7      0.891160      0.312178      0.745681    0.369244      0.052749     0.085432 -1.564941e-01         1
  LGND                2.017203           2.017203            2.017203          0          0           7      0.890630      0.131852      0.276551    0.137096      0.019585     0.026045 -1.320180e-01         1
  INTC                2.008094           2.008094            2.008094          0          0           7      0.888493      0.226754      0.511861    0.254899      0.036414     0.046284 -1.986567e-01         1
    HL                1.987857           1.987857            1.987857          0          0           8      0.886068      0.222648      0.498460    0.250752      0.031344     0.052227 -1.505506e-01         1
  RIGL                2.685194           2.685194            2.685194          0          0           4      0.876396      0.193872      0.324350    0.120792      0.030198     0.052840 -1.207918e-01         1
  NAMS                1.952244           1.952244            1.952244          0          0           7      0.874561      0.159676      0.376766    0.192991      0.027570     0.049491 -1.637593e-01         1
  ADPT                1.913571           1.913571            1.913571          0          0           8      0.866493      0.132904      0.333068    0.174055      0.021757     0.051731 -1.713210e-01         1
  IMAX                1.911413           1.911413            1.911413          0          0           7      0.863445      0.166601      0.379000    0.198283      0.028326     0.029711 -1.582911e-01         1
  PSNL                2.450786           2.450786            2.450786          0          0           4      0.860195      0.216403      0.402312    0.164156      0.041039     0.073062 -1.641564e-01         1
  BLOK                1.892151           1.892151            1.892151          0          0           7      0.857918      0.089669      0.202407    0.106972      0.015282     0.030596 -9.178566e-02         1
  NBIS                1.863222           1.863222            1.863222          0          0           7      0.849269      0.174442      0.481247    0.258287      0.036898     0.072417 -2.437237e-01         1
  GSAT                1.829560           1.829560            1.829560          0          0           9      0.841819      0.293597      0.789571    0.431563      0.047951     0.052423 -2.050522e-01         1
  RYTM                1.835321           1.835321            1.835321          0          0           7      0.840525      0.127840      0.318572    0.173578      0.024797     0.037623 -1.024570e-01         1
   HCA                1.828276           1.828276            1.828276          0          0           7      0.838254      0.061289      0.144661    0.079124      0.011303     0.023531 -7.020988e-02         1
    MP                1.802467           1.802467            1.802467          0          0           8      0.832120      0.180774      0.570069    0.316272      0.039534     0.067575 -2.939301e-01         1
   NYT                2.202803           2.202803            2.202803          0          0           4      0.830560      0.054783      0.104270    0.047335      0.011834     0.017242 -4.733516e-02         1
  PVLA                1.804304           1.804304            1.804304          0          0           7      0.830332      0.180870      0.488925    0.270977      0.038711     0.052025 -1.897483e-01         1
   EIS                1.799460           1.799460            1.799460          0          0           7      0.828695      0.047130      0.109272    0.060725      0.008675     0.010707 -4.167935e-02         1
   MDB                1.820506           1.820506            1.820506          0          0           6      0.827917      0.121405      0.301295    0.165501      0.027583     0.040933 -1.323975e-01         1
    ON                1.783931           1.783931            1.783931          0          0           7      0.823364      0.075136      0.186929    0.104785      0.014969     0.043540 -6.725460e-02         1
   MRK                1.740857           1.740857            1.740857          0          0           9      0.811027      0.069618      0.175658    0.100903      0.011211     0.024025 -6.403761e-02         1
  RDVT                1.724066           1.724066            1.724066          0          0           7      0.801631      0.100597      0.274653    0.159305      0.022758     0.027743 -1.222511e-01         1
  OLMA                1.723034           1.723034            1.723034          0          0           6      0.793547      0.163968      0.647814    0.375973      0.062662     0.072360 -2.210120e-01         1
  PBBK                1.717577           1.717577            1.717577          0          0           6      0.791476      0.020648      0.050815    0.029586      0.004931     0.004293 -2.838550e-02         1
  SNBR                1.985754           1.985754            1.985754          0          0           4      0.788457      0.181086      0.507191    0.255415      0.063854     0.080120 -1.756946e-01         1
   NGL       1000000000.000000        1000.000000            4.000000          1          1           3      0.782883      0.368095      0.330796    0.000000      0.000000     0.036820 -1.000089e-12         1
   HIT               11.001895          11.001895            4.000000          1          0           3      0.782883      0.346970      0.394771    0.035882      0.011961     0.091546 -3.588208e-02         1
   AEO                7.701784           7.701784            4.000000          1          0           3      0.782883      0.281656      0.311433    0.040436      0.013479     0.047613 -4.043648e-02         1
  NRIM               12.772625          12.772625            4.000000          1          0           3      0.782883      0.148609      0.156931    0.012286      0.004095     0.021229 -1.228647e-02         1
  JKHY                1.748509           1.748509            1.748509          0          0           5      0.779462      0.054494      0.138037    0.078945      0.015789     0.017553 -7.727153e-02         1
  NVDA                1.723495           1.723495            1.723495          0          0           5      0.770266      0.075388      0.206939    0.120069      0.024014     0.031073 -1.181192e-01         1
   CEG                1.882540           1.882540            1.882540          0          0           4      0.761506      0.055458      0.135180    0.071807      0.017952     0.042083 -3.680065e-02         1
  NAIL                1.604070           1.604070            1.604070          0          0          10      0.755739      0.216076      0.842659    0.525326      0.052533     0.066378 -2.579529e-01         1
   JPM                1.628687           1.628687            1.628687          0          0           6      0.755591      0.033134      0.091736    0.056325      0.009388     0.019529 -4.748151e-02         1
  VERA                1.861057           1.861057            1.861057          0          0           4      0.755266      0.146093      0.439189    0.235989      0.058997     0.054737 -2.286478e-01         1
  SCHD                1.608874           1.608874            1.608874          0          0           7      0.754593      0.024910      0.068730    0.042719      0.006103     0.010465 -4.230152e-02         1
  XPEL                1.822672           1.822672            1.822672          0          0           4      0.743556      0.080263      0.201276    0.110429      0.027607     0.032176 -1.080326e-01         1
  OWLT                1.574700           1.574700            1.574700          0          0           8      0.741693      0.137352      0.518861    0.329498      0.041187     0.046162 -2.016630e-01         1
  IMMX                1.568010           1.568010            1.568010          0          0           7      0.736335      0.104678      0.439144    0.280064      0.040009     0.060723 -1.982299e-01         1
  NVTS                1.756455           1.756455            1.756455          0          0           4      0.721636      0.144141      0.550987    0.313692      0.078423     0.098052 -2.439828e-01         1
   NEO                1.524752           1.524752            1.524752          0          0           8      0.718477      0.102658      0.446800    0.293031      0.036629     0.051330 -2.544138e-01         1
  CRWV                1.706680           1.706680            1.706680          0          0           4      0.703717      0.132159      0.598471    0.350664      0.087666     0.089219 -2.491662e-01         1
   AEM                1.487838           1.487838            1.487838          0          0          10      0.701548      0.072839      0.252746    0.169875      0.016987     0.029322 -1.145824e-01         1
    GM                1.495971           1.495971            1.495971          0          0           6      0.694984      0.058714      0.203135    0.135788      0.022631     0.024737 -9.443375e-02         1
   BAC                1.451453           1.451453            1.451453          0          0           6      0.673041      0.036984      0.134757    0.092843      0.015474     0.019041 -6.548711e-02         1
  OKLO                1.405586           1.405586            1.405586          0          0           7      0.656957      0.057570      0.809284    0.575763      0.082252     0.096089 -4.542463e-01         1
  FTXL                1.416218           1.416218            1.416218          0          0           6      0.655210      0.019605      0.070562    0.049824      0.008304     0.012906 -4.920464e-02         1
  PLAB                1.415327           1.415327            1.415327          0          0           6      0.654754      0.065666      0.300309    0.212184      0.035364     0.033141 -1.611241e-01         1
  MDGL                1.459186           1.459186            1.459186          0          0           5      0.654672      0.062257      0.305558    0.209403      0.041881     0.038293 -1.558808e-01         1
     M                1.377061           1.377061            1.377061          0          0           7      0.642137      0.060994      0.305507    0.221855      0.031694     0.035668 -1.535291e-01         1
   IQV                1.359413           1.359413            1.359413          0          0          10      0.635945      0.066250      0.321419    0.236440      0.023644     0.025939 -1.368337e-01         1
  CRSP                1.409240           1.409240            1.409240          0          0           5      0.629688      0.044195      0.251258    0.178293      0.035659     0.057046 -1.754767e-01         1
  PTCT                1.365829           1.365829            1.365829          0          0           6      0.629129      0.062990      0.415687    0.304347      0.050725     0.034649 -2.062969e-01         1
   HLF                1.344888           1.344888            1.344888          0          0           8      0.627399      0.058567      0.363260    0.270105      0.033763     0.046047 -1.449484e-01         1
   APH                1.330097           1.330097            1.330097          0          0           8      0.619547      0.040728      0.204985    0.154113      0.019264     0.027277 -7.291474e-02         1
  META                1.332112           1.332112            1.332112          0          0           7      0.618436      0.033381      0.214943    0.161355      0.023051     0.024979 -1.531731e-01         1
   ICE                1.331279           1.331279            1.331279          0          0           7      0.617993      0.031054      0.157985    0.118671      0.016953     0.014857 -1.139356e-01         1
  AVDL                1.375190           1.375190            1.375190          0          0           5      0.612273      0.043302      0.236417    0.171916      0.034383     0.039183 -1.128155e-01         1
  CLSK                1.325873           1.325873            1.325873          0          0           6      0.608078      0.052982      0.414657    0.312743      0.052124     0.082110 -1.583123e-01         1
   RGC                1.464873           1.464873            1.464873          0          0           4      0.600300      0.072623      0.521806    0.356213      0.089053     0.072466 -3.244931e-01         1
   RTX                1.293114           1.293114            1.293114          0          0           7      0.597630      0.030898      0.167472    0.129511      0.018502     0.018756 -1.238107e-01         1
   NSC                1.297449           1.297449            1.297449          0          0           6      0.592968      0.016406      0.080246    0.061849      0.010308     0.016857 -3.165507e-02         1
   UHS                1.286075           1.286075            1.286075          0          0           6      0.586900      0.030872      0.191100    0.148592      0.024765     0.027131 -6.771647e-02         1
   AXP                1.272714           1.272714            1.272714          0          0           6      0.579763      0.022796      0.128949    0.101318      0.016886     0.020916 -4.294839e-02         1
  TCMD                1.264833           1.264833            1.264833          0          0           6      0.575549      0.023095      0.155627    0.123041      0.020507     0.034699 -1.193714e-01         1
   GLD                1.232120           1.232120            1.232120          0          0           9      0.567656      0.022144      0.143135    0.116169      0.012908     0.012373 -1.036850e-01         1
   NVT                1.219452           1.219452            1.219452          0          0           6      0.551282      0.019565      0.184335    0.151162      0.025194     0.030306 -1.458221e-01         1
   HUT                1.174143           1.174143            1.174143          0          0           6      0.527170     -0.021181      0.410924    0.349978      0.058330     0.075929 -2.549559e-01         1
  DDOG                1.208608           1.208608            1.208608          0          0           5      0.524825      0.017801      0.243595    0.201550      0.040310     0.035849 -1.914773e-01         1
  ABBV                1.140214           1.140214            1.140214          0          0           8      0.517984      0.010983      0.141456    0.124061      0.015508     0.020749 -1.045413e-01         1
  ALNY                1.133763           1.133763            1.133763          0          0           8      0.514589      0.006032      0.284848    0.251241      0.031405     0.032827 -2.277850e-01         1
   AIQ                1.128749           1.128749            1.128749          0          0          10      0.512765      0.008166      0.176318    0.156206      0.015621     0.016526 -1.127629e-01         1
  CBRL                1.115769           1.115769            1.115769          0          0           8      0.505165      0.001328      0.315045    0.282357      0.035295     0.048921 -2.561519e-01         1
  AMKR                1.086690           1.086690            1.086690          0          0           8      0.490100     -0.004166      0.222560    0.204805      0.025601     0.040938 -1.965675e-01         1
   TJX                1.086734           1.086734            1.086734          0          0           7      0.488105      0.004616      0.119722    0.110167      0.015738     0.015331 -6.976349e-02         1
  VSCO                1.078982           1.078982            1.078982          0          0           8      0.486146     -0.017439      0.318971    0.295622      0.036953     0.048295 -2.625821e-01         1
     F                1.033770           1.033770            1.033770          0          0           7      0.461341     -0.006245      0.177958    0.172144      0.024592     0.023725 -1.619916e-01         1
  ARKW                0.987494           0.987494            0.987494          0          0           8      0.440715     -0.011748      0.162368    0.164424      0.020553     0.023216 -9.474603e-02         1
  DIBS                1.036055           1.036055            1.036055          0          0           5      0.436620     -0.004638      0.140283    0.135401      0.027080     0.041066 -8.702375e-02         1
  LYFT                0.978966           0.978966            0.978966          0          0           6      0.428432     -0.063351      0.376090    0.384171      0.064028     0.045033 -2.938216e-01         1
  JAZZ                0.973599           0.973599            0.973599          0          0           6      0.425900     -0.016188      0.164591    0.169054      0.028176     0.025302 -8.731075e-02         1
  EPAM                0.964603           0.964603            0.964603          0          0           6      0.421685     -0.026405      0.219722    0.227784      0.037964     0.029929 -2.095823e-01         1
  REGN                0.925764           0.925764            0.925764          0          0           8      0.411953     -0.041067      0.232219    0.250840      0.031355     0.028301 -2.037372e-01         1
   CRS                0.924510           0.924510            0.924510          0          0           7      0.409484     -0.008798      0.074257    0.080320      0.011474     0.034695 -7.880506e-02         1
   VRT                0.901576           0.901576            0.901576          0          0           6      0.393184     -0.049742      0.217682    0.241446      0.040241     0.048220 -1.830541e-01         1
  ILMN                0.872668           0.872668            0.872668          0          0           7      0.386781     -0.047892      0.186591    0.213817      0.030545     0.035727 -2.005226e-01         1
    TK                1.010449           1.010449            1.010449          0          0           4      0.375102     -0.005605      0.099842    0.098810      0.024702     0.027728 -8.198455e-02         1
   WGS                0.844610           0.844610            0.844610          0          0           7      0.375059     -0.064878      0.222797    0.263787      0.037684     0.053459 -1.388731e-01         1
  DSGN                0.842293           0.842293            0.842293          0          0           7      0.374110     -0.119917      0.318275    0.377867      0.053981     0.060832 -3.392043e-01         1
  GRAL                0.835105           0.835105            0.835105          0          0           8      0.373023     -0.046547      0.148025    0.177254      0.022157     0.082115 -1.615499e-01         1
   ZIM                0.842713           0.842713            0.842713          0          0           6      0.368319     -0.058123      0.201467    0.239069      0.039845     0.046297 -2.205644e-01         1
   LMT                0.857248           0.857248            0.857248          0          0           5      0.356097     -0.028006      0.123041    0.143531      0.028706     0.019674 -1.192624e-01         1
  NTRB                0.839093           0.839093            0.839093          0          0           5      0.348775     -0.049865      0.173611    0.206903      0.041381     0.035439 -1.048720e-01         1
  FROG                0.791137           0.791137            0.791137          0          0           6      0.348016     -0.071891      0.196993    0.249000      0.041500     0.036821 -1.253122e-01         1
  TNDM                0.783444           0.783444            0.783444          0          0           6      0.345109     -0.140526      0.252879    0.322779      0.053797     0.055367 -3.163078e-01         1
   MAR                0.749178           0.749178            0.749178          0          0           9      0.340643     -0.021517      0.057855    0.077224      0.008580     0.020799 -5.725760e-02         1
  DELL                0.749863           0.749863            0.749863          0          0           7      0.338561     -0.091668      0.222326    0.296489      0.042356     0.035961 -1.919738e-01         1
  GLSI                0.749841           0.749841            0.749841          0          0           7      0.338553     -0.098157      0.228940    0.305319      0.043617     0.030716 -2.181808e-01         1
  ROST                0.748730           0.748730            0.748730          0          0           7      0.338154     -0.028540      0.073989    0.098820      0.014117     0.018708 -9.032947e-02         1
   XLK                0.685725           0.685725            0.685725          0          0           8      0.318355     -0.037905      0.076631    0.111752      0.013969     0.017899 -7.924155e-02         1
   CNQ                0.667484           0.667484            0.667484          0          0           6      0.305167     -0.072770      0.129964    0.194708      0.032451     0.022349 -1.216141e-01         1
  JACK                0.635056           0.635056            0.635056          0          0           7      0.300825     -0.138688      0.199608    0.314316      0.044902     0.059531 -1.596189e-01         1
  ANDE                0.632796           0.632796            0.632796          0          0           7      0.300152     -0.086381      0.131454    0.207735      0.029676     0.025776 -1.386084e-01         1
   IYW                0.597430           0.597430            0.597430          0          0           9      0.292184     -0.063015      0.090097    0.150808      0.016756     0.014999 -1.075849e-01         1
  ALAB                0.660431           0.660431            0.660431          0          0           5      0.286033     -0.213867      0.318549    0.482334      0.096467     0.072680 -4.110633e-01         1
   INO                0.576175           0.576175            0.576175          0          0           6      0.278723     -0.208344      0.249937    0.433787      0.072298     0.064068 -3.687751e-01         1
  ARKF                0.535053           0.535053            0.535053          0          0           8      0.275211     -0.107206      0.118433    0.221349      0.027669     0.022670 -1.738856e-01         1
  PLTR                0.506807           0.506807            0.506807          0          0           7      0.266726     -0.171378      0.167455    0.330411      0.047202     0.046765 -2.029167e-01         1
  AAPL                0.493903           0.493903            0.493903          0          0           8      0.265377     -0.076227      0.073335    0.148480      0.018560     0.019993 -7.622656e-02         1
   RCL                0.499133           0.499133            0.499133          0          0           6      0.259627     -0.096292      0.091094    0.182505      0.030417     0.031688 -1.356625e-01         1
  TQQQ                0.453775           0.453775            0.453775          0          0           8      0.256526     -0.201177      0.166207    0.366277      0.045785     0.040253 -2.346086e-01         1
  CTSH                0.545030           0.545030            0.545030          0          0           5      0.254363     -0.065837      0.074342    0.136399      0.027280     0.020151 -1.304787e-01         1
   NKE                0.446632           0.446632            0.446632          0          0           7      0.253403     -0.132444      0.103717    0.232220      0.033174     0.028514 -2.132625e-01         1
   TMO                0.430443           0.430443            0.430443          0          0           9      0.252192     -0.120894      0.089790    0.208599      0.023178     0.024180 -1.719229e-01         1
   VTR                0.432763           0.432763            0.432763          0          0           7      0.250555     -0.085104      0.064432    0.148885      0.021269     0.016171 -1.404840e-01         1
  WEAV                0.391266           0.391266            0.391266          0          0           7      0.242505     -0.190478      0.124431    0.318022      0.045432     0.042795 -2.368047e-01         1
   FDN                0.390453           0.390453            0.390453          0          0           7      0.242354     -0.060934      0.038771    0.099298      0.014185     0.014715 -7.162947e-02         1
  ANET                0.410430           0.410430            0.410430          0          0           6      0.240956     -0.198613      0.134288    0.327187      0.054531     0.039425 -2.661003e-01         1
  MSFT                0.377606           0.377606            0.377606          0          0           7      0.240004     -0.044407      0.026581    0.070393      0.010056     0.017993 -6.310107e-02         1
  ACWX                0.354251           0.354251            0.354251          0          0          10      0.238104     -0.065605      0.036436    0.102854      0.010285     0.008607 -9.279505e-02         1
  AVGO                0.349779           0.349779            0.349779          0          0          10      0.237337     -0.258469      0.147250    0.420981      0.042098     0.038222 -2.997377e-01         1
  ADBE                0.463425           0.463425            0.463425          0          0           5      0.235802     -0.112563      0.094235    0.203346      0.040669     0.024628 -1.889893e-01         1
  ANVS                0.339353           0.339353            0.339353          0          0           6      0.228287     -0.201010      0.104139    0.306874      0.051146     0.081295 -2.768907e-01         1
  RGEN                0.286295           0.286295            0.286295          0          0           8      0.226603     -0.322431      0.142368    0.497276      0.062160     0.038063 -3.224311e-01         1
  ASML                0.291770           0.291770            0.291770          0          0           6      0.220826     -0.127117      0.052872    0.181211      0.030202     0.025928 -1.714921e-01         1
   SPY                0.234176           0.234176            0.234176          0          0           9      0.219764     -0.054786      0.016993    0.072567      0.008063     0.010809 -6.572012e-02         1
  AZTA                0.156821           0.156821            0.156821          0          0           8      0.209911     -0.188726      0.036844    0.234942      0.029368     0.038094 -2.176740e-01         1
   NFE                0.099947           0.099947            0.099947          0          0           6      0.197638     -0.583171      0.080504    0.805466      0.134244     0.142923 -6.149782e-01         1
  NFLX                0.000000           0.000000            0.000000          0          0           8      0.195321     -0.366864      0.000000    0.429944      0.053743     0.025512 -3.668644e-01         1
   NOW                0.046201           0.046201            0.046201          0          0           6      0.192774     -0.329941      0.018534    0.401173      0.066862     0.028194 -3.299412e-01         1
  APPN                0.015563           0.015563            0.015563          0          0           4      0.139783     -0.176840      0.002949    0.189473      0.047368     0.037424 -1.768402e-01         1
Saved per-ticker CSV to: output\per_ticker_20260118_2327.csv
real 1150.314   user 0.108      sys 0.076       pcpu 0.01

1/17/2026
To optimize the engine for high-density aggression while maintaining signal quality, 
we shifted away from standard RSI/EMA crossovers to a more adaptive, volatility-adjusted framework.
Here is the summary of the three core pillars we implemented:
1. Kaufman’s Adaptive Moving Average (VIDYA)Instead of a static smoothing period, we used the Volatility Index Dynamic Average (VIDYA). 
This serves as our "intelligent" baseline.The Mechanism: It uses an Efficiency Ratio (ER) that compares the "net" move of price to the "sum" of all absolute changes (noise).
The Result: When the market is trending strongly, the VIDYA speeds up to stay tight to price. When the market is choppy and "noisy," the indicator slows down significantly to prevent false signals.
2. Zero-Lag EMA (ZLEMA) Signal Lines: To generate the actual buy/sell triggers, we applied a Zero-Lag transformation to the VIDYA output.
The Mechanism: The ZLEMA reduces the inherent lag of traditional EMAs by calculating the difference between the current price and the price 4$n$ periods ago, then adding that difference back to the current price before smoothing ($DeLagged = Price + (Price - Price_{lag})$).
The Result: We created a "Fast" and "Slow" ZLEMA relationship. Because they respond faster than standard averages, we can capture the meat of a move several bars earlier than a typical crossover system.
3. Dynamic Regime Filter: To ensure the "aggression" doesn't turn into "suicide" during bear markets, we implemented a Slope-based Regime Filter.The Mechanism: We look at the slope of a long-term EMA ($RegimeRatio \times SlowPeriod$). 
A trade is only eligible if the price is above this line AND the line itself has a positive gradient over the last 5 bars.The Result: This acts as a global "On/Off" switch. 
It automatically sidelines the strategy during sustained downtrends or high-volatility "washouts," protecting your capital for when the trend resumes.

Final Implementation Summary
ComponentFunctionBenefit: Kaufman VIDYA Noise Filter: Stops the strategy from "chasing" noise in flat markets.
ZLEMA DuoTrigger Engine: Provides ultra-fast entry signals with minimal lag.
Regime Filter Risk Control: Prevents buying into "falling knives" during macro downtrends.

Alpaca_options.csv data (281 files)
=== OVERALL (ALL FILES) METRICS (BEST PARAMS) ===
Avg Profit Factor (raw):      35587193.784590
Avg Profit Factor (eff):      40.759679   (loss_floor=0.001)
Avg Profit Factor (diag cap): 3.258582   (cap=5.0)
Avg Trades / Ticker:          7.139
Mean Ticker Score (eligible): 0.881173
Coverage (eligible/total):    281/281 = 1.000
Penalty enabled:              enabled (soft)
Soft penalty center/k:        0.01 / 10.0
Tail-protection ret floor/k:  -0.15 / 2.0
Max-trades penalty:           max_trades=100, k=0.1
PF-floor penalty:             pf_floor=1.0, k=5.0
cooldown_bars:                1 (opt=False) [loss-only]
time_stop_bars:               21 (0=disabled) (opt=True)
num_neg (return<=0):          24
commission_rate_per_side:     0.000600
PF ROC baseline/k:            1.150 / 2.500
Trades ROC baseline/k:        3.000 / 1.200
weight_pf:                    0.700
score_power:                  1.500
min_trades gate:              1
loss_floor (scoring):         0.001
threshold_mode:               fixed
Score (single fill 'next_open'): 0.677204

=== BEST PARAMS ===
fp: 8
reg_ratio: 2.118090063782655
sp: 21
ts: 21
vl: 6
vs: 8
best_score (OPTUNA objective): 0.6772035304071539
fill_mode (final report): next_open

=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===
ticker  profit_factor_raw  profit_factor_eff  profit_factor_diag  num_trades  ticker_score  total_return  gross_profit  gross_loss         maxdd  eligible
   LIT       1.000000e+09        1000.000000            5.000000          11      0.999900      0.589025      0.476066    0.000000 -1.000089e-12         1
   CAT       5.464361e+00           5.464361            5.000000          10      0.999829      0.378273      0.414498    0.075855 -5.426190e-02         1
   FDX       3.272053e+01          32.720530            5.000000           9      0.999595      0.478860      0.421973    0.012896 -1.289629e-02         1
  LRCX       1.556113e+01          15.561135            5.000000           9      0.999595      0.800346      0.674302    0.043332 -3.401586e-02         1
   MLI       1.351049e+01          13.510494            5.000000           9      0.999595      0.219457      0.219865    0.016274 -1.622702e-02         1
   XME       1.148498e+01          11.484981            5.000000           9      0.999595      0.354102      0.346859    0.030201 -2.097443e-02         1
   AMG       8.923112e+00           8.923112            5.000000           9      0.999595      0.393974      0.392638    0.044002 -4.351834e-02         1
    AG       5.482859e+00           5.482859            5.000000           9      0.999595      0.753969      0.770794    0.140582 -8.722151e-02         1
  RKLB       5.252235e+00           5.252235            5.000000           9      0.999595      0.762122      0.795522    0.151463 -1.044732e-01         1
  BABA       4.739959e+00           4.739959            4.739959           9      0.999531      0.389049      0.448674    0.094658 -6.185398e-02         1
   WDC       4.277996e+00           4.277996            4.277996           9      0.999243      0.561720      0.644619    0.150683 -8.217534e-02         1
  AREC       4.224363e+00           4.224363            4.224363           9      0.999182      1.584065      1.603946    0.379689 -2.356322e-01         1
   APH       4.184434e+00           4.184434            4.184434           9      0.999132      0.336039      0.413869    0.098907 -7.229005e-02         1
   AMD       4.102271e+00           4.102271            4.102271           9      0.999011      0.585588      0.688988    0.167953 -1.178093e-01         1
  GLUE       1.000000e+09        1000.000000            5.000000           8      0.998818      1.798680      1.132636    0.000000 -1.000089e-12         1
    GS       1.000000e+09        1000.000000            5.000000           8      0.998818      0.437460      0.380219    0.000000 -1.000089e-12         1
   WWD       1.000000e+09        1000.000000            5.000000           8      0.998818      0.346014      0.309967    0.000000 -1.000089e-12         1
  COHR       3.524035e+01          35.240347            5.000000           8      0.998818      1.390878      0.995660    0.028253 -2.825340e-02         1
   CMI       1.839228e+01          18.392276            5.000000           8      0.998818      0.398296      0.372541    0.020255 -2.025530e-02         1
    MU       1.316523e+01          13.165229            5.000000           8      0.998818      1.097567      0.874343    0.066413 -6.641304e-02         1
  MKSI       1.014095e+01          10.140948            5.000000           8      0.998818      0.691522      0.646713    0.063772 -6.281954e-02         1
  EOSE       8.196835e+00           8.196835            5.000000           8      0.998818      2.161857      1.542749    0.188213 -1.861665e-01         1
 GOOGL       6.261159e+00           6.261159            5.000000           8      0.998818      0.408883      0.431426    0.068905 -4.511360e-02         1
  ARKK       5.048702e+00           5.048702            5.000000           8      0.998818      0.365150      0.426862    0.084549 -8.430094e-02         1
  SOFI       4.890723e+00           4.890723            4.890723           8      0.998796      0.619258      0.691177    0.141324 -1.345935e-01         1
  ASTS       4.848377e+00           4.848377            4.848377           8      0.998786      1.258561      1.209636    0.249493 -2.339313e-01         1
    GH       4.272730e+00           4.272730            4.272730           8      0.998461      0.419207      0.502954    0.117713 -7.172802e-02         1
  APLD       4.073399e+00           4.073399            4.073399           8      0.998185      1.067360      1.181514    0.290056 -1.830830e-01         1
  OLMA       4.037976e+00           4.037976            4.037976           8      0.998120      0.843257      1.077369    0.266809 -2.253374e-01         1
  ARKW       3.908325e+00           3.908325            3.908325           8      0.997827      0.237810      0.313742    0.080275 -7.312308e-02         1
  BLOK       3.874752e+00           3.874752            3.874752           8      0.997734      0.276287      0.357381    0.092233 -9.072442e-02         1
   CDE       3.868652e+00           3.868652            3.868652           8      0.997716      0.401680      0.514157    0.132903 -1.329035e-01         1
   ZIM       3.738357e+00           3.738357            3.738357           8      0.997266      0.706928      0.842043    0.225244 -1.906365e-01         1
    ON       3.704607e+00           3.704607            3.704607           8      0.997123      0.330168      0.453372    0.122381 -9.353484e-02         1
   RIG       3.482683e+00           3.482683            3.482683           9      0.996596      0.541135      0.701798    0.201511 -1.487374e-01         1
  AXTI       1.000000e+09        1000.000000            5.000000           7      0.996260      2.257606      1.318374    0.000000 -1.000089e-12         1
   XBI       1.246678e+02         124.667757            5.000000           7      0.996260      0.245656      0.227861    0.001828 -1.827746e-03         1
   TEL       2.489132e+01          24.891324            5.000000           7      0.996260      0.349605      0.327919    0.013174 -1.317401e-02         1
   TSM       2.483404e+01          24.834035            5.000000           7      0.996260      0.396271      0.370966    0.014938 -9.951575e-03         1
  GOSS       2.414794e+01          24.147945            5.000000           7      0.996260      2.363434      1.451391    0.060104 -6.010411e-02         1
  REMX       2.327047e+01          23.270465            5.000000           7      0.996260      0.457348      0.410808    0.017654 -1.765363e-02         1
    TD       1.874210e+01          18.742099            5.000000           7      0.996260      0.198125      0.195313    0.010421 -7.265433e-03         1
   CRS       1.581718e+01          15.817180            5.000000           7      0.996260      0.335359      0.337056    0.021309 -2.130947e-02         1
  FUTU       1.412853e+01          14.128528            5.000000           7      0.996260      0.694599      0.616019    0.043601 -4.360109e-02         1
  LABU       1.310138e+01          13.101377            5.000000           7      0.996260      0.671103      0.602995    0.046025 -4.270156e-02         1
   XSD       1.237604e+01          12.376043            5.000000           7      0.996260      0.267983      0.275951    0.022297 -1.996200e-02         1
  TERN       1.145384e+01          11.453838            5.000000           7      0.996260      1.747162      1.266070    0.110537 -1.105367e-01         1
    SU       1.133807e+01          11.338067            5.000000           7      0.996260      0.250065      0.254035    0.022406 -1.217618e-02         1
     U       1.030767e+01          10.307665            5.000000           7      0.996260      0.716447      0.652354    0.063288 -5.795147e-02         1
  PASG       9.311650e+00           9.311650            5.000000           7      0.996260      2.137486      2.580560    0.277132 -1.522573e-01         1
   IMO       8.515727e+00           8.515727            5.000000           7      0.996260      0.271526      0.283297    0.033267 -2.772290e-02         1
  DBRG       8.485990e+00           8.485990            5.000000           7      0.996260      0.424176      0.449735    0.052997 -5.222945e-02         1
   FTI       8.356195e+00           8.356195            5.000000           7      0.996260      0.431183      0.431941    0.051691 -4.176269e-02         1
  RVMD       8.307061e+00           8.307061            5.000000           7      0.996260      0.602295      0.576207    0.069364 -6.817352e-02         1
   CLS       8.286957e+00           8.286957            5.000000           7      0.996260      1.121806      0.959394    0.115772 -1.157716e-01         1
  RGTI       6.880869e+00           6.880869            5.000000           7      0.996260      1.267955      1.124672    0.163449 -1.575480e-01         1
    PL       6.287612e+00           6.287612            5.000000           7      0.996260      0.931103      0.896966    0.142656 -1.379137e-01         1
  RYTM       6.069598e+00           6.069598            5.000000           7      0.996260      0.221104      0.256997    0.042342 -2.764888e-02         1
  NVGS       6.050600e+00           6.050600            5.000000           7      0.996260      0.311555      0.342294    0.056572 -4.786213e-02         1
   GEV       5.835532e+00           5.835532            5.000000           7      0.996260      0.636503      0.662983    0.113611 -1.125106e-01         1
    CX       5.649975e+00           5.649975            5.000000           7      0.996260      0.378619      0.417011    0.073808 -7.380759e-02         1
   ERO       5.478027e+00           5.478027            5.000000           7      0.996260      0.647103      0.674585    0.123144 -1.150020e-01         1
  AXSM       5.461505e+00           5.461505            5.000000           7      0.996260      0.396149      0.436970    0.080009 -8.000905e-02         1
   TAN       5.269855e+00           5.269855            5.000000           7      0.996260      0.218797      0.260304    0.049395 -4.901140e-02         1
    GE       5.055927e+00           5.055927            5.000000           7      0.996260      0.277866      0.323748    0.064033 -6.262418e-02         1
   PBF       4.929795e+00           4.929795            4.929795           7      0.996247      0.849655      0.896877    0.181930 -1.491375e-01         1
  DSGN       4.887073e+00           4.887073            4.887073           7      0.996237      0.647484      0.735346    0.150468 -1.169768e-01         1
  AMRC       3.536130e+00           3.536130            3.536130           8      0.996202      0.718804      0.924895    0.261556 -1.812671e-01         1
  NTRA       4.729862e+00           4.729862            4.729862           7      0.996193      0.275959      0.332929    0.070389 -5.716079e-02         1
  INTC       4.499210e+00           4.499210            4.499210           7      0.996087      0.536967      0.622018    0.138250 -1.376606e-01         1
  BBAI       4.433314e+00           4.433314            4.433314           7      0.996044      0.637373      0.763107    0.172130 -1.721302e-01         1
   AMX       3.510853e+00           3.510853            3.510853           8      0.996027      0.211164      0.281766    0.080256 -4.470220e-02         1
  SIMO       4.223463e+00           4.223463            4.223463           7      0.995847      0.369042      0.462676    0.109549 -1.026495e-01         1
  BBIO       3.479492e+00           3.479492            3.479492           8      0.995795      0.177577      0.244977    0.070406 -2.510819e-02         1
   STX       4.155127e+00           4.155127            4.155127           7      0.995757      0.321396      0.410214    0.098725 -9.675516e-02         1
   OGI       4.146135e+00           4.146135            4.146135           7      0.995744      0.476985      0.576812    0.139120 -9.055132e-02         1
    NG       4.124797e+00           4.124797            4.124797           7      0.995712      0.659910      0.777923    0.188597 -1.657299e-01         1
  GRAL       4.117771e+00           4.117771            4.117771           7      0.995701      0.606614      0.755441    0.183459 -1.593072e-01         1
  HOOD       3.448473e+00           3.448473            3.448473           8      0.995547      0.443632      0.610881    0.177145 -1.049454e-01         1
  PAAS       3.971055e+00           3.971055            3.971055           7      0.995423      0.395325      0.493489    0.124271 -1.242714e-01         1
   HLF       3.860003e+00           3.860003            3.860003           7      0.995133      0.280796      0.378711    0.098112 -9.689982e-02         1
  NEGG       3.750710e+00           3.750710            3.750710           7      0.994758      0.990834      1.373707    0.366253 -2.304997e-01         1
  DLTR       3.281656e+00           3.281656            3.281656           9      0.994603      0.233997      0.326890    0.099611 -9.961116e-02         1
    HL       3.257842e+00           3.257842            3.257842           9      0.994294      0.399606      0.551783    0.169371 -1.239496e-01         1
  ENVA       3.555448e+00           3.555448            3.555448           7      0.993772      0.314017      0.409903    0.115289 -7.997924e-02         1
  SNOW       3.273620e+00           3.273620            3.273620           8      0.993726      0.281929      0.402775    0.123037 -1.089985e-01         1
  DIBS       3.515993e+00           3.515993            3.515993           7      0.993508      0.313060      0.419407    0.119286 -7.782835e-02         1
  CIEN       3.483953e+00           3.483953            3.483953           7      0.993273      0.395431      0.530668    0.152318 -8.845930e-02         1
  DINO       3.469941e+00           3.469941            3.469941           7      0.993165      0.237114      0.327084    0.094262 -5.505438e-02         1
   CLF       3.220931e+00           3.220931            3.220931           8      0.993004      0.550048      0.748348    0.232339 -1.231735e-01         1
  TTMI       3.370406e+00           3.370406            3.370406           7      0.992275      0.332602      0.466892    0.138527 -1.337310e-01         1
   BAC       3.362477e+00           3.362477            3.362477           7      0.992194      0.193081      0.270054    0.080314 -5.541138e-02         1
  ZURA       3.303422e+00           3.303422            3.303422           7      0.991540      0.805516      1.084558    0.328314 -2.869648e-01         1
  ADPT       3.053922e+00           3.053922            3.053922          10      0.990993      0.381397      0.555364    0.181853 -1.570995e-01         1
  TQQQ       3.058333e+00           3.058333            3.058333           9      0.990856      0.357150      0.573625    0.187561 -1.496940e-01         1
  ARKF       3.033059e+00           3.033059            3.033059           9      0.990288      0.201645      0.301105    0.099274 -9.790332e-02         1
  NAMS       3.158758e+00           3.158758            3.158758           7      0.989469      0.351222      0.521569    0.165118 -1.503173e-01         1
  FTAI       3.015730e+00           3.015730            3.015730           8      0.989103      0.358875      0.519443    0.172245 -1.671685e-01         1
  PALL       3.120117e+00           3.120117            3.120117           7      0.988779      0.206363      0.296302    0.094965 -8.659201e-02         1
  AMAT       1.000000e+09        1000.000000            5.000000           6      0.987986      0.503362      0.432185    0.000000 -1.000089e-12         1
  ASTL       1.000000e+09        1000.000000            5.000000           6      0.987986      0.816498      0.653928    0.000000 -1.000089e-12         1
  COGT       2.843237e+01          28.432368            5.000000           6      0.987986      1.740922      1.178322    0.041443 -4.144298e-02         1
  RCUS       2.604073e+01          26.040731            5.000000           6      0.987986      0.896116      0.724448    0.027820 -2.781979e-02         1
   GLW       2.419603e+01          24.196030            5.000000           6      0.987986      0.370310      0.344672    0.014245 -1.424497e-02         1
   ATI       1.072526e+01          10.725259            5.000000           6      0.987986      0.484671      0.476529    0.044431 -4.394803e-02         1
   QSI       9.169333e+00           9.169333            5.000000           6      0.987986      1.110660      0.980785    0.106964 -7.114819e-02         1
  UTHR       8.150484e+00           8.150484            5.000000           6      0.987986      0.567057      0.588827    0.072244 -5.498323e-02         1
   NGL       7.699555e+00           7.699555            5.000000           6      0.987986      0.509938      0.513324    0.066669 -6.546932e-02         1
   LVS       6.203389e+00           6.203389            5.000000           6      0.987986      0.362671      0.401826    0.064775 -6.428125e-02         1
  PSNL       5.970187e+00           5.970187            5.000000           6      0.987986      1.026378      0.980044    0.164156 -1.641564e-01         1
  BOOT       5.966179e+00           5.966179            5.000000           6      0.987986      0.329139      0.383377    0.064258 -5.691723e-02         1
  SPHR       5.839841e+00           5.839841            5.000000           6      0.987986      0.642429      0.679405    0.116340 -1.152838e-01         1
   AXP       5.359908e+00           5.359908            5.000000           6      0.987986      0.234784      0.277162    0.051710 -3.464428e-02         1
   DBI       5.226933e+00           5.226933            5.000000           6      0.987986      1.103908      1.111235    0.212598 -2.001619e-01         1
  BITF       5.150912e+00           5.150912            5.000000           6      0.987986      1.473256      1.388962    0.269654 -2.696536e-01         1
    RY       5.141371e+00           5.141371            5.000000           6      0.987986      0.124407      0.150214    0.029217 -2.327394e-02         1
  NDAQ       4.815964e+00           4.815964            4.815964           6      0.987946      0.156242      0.193780    0.040237 -4.023701e-02         1
    SA       4.658015e+00           4.658015            4.658015           6      0.987893      0.362108      0.433764    0.093122 -6.192252e-02         1
   NSC       4.584479e+00           4.584479            4.584479           6      0.987860      0.114946      0.145846    0.031813 -3.175985e-02         1
  CSIQ       4.436952e+00           4.436952            4.436952           6      0.987773      0.814163      0.958609    0.216051 -2.057633e-01         1
  MRNA       4.178229e+00           4.178229            4.178229           6      0.987517      0.342608      0.434559    0.104005 -9.909813e-02         1
  PLAB       4.115568e+00           4.115568            4.115568           6      0.987425      0.380088      0.468241    0.113773 -9.127115e-02         1
  TARS       4.070665e+00           4.070665            4.070665           6      0.987351      0.253513      0.324916    0.079819 -7.811609e-02         1
   APP       3.765279e+00           3.765279            3.765279           6      0.986544      0.708305      0.893548    0.237313 -1.531094e-01         1
  CRSP       3.631905e+00           3.631905            3.631905           6      0.985948      0.337487      0.454038    0.125014 -1.059892e-01         1
  VICR       2.901578e+00           2.901578            2.901578           8      0.985918      0.450923      0.656771    0.226350 -1.207664e-01         1
  ANVS       3.615378e+00           3.615378            3.615378           6      0.985859      0.617935      0.875250    0.242091 -2.274491e-01         1
   MDB       3.562011e+00           3.562011            3.562011           6      0.985547      0.366030      0.485513    0.136303 -1.323975e-01         1
  IREN       2.968032e+00           2.968032            2.968032           7      0.985330      0.918630      1.312880    0.442340 -3.958900e-01         1
  INCY       3.523017e+00           3.523017            3.523017           6      0.985291      0.254767      0.342946    0.097344 -8.007147e-02         1
   LUV       3.506858e+00           3.506858            3.506858           6      0.985177      0.320869      0.438284    0.124979 -1.224892e-01         1
   RCL       2.956023e+00           2.956023            2.956023           7      0.984999      0.374112      0.549251    0.185807 -9.170678e-02         1
  XPEL       3.352822e+00           3.352822            3.352822           6      0.983832      0.318440      0.440530    0.131391 -9.107987e-02         1
   BLD       2.850760e+00           2.850760            2.850760           7      0.981644      0.124006      0.192295    0.067454 -3.803169e-02         1
   JNJ       2.815259e+00           2.815259            2.815259           7      0.980306      0.081714      0.127247    0.045199 -4.498902e-02         1
  ASML       3.093556e+00           3.093556            3.093556           6      0.980014      0.186489      0.279503    0.090350 -9.030324e-02         1
  NVTS       2.984988e+00           2.984988            2.984988           6      0.977536      0.648619      1.048828    0.351368 -3.024522e-01         1
  ANAB       2.737973e+00           2.737973            2.737973           7      0.976965      0.352296      0.552148    0.201663 -1.116249e-01         1
    MD       2.953083e+00           2.953083            2.953083           6      0.976674      0.256807      0.385265    0.130462 -6.772126e-02         1
   SPY       2.666308e+00           2.666308            2.666308           9      0.976573      0.075491      0.125841    0.047197 -3.570808e-02         1
   VET       2.892972e+00           2.892972            2.892972           6      0.974856      0.313151      0.467647    0.161649 -9.189494e-02         1
    TK       2.662345e+00           2.662345            2.662345           7      0.973040      0.219724      0.353021    0.132598 -6.824545e-02         1
  PBBK       2.812575e+00           2.812575            2.812575           6      0.971971      0.032747      0.050815    0.018067 -1.686721e-02         1
     M       2.625707e+00           2.625707            2.625707           7      0.970870      0.311591      0.504638    0.192191 -1.557361e-01         1
  MSFT       2.714356e+00           2.714356            2.714356           6      0.967598      0.098234      0.164280    0.060523 -6.005323e-02         1
  PTGX       2.531220e+00           2.531220            2.531220           8      0.966861      0.135671      0.228994    0.090468 -4.857516e-02         1
  BLTE       2.666886e+00           2.666886            2.666886           6      0.965087      0.204046      0.323871    0.121442 -6.705093e-02         1
   MAR       2.506313e+00           2.506313            2.506313           8      0.964882      0.145274      0.251163    0.100212 -6.257059e-02         1
   PBR       2.661627e+00           2.661627            2.661627           6      0.964791      0.143633      0.231896    0.087126 -8.528738e-02         1
  NVDA       2.522765e+00           2.522765            2.522765           7      0.963672      0.216011      0.376580    0.149273 -1.438594e-01         1
  SATS       1.000000e+09        1000.000000            5.000000           5      0.962738      0.866828      0.684152    0.000000 -1.000089e-12         1
   BFH       1.000000e+09        1000.000000            5.000000           5      0.962738      0.607240      0.505155    0.000000 -1.000089e-12         1
    SN       1.000000e+09        1000.000000            5.000000           5      0.962738      0.695815      0.578451    0.000000 -1.000089e-12         1
   AER       3.805245e+01          38.052446            5.000000           5      0.962738      0.228643      0.220055    0.005783 -5.782951e-03         1
  QBTS       1.823523e+01          18.235226            5.000000           5      0.962738      1.794022      1.273999    0.069865 -6.986470e-02         1
  SNDK       1.814951e+01          18.149507            5.000000           5      0.962738      1.264380      0.985985    0.054326 -5.432572e-02         1
  TCMD       1.045271e+01          10.452712            5.000000           5      0.962738      0.159013      0.175828    0.016821 -1.364168e-02         1
   IBM       8.613263e+00           8.613263            5.000000           5      0.962738      0.231139      0.244913    0.028434 -2.843446e-02         1
  SNDL       8.418109e+00           8.418109            5.000000           5      0.962738      0.523792      0.527199    0.062627 -6.171111e-02         1
  ABUS       6.713902e+00           6.713902            5.000000           5      0.962738      0.242723      0.271143    0.040385 -4.001003e-02         1
   CEG       5.790281e+00           5.790281            5.000000           5      0.962738      0.333369      0.392694    0.067819 -6.359436e-02         1
  XPEV       4.931108e+00           4.931108            4.931108           5      0.962725      0.348219      0.411611    0.083472 -5.326287e-02         1
  RIGL       4.910928e+00           4.910928            4.910928           5      0.962721      0.522828      0.597088    0.121583 -1.215835e-01         1
  IONQ       4.866411e+00           4.866411            4.866411           5      0.962711      0.357613      0.426868    0.087717 -8.771713e-02         1
  XERS       4.630607e+00           4.630607            4.630607           5      0.962634      0.627786      0.711813    0.153719 -1.537191e-01         1
   EGO       4.415501e+00           4.415501            4.415501           5      0.962512      0.343836      0.416367    0.094297 -9.429674e-02         1
   HUT       4.183019e+00           4.183019            4.183019           5      0.962279      0.533790      0.710381    0.169825 -9.186512e-02         1
  RGLD       3.943891e+00           3.943891            3.943891           5      0.961848      0.170441      0.225456    0.057166 -3.376065e-02         1
   NYT       3.887736e+00           3.887736            3.887736           5      0.961703      0.138973      0.184027    0.047335 -4.733516e-02         1
  MDGL       3.452261e+00           3.452261            3.452261           5      0.959538      0.309570      0.425904    0.123369 -9.144885e-02         1
  LGND       2.421381e+00           2.421381            2.421381           9      0.957980      0.177690      0.306539    0.126597 -1.199901e-01         1
  CENX       2.452518e+00           2.452518            2.452518           7      0.957675      0.316258      0.545863    0.222573 -2.030858e-01         1
  FTXL       2.444547e+00           2.444547            2.444547           7      0.956931      0.074297      0.133118    0.054455 -5.360387e-02         1
  PACB       2.443210e+00           2.443210            2.443210           7      0.956805      0.355774      0.666774    0.272909 -1.707652e-01         1
    CM       2.398351e+00           2.398351            2.398351           8      0.954867      0.085755      0.146792    0.061205 -3.976724e-02         1
   SQM       2.417255e+00           2.417255            2.417255           7      0.954281      0.179220      0.324732    0.134339 -7.339066e-02         1
   VRT       2.406633e+00           2.406633            2.406633           7      0.953204      0.271404      0.541523    0.225013 -1.687222e-01         1
  IAUX       3.018264e+00           3.018264            3.018264           5      0.953202      0.384840      0.595952    0.197449 -1.876324e-01         1
  TEVA       2.405781e+00           2.405781            2.405781           7      0.953117      0.256133      0.447259    0.185910 -1.131995e-01         1
  OWLT       2.367144e+00           2.367144            2.367144          10      0.952480      0.504417      0.866757    0.366161 -2.256094e-01         1
  NTRB       2.448224e+00           2.448224            2.448224           6      0.949112      0.139467      0.246332    0.100617 -6.095709e-02         1
   AEO       2.426718e+00           2.426718            2.426718           6      0.947060      0.171802      0.311775    0.128476 -8.537178e-02         1
   NVT       2.333985e+00           2.333985            2.333985           7      0.945111      0.184760      0.352810    0.151162 -1.458221e-01         1
  IMAX       2.401974e+00           2.401974            2.401974           6      0.944573      0.165868      0.296946    0.123626 -1.080290e-01         1
   INO       2.317349e+00           2.317349            2.317349           6      0.934943      0.242491      0.479148    0.206765 -1.963005e-01         1
   JPM       2.250382e+00           2.250382            2.250382           7      0.934040      0.108023      0.204901    0.091052 -8.111008e-02         1
  ENTA       2.241952e+00           2.241952            2.241952           7      0.932808      0.225264      0.445141    0.198551 -1.285675e-01         1
  REGN       2.210442e+00           2.210442            2.210442           9      0.931260      0.195171      0.372837    0.168671 -1.278209e-01         1
  AVDL       2.284454e+00           2.284454            2.284454           6      0.930685      0.199539      0.392734    0.171916 -1.128155e-01         1
   RTX       2.200816e+00           2.200816            2.200816           9      0.929725      0.158366      0.290867    0.132163 -1.262124e-01         1
   VLO       2.220313e+00           2.220313            2.220313           7      0.929543      0.160510      0.306594    0.138086 -7.612901e-02         1
   KOD       2.179425e+00           2.179425            2.179425           8      0.925444      0.523899      1.100640    0.505014 -1.564941e-01         1
  EQNR       2.440091e+00           2.440091            2.440091           5      0.923445      0.089176      0.156905    0.064303 -6.351998e-02         1
  OKLO       2.145500e+00           2.145500            2.145500           8      0.919524      0.439864      1.073903    0.500537 -3.747922e-01         1
  NBIS       2.134131e+00           2.134131            2.134131           8      0.917446      0.319538      0.755938    0.354213 -3.051851e-01         1
  APPN       2.375239e+00           2.375239            2.375239           5      0.916880      0.222255      0.409676    0.172478 -9.172834e-02         1
   MDT       2.122225e+00           2.122225            2.122225           9      0.915972      0.090986      0.175209    0.082559 -2.967745e-02         1
  AMKR       2.116932e+00           2.116932            2.116932           9      0.914964      0.196217      0.402828    0.190289 -1.801345e-01         1
  JAZZ       2.101006e+00           2.101006            2.101006           6      0.900606      0.092490      0.186396    0.088717 -8.731075e-02         1
  CLSK       2.048151e+00           2.048151            2.048151           8      0.900093      0.333939      0.751412    0.366873 -2.908278e-01         1
   NEM       2.042232e+00           2.042232            2.042232           9      0.899537      0.182215      0.378865    0.185515 -1.091354e-01         1
  NRIM       2.088550e+01          20.885503            5.000000           4      0.897599      0.263125      0.256609    0.012286 -1.228647e-02         1
  SNBR       3.952654e+00           3.952654            3.952654           4      0.896749      0.561883      0.748146    0.189277 -1.892769e-01         1
  CTSH       2.081745e+00           2.081745            2.081745           6      0.896739      0.136764      0.278167    0.133622 -1.281271e-01         1
  DDOG       2.071688e+00           2.071688            2.071688           6      0.894660      0.194744      0.417549    0.201550 -1.914773e-01         1
  SBGI       2.071662e+00           2.071662            2.071662           6      0.894655      0.123630      0.257468    0.124281 -7.420275e-02         1
   XLK       2.007470e+00           2.007470            2.007470           9      0.891555      0.105349      0.229187    0.114167 -9.626042e-02         1
   FDN       1.996185e+00           1.996185            1.996185           8      0.888102      0.080266      0.176884    0.088611 -6.607091e-02         1
   CCJ       1.967570e+00           1.967570            1.967570           9      0.881726      0.274557      0.621435    0.315839 -2.884780e-01         1
   TMO       1.936744e+00           1.936744            1.936744           8      0.872882      0.085552      0.189620    0.097906 -5.395049e-02         1
  CRWV       2.574299e+00           2.574299            2.574299           4      0.869824      0.311501      0.641428    0.249166 -2.491662e-01         1
  HTHT       1.920815e+00           1.920815            1.920815           8      0.868518      0.089992      0.199026    0.103615 -4.728246e-02         1
  RDVT       1.879900e+00           1.879900            1.879900           7      0.854307      0.120388      0.284613    0.151398 -1.186301e-01         1
  GLSI       1.865820e+00           1.865820            1.865820           8      0.852490      0.149819      0.365961    0.196140 -1.775304e-01         1
  PGEN       1.863024e+00           1.863024            1.863024           7      0.849209      0.221316      0.699157    0.375281 -1.926286e-01         1
  TNDM       2.287137e+00           2.287137            2.287137           4      0.842485      0.206853      0.444638    0.194408 -1.944081e-01         1
   ASC       1.805729e+00           1.805729            1.805729           9      0.833951      0.122669      0.300961    0.166670 -1.434147e-01         1
  ANET       1.795561e+00           1.795561            1.795561           8      0.829773      0.199990      0.546911    0.304591 -2.624842e-01         1
   EIS       1.791107e+00           1.791107            1.791107           9      0.828976      0.070471      0.165089    0.092172 -3.727617e-02         1
  GSAT       1.789172e+00           1.789172            1.789172           8      0.827580      0.221601      0.650128    0.363368 -1.690138e-01         1
  TSLA       1.802364e+00           1.802364            1.802364           6      0.821895      0.126892      0.325570    0.180635 -1.726565e-01         1
   HCA       1.755831e+00           1.755831            1.755831           7      0.813397      0.059553      0.148106    0.084351 -5.948063e-02         1
  META       1.742943e+00           1.742943            1.742943           7      0.808687      0.117764      0.339981    0.195061 -1.808558e-01         1
    BE       1.707096e+00           1.707096            1.707096          12      0.798525      0.339954      1.184935    0.694123 -5.061819e-01         1
  IMMX       1.715119e+00           1.715119            1.715119           7      0.798221      0.127015      0.410033    0.239070 -1.776698e-01         1
  ILMN       1.705328e+00           1.705328            1.705328           9      0.797535      0.150158      0.430521    0.252456 -2.249599e-01         1
  JKHY       1.786446e+00           1.786446            1.786446           5      0.792791      0.071823      0.175061    0.097994 -9.559325e-02         1
  AZTA       1.651174e+00           1.651174            1.651174           8      0.774997      0.103437      0.326509    0.197743 -1.878866e-01         1
  ALAB       1.622604e+00           1.622604            1.622604           6      0.752990      0.142237      0.647403    0.398990 -3.561577e-01         1
  PVLA       1.604891e+00           1.604891            1.604891           7      0.752848      0.130469      0.483964    0.301556 -2.336488e-01         1
  FROG       1.571480e+00           1.571480            1.571480           7      0.737915      0.106214      0.380253    0.241971 -1.201557e-01         1
   ICE       1.571875e+00           1.571875            1.571875           6      0.730611      0.050154      0.157985    0.100507 -9.752107e-02         1
   LMT       1.541796e+00           1.541796            1.541796           7      0.724221      0.070417      0.223128    0.144719 -1.188406e-01         1
   HIT       2.191207e+00           2.191207            2.191207           3      0.717876      0.056783      0.114283    0.052155 -5.215541e-02         1
  LYFT       1.507754e+00           1.507754            1.507754           7      0.708052      0.123831      0.560229    0.371565 -2.844565e-01         1
  MLYS       1.500483e+00           1.500483            1.500483           8      0.706816      0.094309      0.462190    0.308027 -2.244888e-01         1
   CVI       1.499144e+00           1.499144            1.499144           8      0.706166      0.095112      0.377950    0.252111 -1.280650e-01         1
   AIQ       1.470297e+00           1.470297            1.470297          10      0.692890      0.052086      0.197987    0.134658 -7.467640e-02         1
   GLD       1.465566e+00           1.465566            1.465566          10      0.690536      0.053622      0.183409    0.125145 -1.201381e-01         1
  DELL       1.449069e+00           1.449069            1.449069           8      0.681377      0.093944      0.423754    0.292432 -2.163191e-01         1
   MRK       1.431592e+00           1.431592            1.431592           9      0.673206      0.039303      0.152950    0.106839 -9.338850e-02         1
  VERA       1.427992e+00           1.427992            1.427992           6      0.661210      0.062844      0.440791    0.308678 -2.753888e-01         1
  PTCT       1.412354e+00           1.412354            1.412354           6      0.653232      0.077413      0.431712    0.305668 -2.075961e-01         1
    GM       1.362328e+00           1.362328            1.362328           6      0.627295      0.040280      0.184988    0.135788 -1.131221e-01         1
   IYW       1.342792e+00           1.342792            1.342792           9      0.626954      0.040072      0.210792    0.156980 -1.233704e-01         1
  EPAM       1.337041e+00           1.337041            1.337041           7      0.621052      0.056346      0.340310    0.254525 -2.318288e-01         1
  OMER       1.319778e+00           1.319778            1.319778           7      0.611874      0.016575      0.730708    0.553660 -2.848760e-01         1
    MP       1.311437e+00           1.311437            1.311437           8      0.609596      0.038162      0.541028    0.412546 -3.620665e-01         1
  AVGO       1.297565e+00           1.297565            1.297565          10      0.603028      0.045407      0.357902    0.275826 -2.255711e-01         1
  ANDE       1.284433e+00           1.284433            1.284433           7      0.592979      0.033660      0.223467    0.173981 -1.140888e-01         1
  ARGX       1.272923e+00           1.272923            1.272923           6      0.579874      0.033387      0.234754    0.184421 -1.765408e-01         1
  PLTR       1.251368e+00           1.251368            1.251368           7      0.575232      0.018835      0.347851    0.277976 -2.064227e-01         1
  VSCO       1.243921e+00           1.243921            1.243921           8      0.573357      0.028898      0.381878    0.306995 -2.625821e-01         1
  ALNY       1.240660e+00           1.240660            1.240660           8      0.571604      0.032720      0.316193    0.254859 -2.322695e-01         1
  NAIL       1.229932e+00           1.229932            1.229932          10      0.566674      0.017780      0.632696    0.514416 -2.883542e-01         1
   NKE       1.103328e+00           1.103328            1.103328           8      0.498693      0.001884      0.253731    0.229969 -1.984725e-01         1
   TJX       1.089639e+00           1.089639            1.089639           8      0.491618      0.005617      0.120025    0.110151 -6.976349e-02         1
   UHS       1.065285e+00           1.065285            1.065285           7      0.477159     -0.000397      0.174836    0.164122 -9.828251e-02         1
  JACK       1.054608e+00           1.054608            1.054608           8      0.473758     -0.020400      0.279694    0.265212 -1.645914e-01         1
  ABBV       1.043084e+00           1.043084            1.043084           8      0.467967     -0.000348      0.124454    0.119313 -8.010769e-02         1
   VTR       1.043776e+00           1.043776            1.043776           7      0.466327     -0.000293      0.125608    0.120340 -7.215974e-02         1
     F       1.041856e+00           1.041856            1.041856           7      0.465367     -0.004865      0.177958    0.170808 -1.608287e-01         1
  ROST       1.047816e+00           1.047816            1.047816           6      0.461923     -0.001416      0.138454    0.132136 -7.352480e-02         1
  ACWX       9.916477e-01           0.991648            0.991648          11      0.443534     -0.003438      0.087531    0.088268 -8.483876e-02         1
   LLY       9.905786e-01           0.990579            0.990579           7      0.440246     -0.033583      0.292686    0.295470 -1.495844e-01         1
   AEM       9.802502e-01           0.980250            0.980250           8      0.437253     -0.014029      0.171189    0.174638 -1.144476e-01         1
  WEAV       9.770441e-01           0.977044            0.977044           8      0.435728     -0.051260      0.356267    0.364638 -2.168853e-01         1
  AAPL       9.602707e-01           0.960271            0.960271           9      0.428407     -0.013032      0.144246    0.150214 -8.741155e-02         1
   NEO       9.289045e-01           0.928904            0.928904           7      0.411469     -0.108411      0.416631    0.448518 -3.789446e-01         1
  SCHD       8.611219e-01           0.861122            0.861122           8      0.383767     -0.011425      0.062622    0.072721 -6.492011e-02         1
  CBRL       8.546208e-01           0.854621            0.854621           9      0.381613     -0.117646      0.408760    0.478294 -3.976986e-01         1
  ADBE       8.538764e-01           0.853876            0.853876           8      0.380740     -0.055360      0.218179    0.255515 -2.319229e-01         1
   CNQ       7.920317e-01           0.792032            0.792032           7      0.354209     -0.054826      0.167768    0.211819 -1.276170e-01         1
   RGC       7.777629e-01           0.777763            0.777763           5      0.325317     -0.255497      0.503956    0.647956 -5.246344e-01         1
   IQV       6.062913e-01           0.606291            0.606291           9      0.294681     -0.143133      0.203168    0.335100 -1.821339e-01         1
   NOW       5.554619e-01           0.555462            0.555462           7      0.278711     -0.243450      0.282442    0.508482 -3.656562e-01         1
  bidu       5.474778e-01           0.547478            0.547478           8      0.278337     -0.121260      0.138344    0.252694 -1.551415e-01         1
   WGS       4.346861e-01           0.434686            0.434686           8      0.252562     -0.377713      0.295551    0.679918 -4.159747e-01         1
  RGEN       3.817564e-01           0.381756            0.381756           9      0.242836     -0.224992      0.142368    0.372928 -2.716346e-01         1
  NFLX       3.458901e-01           0.345890            0.345890           7      0.234471     -0.282749      0.154505    0.446689 -3.787377e-01         1
   NFE       9.766245e-02           0.097662            0.097662           6      0.197419     -0.804023      0.138300    1.416099 -8.283975e-01         1
Saved per-ticker CSV to: output\per_ticker_20260118_0834.csv
real 866.754    user 0.045      sys 0.030       pcpu 0.00

=== OVERALL (ALL FILES) METRICS (BEST PARAMS) ===
Avg Profit Factor (raw):      407.5503889641745
Avg Profit Factor (diag):     2.870046
Avg Trades / Ticker:          5.957
Mean Ticker Score (eligible): 0.773194
Coverage (eligible/total):    281/281 = 1.000
Penalty enabled:              enabled (soft)
Soft penalty center/k:        0.01 / 10.0
Tail-protection ret floor/k:  -0.15 / 2.0
Max-trades penalty:           max_trades=100, k=0.1
PF-floor penalty:             pf_floor=1.0, k=5.0
cooldown_bars:                1 (opt=False) [loss-only]
time_stop_bars:               15 (0=disabled) (opt=True)
num_neg (return<=0):          57
commission_rate_per_side:     0.000600
PF ROC baseline/k:            1.150 / 2.500
Trades ROC baseline/k:        3.000 / 1.200
weight_pf:                    0.700
score_power:                  1.500
min_trades gate:              1
loss_floor (scoring):         0.001
threshold_mode:               fixed
Score (single fill 'next_open'): 0.673830

=== BEST PARAMS ===
fp: 10
reg_ratio: 2.056250837299421
sp: 21
ts: 15
vl: 9
vs: 5
best_score (OPTUNA objective): 0.6738304377683826
fill_mode (final report): next_open

=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===
ticker  profit_factor  profit_factor_diag  num_trades  ticker_score  total_return  gross_profit  gross_loss    maxdd  eligible
  GLUE   12025.268400            5.000000           7      0.996260      1.984440      1.202527    0.000000 0.000000         1
  SNDK    9048.991273            5.000000           4      0.897599      1.221788      0.904899    0.000000 0.000000         1
  AXTI    8881.280837            5.000000           6      0.987986      1.206379      0.888128    0.000000 0.000000         1
    NG    8204.369871            5.000000           6      0.987986      1.131174      0.820437    0.000000 0.000000         1
  COGT    7817.355022            5.000000           5      0.962738      1.048629      0.781736    0.000000 0.000000         1
  APLD    7612.393412            5.000000           5      0.962738      0.980811      0.761239    0.000000 0.000000         1
    MU    7575.021263            5.000000           7      0.996260      1.040880      0.757502    0.000000 0.000000         1
  COHR    6468.471053            5.000000           5      0.962738      0.823877      0.646847    0.000000 0.000000         1
  LABU    6376.505908            5.000000           6      0.987986      0.813567      0.637651    0.000000 0.000000         1
  SATS    6103.426822            5.000000           4      0.897599      0.738702      0.610343    0.000000 0.000000         1
  INCY    4518.754097            5.000000           5      0.962738      0.538898      0.451875    0.000000 0.000000         1
  CRSP    4441.254137            5.000000           5      0.962738      0.512904      0.444125    0.000000 0.000000         1
   EGO    3843.853591            5.000000           4      0.897599      0.439238      0.384385    0.000000 0.000000         1
   LIT    3322.750845            5.000000           7      0.996260      0.381870      0.332275    0.000000 0.000000         1
    SN    3312.829729            5.000000           3      0.783597      0.362022      0.331283    0.000000 0.000000         1
   BFH    3017.975973            5.000000           4      0.897599      0.334090      0.301798    0.000000 0.000000         1
 GOOGL    2553.655499            5.000000           7      0.996260      0.280794      0.255366    0.000000 0.000000         1
    SU    2528.439335            5.000000           6      0.987986      0.276961      0.252844    0.000000 0.000000         1
   XBI    2132.667322            5.000000           6      0.987986      0.231720      0.213267    0.000000 0.000000         1
   CMI    2130.181041            5.000000           6      0.987986      0.229048      0.213018    0.000000 0.000000         1
  XPEV     432.992160            5.000000           4      0.897599      0.351342      0.322193    0.000744 0.000744         1
    SA     343.418740            5.000000           4      0.897599      0.469008      0.412102    0.001200 0.001200         1
  TCMD     239.324292            5.000000           5      0.962738      0.312484      0.287189    0.001200 0.001200         1
   MLI     146.046866            5.000000           6      0.987986      0.156598      0.149423    0.001023 0.001023         1
  QBTS     111.690842            5.000000           5      0.962738      1.424282      1.013998    0.009079 0.009079         1
  GOSS      99.155795            5.000000           5      0.962738      1.083403      0.835431    0.008425 0.008425         1
    GE      58.969754            5.000000           5      0.962738      0.234709      0.221264    0.003752 0.003752         1
   HUT      58.898757            5.000000           5      0.962738      0.902867      0.765551    0.012998 0.012998         1
  PAAS      53.956258            5.000000           6      0.987986      0.570954      0.490648    0.009093 0.009093         1
  RGLD      48.947774            5.000000           5      0.962738      0.302299      0.282640    0.005774 0.004352         1
    CM      35.360948            5.000000           6      0.987986      0.209561      0.200156    0.005660 0.005660         1
  SNDL      31.554499            5.000000           4      0.897599      0.594279      0.525194    0.016644 0.016644         1
    TD      30.433878            5.000000           6      0.987986      0.173659      0.169358    0.005565 0.003285         1
   BLD      28.196835            5.000000           5      0.962738      0.311388      0.294425    0.010442 0.010442         1
  ABUS      25.667838            5.000000           5      0.962738      0.311122      0.296200    0.011540 0.011540         1
    GH      24.793738            5.000000           5      0.962738      0.541360      0.483255    0.019491 0.019491         1
   GLW      21.909511            5.000000           5      0.962738      0.178207      0.177363    0.008095 0.008095         1
  TARS      21.385003            5.000000           5      0.962738      0.580252      0.512090    0.023946 0.023946         1
  PASG      18.165812            5.000000           6      0.987986      2.866447      2.921725    0.160836 0.152257         1
   TEL      16.141339            5.000000           5      0.962738      0.158110      0.161970    0.010034 0.009607         1
  NRIM      15.934482            5.000000           3      0.783597      0.151999      0.157496    0.009884 0.009884         1
   CDE      15.495623            5.000000           8      0.998818      0.588687      0.534570    0.034498 0.034498         1
   AER      15.426513            5.000000           4      0.897599      0.084864      0.089211    0.005783 0.005783         1
   AMG      13.116553            5.000000           8      0.998818      0.275379      0.270953    0.020657 0.020657         1
   GLD      12.885983            5.000000           7      0.996260      0.168178      0.171807    0.013333 0.013333         1
  TERN      12.281391            5.000000           6      0.987986      1.007868      0.854231    0.069555 0.067182         1
  PSNL      12.265280            5.000000           5      0.962738      0.874429      0.761084    0.062052 0.062052         1
   JNJ      11.753537            5.000000           5      0.962738      0.093571      0.099977    0.008506 0.004330         1
  VSCO      11.308265            5.000000           8      0.998818      0.243529      0.258414    0.022852 0.018882         1
  LGND      10.920792            5.000000           8      0.998818      0.288446      0.289507    0.026510 0.026510         1
  NTRA      10.155757            5.000000           5      0.962738      0.314139      0.322039    0.031710 0.031710         1
  MDGL       9.978372            5.000000           4      0.897599      0.423130      0.423868    0.042479 0.042479         1
   FDX       9.884892            5.000000           7      0.996260      0.344991      0.343395    0.034739 0.034739         1
  DIBS       9.725845            5.000000           6      0.987986      0.352656      0.368397    0.037878 0.022629         1
  BBAI       9.492257            5.000000           6      0.987986      0.723636      0.698496    0.073586 0.072242         1
   TSM       9.475650            5.000000           6      0.987986      0.143473      0.155414    0.016401 0.016348         1
  PVLA       9.163128            5.000000           5      0.962738      0.541483      0.526938    0.057506 0.057506         1
   AMD       8.737458            5.000000           6      0.987986      0.466231      0.466236    0.053361 0.027287         1
  REMX       8.548198            5.000000           6      0.987986      0.367229      0.372296    0.043553 0.043553         1
  SIMO       8.483353            5.000000           6      0.987986      0.235682      0.254520    0.030002 0.018512         1
   SQM       8.019312            5.000000           5      0.962738      0.305862      0.327719    0.040866 0.040866         1
    PL       7.975297            5.000000           6      0.987986      0.814527      0.784950    0.098423 0.096013         1
  IAUX       7.531019            5.000000           5      0.962738      0.580634      0.590672    0.078432 0.068616         1
    GS       7.190535            5.000000           5      0.962738      0.100186      0.114844    0.015971 0.014563         1
   NEM       6.960715            5.000000           7      0.996260      0.517282      0.516856    0.074253 0.074253         1
   STX       6.775641            5.000000           6      0.987986      0.172216      0.196142    0.028948 0.017885         1
  OLMA       6.763141            5.000000           6      0.987986      0.866290      0.929181    0.137389 0.069291         1
  PTGX       6.477959            5.000000           7      0.996260      0.261013      0.289374    0.044671 0.035709         1
    AG       6.433033            5.000000           6      0.987986      0.607246      0.635277    0.098752 0.098752         1
  RGTI       6.301252            5.000000           6      0.987986      0.687511      0.713275    0.113196 0.113196         1
  BBIO       6.257992            5.000000           7      0.996260      0.223069      0.253159    0.040454 0.029852         1
   FTI       6.225936            5.000000           6      0.987986      0.206424      0.232355    0.037321 0.024808         1
  RKLB       6.172223            5.000000           7      0.996260      0.848972      0.855366    0.138583 0.134764         1
  LRCX       5.658462            5.000000           7      0.996260      0.314733      0.353706    0.062509 0.049531         1
     U       5.621853            5.000000           6      0.987986      0.512193      0.559691    0.099556 0.079414         1
   XME       5.547497            5.000000           7      0.996260      0.260866      0.298364    0.053784 0.053784         1
   IBM       5.397142            5.000000           4      0.897599      0.129626      0.155470    0.028806 0.028806         1
  RYTM       5.354347            5.000000           7      0.996260      0.298092      0.340578    0.063608 0.056805         1
   VET       5.338402            5.000000           5      0.962738      0.298910      0.347453    0.065086 0.064136         1
   IMO       5.241923            5.000000           7      0.996260      0.165052      0.197964    0.037765 0.029565         1
     M       5.192235            5.000000           6      0.987986      0.335958      0.386699    0.074476 0.054615         1
  DSGN       5.118631            5.000000           6      0.987986      0.247272      0.313839    0.061313 0.057817         1
  AREC       5.117007            5.000000           8      0.998818      2.157201      1.820260    0.355728 0.251586         1
  BLOK       5.015497            5.000000           5      0.962738      0.195402      0.235112    0.046877 0.046574         1
  EOSE       4.993694            4.993694           6      0.987985      0.844096      0.911568    0.182544 0.182544         1
  SPHR       4.954425            4.954425           5      0.962730      0.343809      0.407982    0.082347 0.082347         1
  UTHR       4.932081            4.932081           5      0.962726      0.342735      0.434844    0.088166 0.056714         1
  DBRG       4.925408            4.925408           6      0.987972      0.193646      0.252967    0.051360 0.045925         1
   ZIM       4.766735            4.766735           5      0.962684      0.392552      0.466393    0.097843 0.097843         1
   NGL       4.765929            4.765929           5      0.962684      0.311599      0.374012    0.078476 0.078476         1
   OGI       4.714239            4.714239           6      0.987914      0.376837      0.450027    0.095461 0.095348         1
  MRNA       4.684765            4.684765           6      0.987903      0.428518      0.495812    0.105835 0.064635         1
   KOD       4.663551            4.663551           7      0.996168      0.669970      0.765420    0.164128 0.115486         1
  IONQ       4.524750            4.524750           5      0.962582      0.237930      0.308633    0.068210 0.058820         1
  AMAT       4.445434            4.445434           6      0.987779      0.174413      0.217826    0.049000 0.026786         1
  CENX       4.354577            4.354577           7      0.995981      0.472305      0.559800    0.128554 0.128554         1
   CRS       4.334146            4.334146           6      0.987690      0.045298      0.059290    0.013680 0.013443         1
  APPN       4.268020            4.268020           4      0.897249      0.312142      0.391498    0.091728 0.091728         1
  ARKF       4.189930            4.189930           5      0.962288      0.084666      0.111377    0.026582 0.026406         1
   CAT       4.071548            4.071548           9      0.998958      0.198865      0.249634    0.061312 0.034525         1
  IREN       4.052520            4.052520           6      0.987318      0.878072      1.018493    0.251323 0.237545         1
  PALL       4.032240            4.032240           5      0.962038      0.189041      0.245367    0.060851 0.060851         1
   AEO       3.757993            3.757993           5      0.961281      0.224564      0.307292    0.081770 0.073947         1
  NVGS       3.757200            3.757200           6      0.986514      0.189391      0.250516    0.066676 0.066676         1
  PTCT       3.747057            3.747057           6      0.986474      0.318319      0.417109    0.111317 0.111317         1
  VICR       3.731786            3.731786           7      0.994682      0.386086      0.501141    0.134290 0.125446         1
   PBF       3.675097            3.675097           6      0.986163      0.463270      0.610919    0.166232 0.162652         1
   WDC       3.636208            3.636208           8      0.996795      0.329761      0.423545    0.116480 0.082175         1
  ANVS       3.405474            3.405474           5      0.959134      0.481454      0.704758    0.206948 0.116923         1
    RY       3.389496            3.389496           5      0.958985      0.077043      0.108902    0.032129 0.032129         1
  ANDE       3.379032            3.379032           7      0.992361      0.201479      0.278074    0.082294 0.065195         1
   DBI       3.372264            3.372264           5      0.958817      0.346739      0.513173    0.152175 0.147233         1
  ANAB       3.349836            3.349836           6      0.983800      0.275458      0.385114    0.114965 0.114965         1
  HTHT       3.334359            3.334359           5      0.958423      0.130068      0.186153    0.055829 0.041350         1
  SNBR       3.313191            3.313191           5      0.958186      0.490636      0.740375    0.223463 0.177383         1
  ENVA       3.234016            3.234016           6      0.982380      0.271580      0.378691    0.117096 0.081589         1
   WWD       3.223773            3.223773           7      0.990491      0.095573      0.138757    0.043042 0.026021         1
  PLAB       3.189033            3.189033           5      0.956516      0.206512      0.303641    0.095214 0.091271         1
  OWLT       3.162024            3.162024           9      0.992852      0.599316      0.796680    0.251953 0.168452         1
  BOOT       3.152276            3.152276           5      0.955915      0.091351      0.139870    0.044371 0.040823         1
   MRK       3.135076            3.135076           7      0.989054      0.129196      0.189825    0.060549 0.059688         1
   PBR       3.133217            3.133217           5      0.955581      0.159059      0.231896    0.074012 0.072950         1
  ASTL       3.124405            3.124405           4      0.890452      0.133250      0.203029    0.064982 0.063931         1
  NAIL       3.042925            3.042925           7      0.987189      0.424993      0.646496    0.212459 0.170480         1
    HL       2.968358            2.968358           7      0.985339      0.410379      0.618505    0.208366 0.173777         1
  CRWV       2.959005            2.959005           4      0.886805      0.431522      0.737284    0.249166 0.249166         1
  NEGG       2.956429            2.956429           7      0.985010      0.768637      1.348297    0.456056 0.283218         1
  ARGX       2.928559            2.928559           5      0.950820      0.138830      0.212106    0.072427 0.064748         1
   CLF       2.927010            2.927010           7      0.984158      0.498134      0.734117    0.250808 0.139009         1
   RCL       2.919987            2.919987           6      0.975706      0.217571      0.333162    0.114097 0.098157         1
  RVMD       2.897847            2.897847           6      0.975014      0.282554      0.427788    0.147623 0.143580         1
  ARKW       2.774128            2.774128           5      0.945285      0.090197      0.144218    0.051987 0.051454         1
   AXP       2.764836            2.764836           5      0.944882      0.089326      0.140645    0.050869 0.033488         1
   INO       2.734000            2.734000           6      0.968557      0.324955      0.526840    0.192699 0.140005         1
  BABA       2.726083            2.726083           7      0.976394      0.151757      0.247042    0.090622 0.058089         1
  INTC       2.675275            2.675275           6      0.965551      0.284206      0.469971    0.175672 0.137661         1
   AMX       2.672613            2.672613           6      0.965405      0.072962      0.117604    0.044003 0.023233         1
  AXSM       2.670967            2.670967           6      0.965314      0.201086      0.324016    0.121310 0.117979         1
  MKSI       2.655728            2.655728           8      0.975200      0.250958      0.398458    0.150037 0.124193         1
   CLS       2.654897            2.654897           6      0.964407      0.324216      0.555892    0.209383 0.209383         1
  TTMI       2.643534            2.643534           6      0.963744      0.174956      0.289789    0.109622 0.106794         1
   TMO       2.634117            2.634117           6      0.963181      0.071192      0.116383    0.044183 0.044183         1
  ARKK       2.631226            2.631226           7      0.971209      0.159964      0.262366    0.099712 0.084332         1
    TK       2.628264            2.628264           5      0.937795      0.113599      0.189539    0.072116 0.071852         1
  XERS       2.622496            2.622496           5      0.937442      0.277505      0.472388    0.180129 0.176069         1
   GEV       2.618047            2.618047           6      0.962191      0.223587      0.383205    0.146371 0.138648         1
  ASTS       2.528135            2.528135           6      0.955908      0.457286      0.846799    0.334950 0.308151         1
   APH       2.481722            2.481722           7      0.960286      0.124210      0.210562    0.084845 0.073079         1
    MD       2.330192            2.330192           7      0.944651      0.204126      0.369939    0.158759 0.067186         1
  DLTR       2.323618            2.323618          10      0.947351      0.204667      0.366498    0.157727 0.099611         1
  OMER       2.308057            2.308057           5      0.908998      0.289217      0.694426    0.300870 0.240470         1
  BLTE       2.265740            2.265740           6      0.928124      0.150527      0.275155    0.121442 0.067051         1
   ATI       2.228258            2.228258           4      0.834388      0.096722      0.186833    0.083847 0.064953         1
  XPEL       2.178544            2.178544           5      0.890134      0.146492      0.286241    0.131391 0.080756         1
   APP       2.147436            2.147436           5      0.884782      0.203603      0.430308    0.200382 0.151667         1
   LUV       2.125049            2.125049           5      0.880713      0.119067      0.245862    0.115697 0.100109         1
  SOFI       2.122231            2.122231           7      0.912734      0.146439      0.308595    0.145411 0.139638         1
  FUTU       2.078183            2.078183           7      0.904017      0.158522      0.336658    0.161996 0.137203         1
  LYFT       2.067862            2.067862           6      0.893859      0.226717      0.487094    0.235554 0.160603         1
  ZURA       2.063896            2.063896           6      0.893021      0.282022      0.673594    0.326370 0.282173         1
    CX       2.018938            2.018938           5      0.858751      0.100075      0.211278    0.104648 0.102372         1
  IMAX       2.013346            2.013346           5      0.857463      0.101401      0.224492    0.111502 0.109309         1
  ASML       1.995339            1.995339           5      0.853223      0.082720      0.179241    0.089830 0.089830         1
   BAC       1.992897            1.992897           6      0.876892      0.051398      0.106375    0.053377 0.035163         1
   HCA       1.982672            1.982672           7      0.882330      0.074780      0.158339    0.079861 0.063968         1
   ERO       1.975761            1.975761           7      0.880603      0.133889      0.292358    0.147972 0.121382         1
   RTX       1.909555            1.909555           6      0.855035      0.065173      0.144018    0.075419 0.073785         1
    BE       1.887495            1.887495          10      0.859949      0.408838      1.093551    0.579366 0.476505         1
  SBGI       1.883208            1.883208           5      0.823450      0.098051      0.236008    0.125322 0.079913         1
  PACB       1.865753            1.865753           7      0.850043      0.231778      0.640011    0.343031 0.165486         1
  PGEN       1.864804            1.864804           6      0.841908      0.165029      0.531197    0.284854 0.192629         1
  DINO       1.834460            1.834460           6      0.832433      0.057269      0.137443    0.074923 0.062796         1
   NEO       1.827444            1.827444           6      0.830175      0.147798      0.421931    0.230886 0.216185         1
  ENTA       1.808241            1.808241           6      0.823864      0.177464      0.490151    0.271065 0.142497         1
   NSC       1.784084            1.784084           5      0.791983      0.022868      0.053596    0.030041 0.030041         1
   VTR       1.777333            1.777333           5      0.789657      0.030760      0.074845    0.042111 0.030767         1
  AVDL       1.762547            1.762547           5      0.784481      0.056902      0.143058    0.081166 0.059101         1
   EIS       1.753730            1.753730           7      0.812635      0.044636      0.107626    0.061370 0.031384         1
   XSD       1.748427            1.748427           6      0.802979      0.046810      0.115545    0.066085 0.046123         1
  FTAI       1.715893            1.715893           6      0.790834      0.105136      0.310926    0.181204 0.174524         1
   ASC       1.700534            1.700534           7      0.792574      0.092767      0.250449    0.147277 0.102372         1
   LLY       1.687371            1.687371           6      0.779734      0.086706      0.246186    0.145899 0.135162         1
   AEM       1.667158            1.667158           8      0.781593      0.097062      0.267025    0.160168 0.109999         1
  TSLA       1.644915            1.644915           7      0.770031      0.104836      0.319232    0.194072 0.146293         1
  ILMN       1.638097            1.638097           8      0.769505      0.121376      0.372539    0.227422 0.211989         1
  AMRC       1.562634            1.562634           7      0.733875      0.122832      0.517983    0.331481 0.251544         1
  DELL       1.553158            1.553158           7      0.729509      0.057819      0.188622    0.121444 0.105174         1
  GSAT       1.551538            1.551538           8      0.731064      0.133275      0.521802    0.336312 0.165062         1
   TJX       1.548694            1.548694           6      0.719990      0.038403      0.117481    0.075858 0.063144         1
  NBIS       1.548414            1.548414           7      0.727308      0.114669      0.598077    0.386251 0.342656         1
   HIT       1.534588            1.534588           6      0.713412      0.082389      0.603884    0.393516 0.298004         1
   LMT       1.520827            1.520827           6      0.706915      0.056709      0.187075    0.123009 0.119262         1
  SNOW       1.513979            1.513979           8      0.713330      0.064845      0.237214    0.156683 0.078771         1
  VERA       1.511307            1.511307           4      0.622036      0.081346      0.442270    0.292640 0.274019         1
  DDOG       1.477665            1.477665           6      0.686047      0.074360      0.297823    0.201550 0.191477         1
  MLYS       1.441497            1.441497           7      0.675306      0.052797      0.348945    0.242071 0.159679         1
    MP       1.426920            1.426920           7      0.667903      0.069587      0.499034    0.349728 0.320082         1
  NVDA       1.424240            1.424240           6      0.659302      0.044516      0.192663    0.135274 0.131322         1
  REGN       1.409127            1.409127           7      0.658783      0.056274      0.254154    0.180362 0.129664         1
  MSFT       1.386000            1.386000           5      0.617831      0.018408      0.076809    0.055418 0.055315         1
  CSIQ       1.375460            1.375460           6      0.634159      0.050458      0.529219    0.384758 0.191390         1
    ON       1.363681            1.363681           6      0.628004      0.028524      0.134009    0.098270 0.080261         1
   UHS       1.359784            1.359784           7      0.633071      0.051668      0.245812    0.180773 0.123520         1
   JPM       1.358528            1.358528           6      0.625302      0.023450      0.103111    0.075899 0.056498         1
  ALAB       1.346519            1.346519           4      0.542103      0.036624      0.415641    0.308678 0.288930         1
  RIGL       1.341389            1.341389           4      0.539515      0.041700      0.274820    0.204877 0.121583         1
  ADPT       1.333786            1.333786           6      0.612267      0.032405      0.278694    0.208950 0.203742         1
  HOOD       1.319273            1.319273           7      0.611605      0.046819      0.303814    0.230289 0.129448         1
  OKLO       1.308028            1.308028           7      0.605606      0.016682      0.510933    0.390613 0.361277         1
   MDT       1.306502            1.306502           8      0.606957      0.018445      0.089899    0.068809 0.053404         1
  CTSH       1.298613            1.298613           5      0.572321      0.026845      0.149310    0.114977 0.110861         1
  RCUS       1.286872            1.286872           7      0.594286      0.032569      0.360226    0.279924 0.208003         1
  AZTA       1.279340            1.279340           5      0.562161      0.024047      0.155603    0.121627 0.116865         1
   TAN       1.266956            1.266956           6      0.576684      0.030374      0.216670    0.171016 0.097742         1
  TEVA       1.249526            1.249526           6      0.567361      0.030728      0.231443    0.185224 0.124173         1
   CCJ       1.242872            1.242872           8      0.572794      0.039696      0.377046    0.303366 0.205609         1
   MDB       1.238197            1.238197           5      0.540431      0.025821      0.255970    0.206728 0.196501         1
   RIG       1.218260            1.218260           8      0.559567      0.026002      0.441775    0.362628 0.181682         1
   LVS       1.208490            1.208490           5      0.524763      0.015191      0.156885    0.129819 0.125686         1
   WGS       1.172290            1.172290           6      0.526189      0.003223      0.400690    0.341801 0.182786         1
  ABBV       1.171318            1.171318           7      0.532384      0.012362      0.118695    0.101335 0.086229         1
    GM       1.154902            1.154902           5      0.496709      0.011921      0.156822    0.135788 0.103011         1
   VLO       1.119841            1.119841           6      0.498658      0.007540      0.158448    0.141492 0.076129         1
  PBBK       1.103978            1.103978           6      0.490444      0.003074      0.037811    0.034250 0.023321         1
  GRAL       1.047722            1.047722           6      0.461875     -0.021623      0.230908    0.220391 0.183557         1
  EPAM       1.035792            1.035792           6      0.455946     -0.007896      0.194299    0.187585 0.132442         1
  NAMS       1.035659            1.035659           7      0.462279     -0.023071      0.258735    0.249827 0.138147         1
  TNDM       1.032996            1.032996           5      0.435130     -0.038636      0.277222    0.268366 0.253103         1
   XLK       1.022317            1.022317           6      0.449309     -0.001276      0.074707    0.073076 0.063557         1
   CEG       1.016814            1.016814           5      0.427300     -0.013058      0.152474    0.149953 0.124869         1
  CBRL       1.010748            1.010748           6      0.443665     -0.030707      0.280531    0.277548 0.255377         1
   SPY       1.007678            1.007678           6      0.442176     -0.000001      0.013697    0.013593 0.013534         1
  ANET       0.985814            0.985814           7      0.437964     -0.032601      0.270096    0.273982 0.161202         1
  ROST       0.966959            0.966959           6      0.422786     -0.011637      0.126279    0.130594 0.073690         1
   MAR       0.959577            0.959577           6      0.419346     -0.010537      0.096767    0.100843 0.066302         1
  BITF       0.953137            0.953137           6      0.416364     -0.191079      0.669693    0.702620 0.493469         1
   QSI       0.947079            0.947079           5      0.394754     -0.107459      0.399290    0.421602 0.306041         1
  CIEN       0.939121            0.939121           5      0.391174     -0.041499      0.227982    0.242762 0.174540         1
  GLSI       0.931091            0.931091           7      0.412460     -0.039382      0.228603    0.245522 0.209881         1
   HLF       0.929017            0.929017           6      0.405365     -0.036372      0.186615    0.200874 0.188911         1
     F       0.927232            0.927232           6      0.404562     -0.011275      0.084398    0.091021 0.088226         1
  IMMX       0.910558            0.910558           5      0.378566     -0.056093      0.221327    0.243067 0.203846         1
  EQNR       0.909614            0.909614           4      0.330938     -0.008070      0.057898    0.063651 0.062358         1
  RGEN       0.865184            0.865184           7      0.383615     -0.052156      0.199794    0.230926 0.198000         1
  AMKR       0.855939            0.855939           8      0.381599     -0.059948      0.243128    0.284049 0.226504         1
  NDAQ       0.842596            0.842596           5      0.350175     -0.014621      0.060803    0.072161 0.068492         1
  CLSK       0.829408            0.829408           6      0.362947     -0.124458      0.343319    0.413933 0.301548         1
  NTRB       0.825213            0.825213           5      0.343292     -0.053625      0.172858    0.209471 0.159224         1
  AAPL       0.821189            0.821189           8      0.367422     -0.017264      0.071487    0.087053 0.057465         1
  JACK       0.789558            0.789558           8      0.355076     -0.129474      0.324986    0.411605 0.217936         1
  JAZZ       0.773926            0.773926           5      0.323916     -0.029222      0.081498    0.105305 0.073363         1
   IQV       0.766543            0.766543           8      0.346432     -0.073165      0.194874    0.254225 0.155897         1
  JKHY       0.730307            0.730307           4      0.264565     -0.028777      0.068185    0.093364 0.090772         1
  FROG       0.727724            0.727724           6      0.325011     -0.094898      0.199543    0.274202 0.148554         1
   NYT       0.712971            0.712971           4      0.259031     -0.036862      0.080157    0.112426 0.109343         1
   IYW       0.699088            0.699088           6      0.315335     -0.028916      0.061741    0.088316 0.080703         1
  NVTS       0.688289            0.688289           4      0.251422     -0.175654      0.261143    0.379410 0.245364         1
  ALNY       0.653793            0.653793           7      0.306503     -0.106786      0.177867    0.272054 0.219567         1
  FTXL       0.647129            0.647129           6      0.298899     -0.032170      0.055170    0.085253 0.083645         1
  bidu       0.639016            0.639016           6      0.296461     -0.086391      0.134824    0.210987 0.131636         1
  PLTR       0.627640            0.627640           6      0.293100     -0.120760      0.174395    0.277859 0.137813         1
  SCHD       0.610496            0.610496           7      0.293658     -0.026146      0.039939    0.065420 0.054852         1
  TQQQ       0.583656            0.583656           5      0.264226     -0.074583      0.091485    0.156745 0.153365         1
  META       0.578899            0.578899           5      0.262972     -0.101596      0.127203    0.219733 0.202979         1
   CNQ       0.549023            0.549023           7      0.277061     -0.105573      0.122195    0.222567 0.117830         1
   VRT       0.521071            0.521071           6      0.264780     -0.215492      0.215929    0.414395 0.229785         1
  RDVT       0.517323            0.517323           5      0.247723     -0.090056      0.091187    0.176267 0.149489         1
   NVT       0.512969            0.512969           6      0.262851     -0.085577      0.083504    0.162786 0.148993         1
   ICE       0.502958            0.502958           5      0.244420     -0.056153      0.054670    0.108698 0.105079         1
  ACWX       0.484602            0.484602           9      0.263760     -0.028948      0.026994    0.055703 0.038950         1
   AIQ       0.435877            0.435877           6      0.245971     -0.064291      0.048663    0.111644 0.070431         1
   CVI       0.431579            0.431579           7      0.250315     -0.218814      0.165714    0.383972 0.218814         1
   NKE       0.400174            0.400174           8      0.245777     -0.162701      0.108931    0.272209 0.245361         1
   RGC       0.389864            0.389864           5      0.221511     -0.359934      0.216207    0.554569 0.446366         1
  AVGO       0.388467            0.388467           7      0.241987     -0.129835      0.079886    0.205645 0.195614         1
   FDN       0.354555            0.354555           7      0.235945     -0.056731      0.030991    0.087407 0.075098         1
  ADBE       0.302904            0.302904           7      0.227549     -0.205415      0.094023    0.310406 0.273703         1
  WEAV       0.255108            0.255108           7      0.220580     -0.328152      0.124431    0.487759 0.354284         1
   NFE       0.107331            0.107331           5      0.183695     -0.612446      0.093820    0.874123 0.646315         1
   NOW       0.084500            0.084500           6      0.196177     -0.386001      0.042038    0.497486 0.386001         1
  NFLX       0.000000            0.000000           6      0.189057     -0.357039      0.000000    0.415332 0.357039         1
Saved per-ticker CSV to: output\per_ticker_20260117_2204.csv
real 799.647    user 0.076      sys 0.092       pcpu 0.02

Vidya_RSI_v0.1
============================================================
BEST SCORE: 0.41185593770446904
BEST PARAMS: {'vl': 7, 'vs': 5, 'fp': 4, 'sp': 22, 'ts': 31}
============================================================
Ticker          | PF uncap | PF diag  | Return    |  Score  | Trades
------------------------------------------------------------------------------
PL              |  1000.00 |     5.00 |    60.12% | 0.674878 | 2
GSAT            |  1000.00 |     5.00 |    31.59% | 0.674878 | 2
PAAS            |  1000.00 |     5.00 |    27.96% | 0.783597 | 3
IREN            |  1000.00 |     5.00 |    25.96% | 0.674878 | 2
TERN            |  1000.00 |     5.00 |    25.16% | 0.617195 | 1
EIS             |  1000.00 |     5.00 |     3.74% | 0.617195 | 1
QBTS            |  1000.00 |     5.00 |   155.08% | 0.897599 | 4
ASTS            |  1000.00 |     5.00 |   102.07% | 0.783597 | 3
SNBR            |  1000.00 |     5.00 |    82.89% | 0.674878 | 2
PGEN            |  1000.00 |     5.00 |    70.55% | 0.897599 | 4
AXTI            |  1000.00 |     5.00 |    56.24% | 0.674878 | 2
FTAI            |  1000.00 |     5.00 |    54.15% | 0.783597 | 3
HUT             |  1000.00 |     5.00 |    51.12% | 0.674878 | 2
RIGL            |  1000.00 |     5.00 |    49.22% | 0.783597 | 3
RYTM            |  1000.00 |     5.00 |    47.84% | 0.897599 | 4
GLUE            |  1000.00 |     5.00 |    47.69% | 0.674878 | 2
U               |  1000.00 |     5.00 |    46.16% | 0.674878 | 2
SATS            |  1000.00 |     5.00 |    44.51% | 0.783597 | 3
RIG             |  1000.00 |     5.00 |    42.95% | 0.897599 | 4
PBF             |  1000.00 |     5.00 |    41.54% | 0.897599 | 4
TTMI            |  1000.00 |     5.00 |    35.11% | 0.674878 | 2
MP              |  1000.00 |     5.00 |    33.65% | 0.783597 | 3
CDE             |  1000.00 |     5.00 |    33.48% | 0.783597 | 3
SA              |  1000.00 |     5.00 |    33.08% | 0.674878 | 2
CCJ             |  1000.00 |     5.00 |    32.94% | 0.674878 | 2
SPHR            |  1000.00 |     5.00 |    32.81% | 0.674878 | 2
LRCX            |  1000.00 |     5.00 |    32.66% | 0.674878 | 2
OGI             |  1000.00 |     5.00 |    32.30% | 0.674878 | 2
MLYS            |  1000.00 |     5.00 |    31.71% | 0.617195 | 1
MU              |  1000.00 |     5.00 |    31.12% | 0.674878 | 2
RKLB            |  1000.00 |     5.00 |    29.80% | 0.674878 | 2
NTRB            |  1000.00 |     5.00 |    29.59% | 0.674878 | 2
TNDM            |  1000.00 |     5.00 |    28.51% | 0.783597 | 3
DDOG            |  1000.00 |     5.00 |    27.23% | 0.674878 | 2
PALL            |  1000.00 |     5.00 |    26.82% | 0.783597 | 3
EGO             |  1000.00 |     5.00 |    26.82% | 0.783597 | 3
BLTE            |  1000.00 |     5.00 |    26.65% | 0.783597 | 3
AMAT            |  1000.00 |     5.00 |    26.48% | 0.674878 | 2
NEM             |  1000.00 |     5.00 |    25.44% | 0.674878 | 2
MD              |  1000.00 |     5.00 |    25.42% | 0.783597 | 3
BAC             |  1000.00 |     5.00 |    25.07% | 0.783597 | 3
IONQ            |  1000.00 |     5.00 |    24.37% | 0.617195 | 1
IMMX            |  1000.00 |     5.00 |    23.65% | 0.617195 | 1
BE              |  1000.00 |     5.00 |    23.47% | 0.674878 | 2
TEVA            |  1000.00 |     5.00 |    23.40% | 0.674878 | 2
NGL             |  1000.00 |     5.00 |    23.08% | 0.674878 | 2
COHR            |  1000.00 |     5.00 |    22.25% | 0.674878 | 2
NG              |  1000.00 |     5.00 |    21.66% | 0.674878 | 2
ENVA            |  1000.00 |     5.00 |    21.07% | 0.783597 | 3
PVLA            |  1000.00 |     5.00 |    20.58% | 0.674878 | 2
VSCO            |  1000.00 |     5.00 |    20.05% | 0.674878 | 2
FTI             |  1000.00 |     5.00 |    20.00% | 0.783597 | 3
TK              |  1000.00 |     5.00 |    19.32% | 0.674878 | 2
AMD             |  1000.00 |     5.00 |    18.27% | 0.674878 | 2
AG              |  1000.00 |     5.00 |    17.03% | 0.783597 | 3
CAT             |  1000.00 |     5.00 |    16.68% | 0.674878 | 2
TCMD            |  1000.00 |     5.00 |    16.60% | 0.617195 | 1
IBM             |  1000.00 |     5.00 |    16.10% | 0.674878 | 2
MDB             |  1000.00 |     5.00 |    15.79% | 0.783597 | 3
CVI             |  1000.00 |     5.00 |    15.60% | 0.674878 | 2
INCY            |  1000.00 |     5.00 |    15.20% | 0.674878 | 2
RVMD            |  1000.00 |     5.00 |    14.03% | 0.617195 | 1
NDAQ            |  1000.00 |     5.00 |    14.01% | 0.674878 | 2
BBIO            |  1000.00 |     5.00 |    13.87% | 0.617195 | 1
DINO            |  1000.00 |     5.00 |    13.59% | 0.617195 | 1
SQM             |  1000.00 |     5.00 |    13.54% | 0.617195 | 1
GOOGL           |  1000.00 |     5.00 |    13.00% | 0.617195 | 1
LMT             |  1000.00 |     5.00 |    12.96% | 0.674878 | 2
DELL            |  1000.00 |     5.00 |    11.96% | 0.617195 | 1
MAR             |  1000.00 |     5.00 |    11.12% | 0.783597 | 3
ICE             |  1000.00 |     5.00 |    10.77% | 0.674878 | 2
LVS             |  1000.00 |     5.00 |    10.77% | 0.674878 | 2
CIEN            |  1000.00 |     5.00 |    10.60% | 0.617195 | 1
ANAB            |  1000.00 |     5.00 |    10.57% | 0.674878 | 2
XLK             |  1000.00 |     5.00 |    10.39% | 0.617195 | 1
XME             |  1000.00 |     5.00 |    10.30% | 0.617195 | 1
ARGX            |  1000.00 |     5.00 |     9.84% | 0.617195 | 1
IQV             |  1000.00 |     5.00 |     9.20% | 0.674878 | 2
F               |  1000.00 |     5.00 |     8.90% | 0.674878 | 2
APH             |  1000.00 |     5.00 |     8.82% | 0.617195 | 1
IMO             |  1000.00 |     5.00 |     7.86% | 0.617195 | 1
XBI             |  1000.00 |     5.00 |     7.73% | 0.617195 | 1
BLOK            |  1000.00 |     5.00 |     7.66% | 0.674878 | 2
SPY             |  1000.00 |     5.00 |     7.57% | 0.674878 | 2
ROST            |  1000.00 |     5.00 |     7.13% | 0.617195 | 1
HCA             |  1000.00 |     5.00 |     6.86% | 0.617195 | 1
NSC             |  1000.00 |     5.00 |     6.42% | 0.783597 | 3
AMX             |  1000.00 |     5.00 |     6.28% | 0.617195 | 1
NYT             |  1000.00 |     5.00 |     6.12% | 0.674878 | 2
AER             |  1000.00 |     5.00 |     5.53% | 0.674878 | 2
TJX             |  1000.00 |     5.00 |     5.28% | 0.617195 | 1
CBRL            |  1000.00 |     5.00 |     4.23% | 0.617195 | 1
SCHD            |  1000.00 |     5.00 |     2.90% | 0.783597 | 3
ARKK            |  1000.00 |     5.00 |     0.46% | 0.674878 | 2
TEL             |  1000.00 |     5.00 |     0.12% | 0.617195 | 1
AREC            |  1000.00 |     5.00 |    54.31% | 0.674878 | 2
OKLO            |  1000.00 |     5.00 |    39.83% | 0.674878 | 2
QSI             |  1000.00 |     5.00 |    36.19% | 0.674878 | 2
EOSE            |  1000.00 |     5.00 |    33.12% | 0.674878 | 2
VICR            |  1000.00 |     5.00 |    33.09% | 0.783597 | 3
TMO             |  1000.00 |     5.00 |    29.10% | 0.783597 | 3
XPEV            |  1000.00 |     5.00 |    29.04% | 0.897599 | 4
GOSS            |  1000.00 |     5.00 |    28.90% | 0.674878 | 2
GRAL            |  1000.00 |     5.00 |    23.37% | 0.674878 | 2
TQQQ            |  1000.00 |     5.00 |    23.31% | 0.617195 | 1
NEO             |  1000.00 |     5.00 |    21.98% | 0.617195 | 1
CSIQ            |  1000.00 |     5.00 |    21.51% | 0.617195 | 1
BOOT            |  1000.00 |     5.00 |    20.56% | 0.897599 | 4
CNQ             |  1000.00 |     5.00 |    20.49% | 0.783597 | 3
NVGS            |  1000.00 |     5.00 |    20.15% | 0.783597 | 3
ANDE            |  1000.00 |     5.00 |    19.11% | 0.783597 | 3
AXP             |  1000.00 |     5.00 |    15.98% | 0.897599 | 4
SNDK            |  1000.00 |     5.00 |    15.82% | 0.617195 | 1
OWLT            |  1000.00 |     5.00 |    15.23% | 0.674878 | 2
MKSI            |  1000.00 |     5.00 |    14.14% | 0.674878 | 2
CRS             |  1000.00 |     5.00 |    13.63% | 0.617195 | 1
RDVT            |  1000.00 |     5.00 |    13.23% | 0.674878 | 2
SNOW            |  1000.00 |     5.00 |    12.93% | 0.617195 | 1
ERO             |  1000.00 |     5.00 |    12.33% | 0.617195 | 1
AMKR            |  1000.00 |     5.00 |    11.92% | 0.617195 | 1
DLTR            |  1000.00 |     5.00 |    11.80% | 0.617195 | 1
NTRA            |  1000.00 |     5.00 |    11.24% | 0.674878 | 2
AVDL            |  1000.00 |     5.00 |    10.52% | 0.674878 | 2
AIQ             |  1000.00 |     5.00 |    10.21% | 0.674878 | 2
FTXL            |  1000.00 |     5.00 |     9.87% | 0.783597 | 3
MDT             |  1000.00 |     5.00 |     6.82% | 0.783597 | 3
RY              |  1000.00 |     5.00 |     4.09% | 0.674878 | 2
LIT             |  1000.00 |     5.00 |     3.91% | 0.783597 | 3
PBBK            |  1000.00 |     5.00 |     3.09% | 0.674878 | 2
MDGL            |   195.05 |     5.00 |    48.54% | 0.897599 | 4
VLO             |    90.49 |     5.00 |    19.69% | 0.897599 | 4
TAN             |    47.08 |     5.00 |    17.14% | 0.783597 | 3
FDN             |    17.80 |     5.00 |     0.13% | 0.674878 | 2
DBRG            |    17.14 |     5.00 |    24.37% | 0.897599 | 4
PASG            |    16.39 |     5.00 |   236.18% | 0.783597 | 3
NVTS            |    15.54 |     5.00 |    82.91% | 0.783597 | 3
NKE             |    14.87 |     5.00 |    11.63% | 0.783597 | 3
IYW             |    14.63 |     5.00 |     9.01% | 0.783597 | 3
PTCT            |    13.17 |     5.00 |    16.59% | 0.897599 | 4
SU              |    11.43 |     5.00 |    17.92% | 0.962738 | 5
APPN            |    10.53 |     5.00 |    13.27% | 0.783597 | 3
CRSP            |     8.18 |     5.00 |    34.08% | 0.783597 | 3
NVDA            |     7.59 |     5.00 |    12.97% | 0.783597 | 3
GLSI            |     7.45 |     5.00 |    24.02% | 0.783597 | 3
VET             |     6.65 |     5.00 |    27.90% | 0.897599 | 4
APLD            |     6.50 |     5.00 |    64.93% | 0.897599 | 4
COGT            |     5.97 |     5.00 |    20.92% | 0.674878 | 2
HTHT            |     5.92 |     5.00 |    14.87% | 0.783597 | 3
TARS            |     5.88 |     5.00 |    21.28% | 0.783597 | 3
JKHY            |     5.78 |     5.00 |     5.20% | 0.783597 | 3
HLF             |     5.51 |     5.00 |    29.11% | 0.783597 | 3
SIMO            |     5.06 |     5.00 |    12.09% | 0.783597 | 3
DSGN            |     4.96 |     4.96 |    14.95% | 0.783590 | 3
LUV             |     4.84 |     4.84 |     8.69% | 0.897566 | 4
ANVS            |     4.78 |     4.78 |    55.63% | 0.783549 | 3
GEV             |     4.67 |     4.67 |    12.77% | 0.783514 | 3
JAZZ            |     4.42 |     4.42 |     6.95% | 0.897379 | 4
CLF             |     4.31 |     4.31 |    17.19% | 0.783302 | 3
AEM             |     4.30 |     4.30 |     6.76% | 0.783295 | 3
HOOD            |     4.05 |     4.05 |    12.29% | 0.674279 | 2
RTX             |     3.81 |     3.81 |    14.59% | 0.896371 | 4
ON              |     3.81 |     3.81 |    23.60% | 0.896362 | 4
GH              |     3.61 |     3.61 |    20.19% | 0.781582 | 3
ASML            |     3.59 |     3.59 |    20.40% | 0.895423 | 4
RCUS            |     3.42 |     3.42 |    15.83% | 0.780318 | 3
AAPL            |     3.35 |     3.35 |     8.97% | 0.893512 | 4
REMX            |     3.27 |     3.27 |    12.90% | 0.778886 | 3
ZIM             |     3.19 |     3.19 |    25.32% | 0.777754 | 3
IAUX            |     3.02 |     3.02 |    23.03% | 0.888282 | 4
NRIM            |     3.01 |     3.01 |     8.46% | 0.666244 | 2
JNJ             |     2.93 |     2.93 |     3.65% | 0.664454 | 2
JPM             |     2.79 |     2.79 |     3.97% | 0.660019 | 2
ILMN            |     2.76 |     2.76 |    17.16% | 0.879908 | 4
MRK             |     2.67 |     2.67 |     5.19% | 0.875493 | 4
CENX            |     2.62 |     2.62 |    19.95% | 0.759575 | 3
ALAB            |     2.57 |     2.57 |    16.76% | 0.756519 | 3
WGS             |     2.51 |     2.51 |    20.80% | 0.752235 | 3
TSM             |     2.48 |     2.48 |    11.77% | 0.750489 | 3
DIBS            |     2.44 |     2.44 |     5.97% | 0.746789 | 3
CEG             |     2.41 |     2.41 |     7.47% | 0.637505 | 2
RGLD            |     2.40 |     2.40 |     9.80% | 0.743044 | 3
BBAI            |     2.38 |     2.38 |    17.99% | 0.741606 | 3
HIT             |     2.30 |     2.30 |    13.70% | 0.626463 | 2
ADPT            |     2.25 |     2.25 |    18.85% | 0.901574 | 5
IMAX            |     2.22 |     2.22 |     9.82% | 0.722149 | 3
bidu            |     2.08 |     2.08 |    12.21% | 0.895655 | 6
XSD             |     1.93 |     1.93 |     4.89% | 0.666817 | 3
NEGG            |     1.93 |     1.93 |    10.82% | 0.563121 | 2
UTHR            |     1.91 |     1.91 |     5.27% | 0.660269 | 3
DBI             |     1.88 |     1.88 |    15.81% | 0.823607 | 5
PLAB            |     1.84 |     1.84 |     6.24% | 0.642410 | 3
META            |     1.79 |     1.79 |     4.23% | 0.627801 | 3
FROG            |     1.79 |     1.79 |     5.26% | 0.526527 | 2
BLD             |     1.75 |     1.75 |     5.09% | 0.778409 | 5
RGEN            |     1.68 |     1.68 |     7.76% | 0.754027 | 5
UHS             |     1.67 |     1.67 |     2.59% | 0.585249 | 3
APP             |     1.66 |     1.66 |    11.44% | 0.684775 | 4
REGN            |     1.65 |     1.65 |     5.88% | 0.680337 | 4
PTGX            |     1.54 |     1.54 |     3.26% | 0.532962 | 3
NBIS            |     1.50 |     1.50 |     3.93% | 0.515926 | 3
TSLA            |     1.47 |     1.47 |     4.25% | 0.603443 | 4
MRNA            |     1.46 |     1.46 |     3.37% | 0.599264 | 4
FUTU            |     1.46 |     1.46 |     3.19% | 0.499413 | 3
LGND            |     1.44 |     1.44 |     2.67% | 0.491646 | 3
VERA            |     1.40 |     1.40 |     2.69% | 0.623902 | 5
XERS            |     1.34 |     1.34 |     2.13% | 0.445707 | 3
INTC            |     1.33 |     1.33 |     3.38% | 0.589100 | 5
AXSM            |     1.29 |     1.29 |     2.66% | 0.570235 | 5
M               |     1.23 |     1.23 |     0.60% | 0.389927 | 3
HL              |     1.13 |     1.13 |    -0.32% | 0.263092 | 2
GLW             |     1.10 |     1.10 |    -0.34% | 0.250040 | 2
PSNL            |     1.09 |     1.09 |    -2.63% | 0.245827 | 2
CMI             |     1.08 |     1.08 |    -0.13% | 0.244545 | 2
SNDL            |     1.08 |     1.08 |    -2.11% | 0.242923 | 2
AZTA            |     1.08 |     1.08 |    -0.92% | 0.408072 | 4
FDX             |     1.06 |     1.06 |    -0.61% | 0.396710 | 4
MLI             |     1.06 |     1.06 |    -0.35% | 0.232486 | 2
INO             |     1.05 |     1.05 |    -4.99% | 0.307513 | 3
SN              |     1.02 |     1.02 |    -1.47% | 0.296101 | 3
VRT             |     1.00 |     1.00 |    -3.01% | 0.211493 | 2
RGC             |     1.00 |     1.00 |    -9.81% | 0.210872 | 2
NAIL            |     0.87 |     0.87 |    -9.15% | 0.316541 | 4
EQNR            |     0.86 |     0.86 |    -2.95% | 0.312528 | 4
ABUS            |     0.84 |     0.84 |    -4.47% | 0.302815 | 4
ENTA            |     0.82 |     0.82 |    -9.31% | 0.218524 | 3
NVT             |     0.72 |     0.72 |    -2.00% | 0.123013 | 2
ASTL            |     0.66 |     0.66 |   -11.85% | 0.286269 | 5
ADBE            |     0.62 |     0.62 |    -6.15% | 0.232562 | 4
AEO             |     0.58 |     0.58 |   -14.49% | 0.153453 | 3
ABBV            |     0.53 |     0.53 |    -7.53% | 0.250135 | 5
LYFT            |     0.51 |     0.51 |   -17.30% | 0.205306 | 4
EPAM            |     0.43 |     0.43 |   -16.22% | 0.190068 | 4
PBR             |     0.07 |     0.07 |   -18.61% | 0.180192 | 5
CLS             |     0.00 |     0.00 |     0.00% | 0.000000 | 0
ARKW            |     0.00 |     0.00 |     0.00% | 0.000000 | 0
AMG             |     0.00 |     0.00 |     0.00% | 0.000000 | 0
STX             |     0.00 |     0.00 |     0.00% | 0.000000 | 0
CM              |     0.00 |     0.00 |     0.00% | 0.000000 | 0
WDC             |     0.00 |     0.00 |     0.00% | 0.000000 | 0
ATI             |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
MSFT            |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
CX              |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
WWD             |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
TD              |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
AMRC            |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
CRWV            |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
GE              |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
ACWX            |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
BITF            |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
GS              |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
RGTI            |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
CLSK            |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
OMER            |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
NFLX            |     0.00 |     0.00 |    -0.12% | 0.015563 | 1
BABA            |     0.00 |     0.00 |    -0.24% | 0.034915 | 2
VTR             |     0.00 |     0.00 |    -0.24% | 0.034915 | 2
OLMA            |     0.00 |     0.00 |    -0.24% | 0.034915 | 2
GM              |     0.00 |     0.00 |    -1.76% | 0.034915 | 2
GLD             |     0.00 |     0.00 |    -3.78% | 0.015563 | 1
PLTR            |     0.00 |     0.00 |    -4.37% | 0.034915 | 2
ALNY            |     0.00 |     0.00 |    -5.38% | 0.015563 | 1
SOFI            |     0.00 |     0.00 |    -5.53% | 0.015563 | 1
CTSH            |     0.00 |     0.00 |    -5.98% | 0.081113 | 3
ARKF            |     0.00 |     0.00 |    -7.60% | 0.015563 | 1
RCL             |     0.00 |     0.00 |    -8.61% | 0.034915 | 2
XPEL            |     0.00 |     0.00 |   -10.13% | 0.081113 | 3
KOD             |     0.00 |     0.00 |   -10.32% | 0.034915 | 2
ANET            |     0.00 |     0.00 |   -10.73% | 0.034915 | 2
AVGO            |     0.00 |     0.00 |   -12.42% | 0.015563 | 1
LLY             |     0.00 |     0.00 |   -12.87% | 0.081113 | 3
ASC             |     0.00 |     0.00 |   -12.99% | 0.081113 | 3
NFE             |     0.00 |     0.00 |   -14.77% | 0.034915 | 2
WEAV            |     0.00 |     0.00 |   -14.78% | 0.081113 | 3
BFH             |     0.00 |     0.00 |   -14.93% | 0.081113 | 3
ZURA            |     0.00 |     0.00 |   -15.50% | 0.081113 | 3
NOW             |     0.00 |     0.00 |   -16.33% | 0.081113 | 3
JACK            |     0.00 |     0.00 |   -18.37% | 0.138693 | 4
LABU            |     0.00 |     0.00 |   -18.89% | 0.034915 | 2
PACB            |     0.00 |     0.00 |   -21.11% | 0.081113 | 3
NAMS            |     0.00 |     0.00 |   -22.12% | 0.081113 | 3
SBGI            |     0.00 |     0.00 |   -38.87% | 0.189057 | 6
[DONE] Detailed results archived in: output\opt_results_20260117_1925.csv
real 1833.198   user 0.061      sys 0.139       pcpu 0.01

BEST SCORE: 0.3971465611680196
BEST PARAMS: {'vl': 10, 'vs': 5, 'fp': 10, 'sp': 21, 'ts': 43}
============================================================
Ticker          | Trades   | Return     | PF Diag    | Score
------------------------------------------------------------
QBTS            | 3        |    44.55% |       5.00 | 0.7836
AER             | 2        |     4.78% |       5.00 | 0.6749
IQV             | 2        |     7.77% |       5.00 | 0.6749
OLMA            | 2        |    46.48% |       5.00 | 0.6749
ARKF            | 1        |    10.87% |       5.00 | 0.6172
TSM             | 3        |    26.97% |       5.00 | 0.7836
MDGL            | 2        |    10.95% |       5.00 | 0.6749
NBIS            | 2        |    -0.18% |       0.00 | 0.0349
IYW             | 2        |     8.21% |       5.00 | 0.6749
NEM             | 1        |    11.27% |       5.00 | 0.6172
CSIQ            | 2        |    25.14% |       5.00 | 0.6749
TTMI            | 2        |    27.17% |       5.00 | 0.6749
ARKK            | 1        |    14.16% |       5.00 | 0.6172
ALNY            | 1        |    12.01% |       5.00 | 0.6172
OWLT            | 2        |    11.01% |       5.00 | 0.6749
HUT             | 2        |    28.45% |       5.00 | 0.6749
CRS             | 1        |    11.81% |       5.00 | 0.6172
NFE             | 2        |   -40.06% |       0.00 | 0.0349
AZTA            | 3        |    -2.51% |       0.94 | 0.2609
CIEN            | 2        |    23.25% |       5.00 | 0.6749
NFLX            | 1        |    -0.12% |       0.00 | 0.0156
OMER            | 1        |    -0.12% |       0.00 | 0.0156
RCUS            | 2        |    19.44% |       5.00 | 0.6749
PLAB            | 3        |   -14.27% |       0.40 | 0.1203
XLK             | 1        |     6.94% |       5.00 | 0.6172
VTR             | 1        |    -0.12% |       0.00 | 0.0156
F               | 1        |    -0.12% |       0.00 | 0.0156
NVT             | 2        |    15.03% |       5.00 | 0.6749
PSNL            | 3        |    12.08% |       1.85 | 0.6452
RIGL            | 3        |    11.14% |       5.00 | 0.7836
XBI             | 2        |     8.56% |       5.00 | 0.6749
XME             | 2        |    16.47% |       5.00 | 0.6749
RGEN            | 2        |    12.87% |       5.00 | 0.6749
ASTL            | 4        |   -24.57% |       0.40 | 0.1837
CTSH            | 2        |     0.51% |       1.20 | 0.2936
TERN            | 1        |    21.60% |       5.00 | 0.6172
SOFI            | 1        |    15.65% |       5.00 | 0.6172
MU              | 2        |    25.48% |       5.00 | 0.6749
DBRG            | 3        |    10.59% |       1.94 | 0.6695
LUV             | 3        |    20.35% |       5.00 | 0.7836
BFH             | 3        |    -9.11% |       0.53 | 0.1428
MAR             | 2        |     6.66% |       5.00 | 0.6749
PTCT            | 3        |    19.70% |       5.00 | 0.7836
MKSI            | 2        |    22.76% |       5.00 | 0.6749
ARKW            | 1        |    10.95% |       5.00 | 0.6172
ENVA            | 4        |     5.53% |       4.90 | 0.8976
RGLD            | 2        |    15.38% |       5.00 | 0.6749
NTRB            | 3        |    17.80% |       4.51 | 0.7834
IONQ            | 1        |    16.13% |       5.00 | 0.6172
RVMD            | 1        |    11.07% |       5.00 | 0.6172
RYTM            | 1        |    12.39% |       5.00 | 0.6172
NKE             | 2        |    -9.01% |       0.00 | 0.0349
BAC             | 3        |    13.32% |       5.00 | 0.7836
EIS             | 2        |     6.33% |       5.00 | 0.6749
IAUX            | 3        |    30.84% |       5.00 | 0.7836
WGS             | 2        |    18.78% |       5.00 | 0.6749
CENX            | 2        |    -1.60% |       1.05 | 0.2303
BBIO            | 1        |    11.32% |       5.00 | 0.6172
SA              | 1        |    -0.12% |       0.00 | 0.0156
XPEV            | 3        |    11.38% |       2.03 | 0.6886
ASML            | 2        |    14.28% |       5.00 | 0.6749
CLF             | 3        |    30.97% |       5.00 | 0.7836
FTAI            | 3        |    27.28% |       5.00 | 0.7836
U               | 1        |    19.49% |       5.00 | 0.6172
HTHT            | 2        |    -2.34% |       0.84 | 0.1552
WEAV            | 2        |    -7.39% |       0.00 | 0.0349
AMG             | 1        |     5.37% |       5.00 | 0.6172
TQQQ            | 1        |    14.16% |       5.00 | 0.6172
LRCX            | 1        |    10.18% |       5.00 | 0.6172
PACB            | 2        |    47.87% |       5.00 | 0.6749
TARS            | 2        |    22.00% |       5.00 | 0.6749
HL              | 1        |    13.73% |       5.00 | 0.6172
LYFT            | 4        |     3.57% |       1.33 | 0.5343
NOW             | 2        |   -12.62% |       0.00 | 0.0349
MDB             | 2        |    13.84% |       5.00 | 0.6749
AEO             | 3        |    -3.12% |       0.93 | 0.2590
INCY            | 2        |     8.60% |       5.00 | 0.6749
NDAQ            | 2        |    10.83% |       5.00 | 0.6749
ZIM             | 3        |   -16.64% |       0.46 | 0.1300
RCL             | 3        |     2.30% |       1.46 | 0.4978
TAN             | 3        |     0.39% |       1.17 | 0.3613
HOOD            | 0        |     0.00% |       0.00 | 0.0000
ICE             | 2        |     8.06% |       5.00 | 0.6749
NSC             | 2        |     1.49% |       1.55 | 0.4422
LVS             | 2        |    18.98% |       5.00 | 0.6749
GSAT            | 2        |    26.51% |       5.00 | 0.6749
MRNA            | 3        |    13.63% |       2.09 | 0.7002
JACK            | 4        |     1.48% |       1.28 | 0.5077
GOSS            | 2        |    44.64% |       5.00 | 0.6749
EQNR            | 3        |    13.85% |       5.00 | 0.7836
SNBR            | 3        |    73.03% |       5.00 | 0.7836
PBR             | 3        |   -13.73% |       0.00 | 0.0811
CEG             | 2        |    12.80% |       5.00 | 0.6749
AXSM            | 3        |    -7.95% |       0.54 | 0.1439
COGT            | 1        |    22.57% |       5.00 | 0.6172
KOD             | 2        |    18.92% |       5.00 | 0.6749
GRAL            | 2        |    11.40% |       3.57 | 0.6728
BITF            | 2        |    21.12% |       5.00 | 0.6749
UHS             | 3        |     5.78% |       5.00 | 0.7836
DLTR            | 1        |    10.20% |       5.00 | 0.6172
TSLA            | 4        |    20.49% |       4.25 | 0.8972
AG              | 3        |    35.39% |       5.00 | 0.7836
INO             | 2        |    -6.64% |       0.91 | 0.1795
FDN             | 1        |     6.24% |       5.00 | 0.6172
GLSI            | 2        |    18.22% |       5.00 | 0.6749
SQM             | 1        |    10.79% |       5.00 | 0.6172
REMX            | 3        |     2.23% |       1.52 | 0.5264
AXTI            | 2        |    21.34% |       5.00 | 0.6749
CMI             | 1        |    -6.36% |       0.00 | 0.0156
SBGI            | 4        |   -34.76% |       0.00 | 0.1387
BOOT            | 3        |    15.41% |       5.00 | 0.7836
DELL            | 2        |    25.20% |       5.00 | 0.6749
NAIL            | 3        |   -19.33% |       0.00 | 0.0811
QSI             | 2        |    63.47% |       5.00 | 0.6749
JNJ             | 2        |     3.39% |       3.76 | 0.6736
SNOW            | 2        |    23.65% |       5.00 | 0.6749
HLF             | 3        |    17.40% |       2.79 | 0.7679
ANAB            | 1        |    20.90% |       5.00 | 0.6172
IMO             | 2        |     7.26% |       5.00 | 0.6749
ERO             | 2        |    22.04% |       5.00 | 0.6749
ANVS            | 3        |    14.11% |       2.88 | 0.7711
XERS            | 3        |     8.48% |       2.54 | 0.7549
EOSE            | 1        |    25.05% |       5.00 | 0.6172
TJX             | 1        |     4.64% |       5.00 | 0.6172
AVGO            | 1        |    14.22% |       5.00 | 0.6172
VICR            | 2        |    12.09% |       5.00 | 0.6749
AEM             | 2        |    -0.24% |       0.00 | 0.0349
NVGS            | 3        |     7.65% |       5.00 | 0.7836
CAT             | 1        |     6.24% |       5.00 | 0.6172
CDE             | 2        |    36.06% |       5.00 | 0.6749
PAAS            | 2        |    24.04% |       5.00 | 0.6749
SNDK            | 1        |    13.14% |       5.00 | 0.6172
BLTE            | 3        |     9.94% |       2.88 | 0.7711
NTRA            | 1        |     9.35% |       5.00 | 0.6172
DINO            | 1        |    -0.12% |       0.00 | 0.0156
PBF             | 3        |    13.62% |       2.12 | 0.7068
AMX             | 0        |     0.00% |       0.00 | 0.0000
VRT             | 3        |    10.94% |       3.21 | 0.7781
JKHY            | 2        |     0.67% |       1.25 | 0.3163
CNQ             | 4        |    34.33% |       5.00 | 0.8976
MSFT            | 2        |    11.12% |       5.00 | 0.6749
XPEL            | 3        |   -27.45% |       0.00 | 0.0811
NRIM            | 2        |    -6.43% |       0.61 | 0.0989
SIMO            | 2        |    26.59% |       5.00 | 0.6749
CCJ             | 2        |    13.41% |       5.00 | 0.6749
ZURA            | 2        |     1.55% |       5.00 | 0.6749
WWD             | 2        |    15.47% |       5.00 | 0.6749
GLD             | 1        |     3.00% |       5.00 | 0.6172
VLO             | 4        |    27.34% |       5.00 | 0.8976
VET             | 3        |     1.42% |       1.31 | 0.4283
AMD             | 2        |    26.36% |       5.00 | 0.6749
ADPT            | 4        |    -2.16% |       1.08 | 0.4059
NVTS            | 3        |     0.88% |       1.29 | 0.4212
AMKR            | 1        |     9.64% |       5.00 | 0.6172
GEV             | 3        |   -21.33% |       0.09 | 0.0866
NEGG            | 1        |    -0.12% |       0.00 | 0.0156
ADBE            | 3        |   -11.80% |       0.00 | 0.0811
PVLA            | 2        |    -3.33% |       0.99 | 0.2058
IBM             | 2        |     5.91% |       5.00 | 0.6749
MRK             | 2        |    -0.24% |       0.00 | 0.0349
PALL            | 3        |    21.99% |       5.00 | 0.7836
ON              | 3        |    24.38% |       5.00 | 0.7836
MLYS            | 2        |    25.31% |       5.00 | 0.6749
BABA            | 2        |   -10.56% |       0.00 | 0.0349
WDC             | 1        |    13.32% |       5.00 | 0.6172
ATI             | 2        |    22.28% |       5.00 | 0.6749
PTGX            | 2        |    11.40% |       5.00 | 0.6749
CRSP            | 2        |    31.05% |       5.00 | 0.6749
LGND            | 3        |    -7.04% |       0.55 | 0.1462
TEL             | 2        |    10.53% |       5.00 | 0.6749
FDX             | 2        |     7.47% |       5.00 | 0.6749
IMAX            | 3        |    10.37% |       2.93 | 0.7724
TEVA            | 2        |    -5.88% |       0.00 | 0.0349
GM              | 2        |     7.04% |       5.00 | 0.6749
bidu            | 4        |    12.16% |       2.78 | 0.8808
NAMS            | 3        |   -16.20% |       0.00 | 0.0811
ACWX            | 1        |     2.45% |       5.00 | 0.6172
RGTI            | 1        |    33.91% |       5.00 | 0.6172
MD              | 2        |    -2.89% |       0.84 | 0.1569
GH              | 3        |    13.63% |       2.63 | 0.7603
RIG             | 3        |    19.93% |       5.00 | 0.7836
M               | 2        |    -9.30% |       0.00 | 0.0349
ASC             | 2        |     8.13% |       5.00 | 0.6749
SATS            | 3        |    14.11% |       5.00 | 0.7836
TMO             | 2        |    17.62% |       5.00 | 0.6749
RGC             | 2        |    -9.83% |       0.91 | 0.1804
GOOGL           | 1        |     9.03% |       5.00 | 0.6172
VSCO            | 2        |    -4.41% |       0.92 | 0.1806
PASG            | 3        |    13.21% |       1.98 | 0.6783
BLD             | 4        |     8.86% |       2.17 | 0.8246
OKLO            | 1        |    31.53% |       5.00 | 0.6172
MP              | 3        |    47.98% |       5.00 | 0.7836
EPAM            | 4        |   -20.72% |       0.00 | 0.1387
FROG            | 3        |    13.52% |       2.46 | 0.7489
ILMN            | 3        |    35.09% |       5.00 | 0.7836
AXP             | 3        |     5.09% |       5.00 | 0.7836
RTX             | 2        |     4.99% |       5.00 | 0.6749
TD              | 0        |     0.00% |       0.00 | 0.0000
SU              | 4        |    14.02% |       5.00 | 0.8976
ASTS            | 3        |    60.80% |       5.00 | 0.7836
NGL             | 2        |    30.18% |       5.00 | 0.6749
EGO             | 2        |    22.66% |       5.00 | 0.6749
PGEN            | 3        |    22.32% |       5.00 | 0.7836
MLI             | 2        |    -1.36% |       0.90 | 0.1746
ANDE            | 2        |     8.01% |       5.00 | 0.6749
AAPL            | 1        |     7.10% |       5.00 | 0.6172
JPM             | 3        |     0.05% |       1.14 | 0.3472
BBAI            | 2        |    72.09% |       5.00 | 0.6749
COHR            | 2        |    16.50% |       5.00 | 0.6749
NG              | 2        |    18.37% |       5.00 | 0.6749
PBBK            | 2        |     1.53% |       5.00 | 0.6749
DDOG            | 2        |    20.88% |       5.00 | 0.6749
ABBV            | 5        |     0.04% |       1.13 | 0.4827
CM              | 0        |     0.00% |       0.00 | 0.0000
ENTA            | 3        |    22.60% |       2.79 | 0.7678
GE              | 2        |     8.33% |       5.00 | 0.6749
LIT             | 2        |     0.01% |       1.11 | 0.2532
STX             | 1        |    11.75% |       5.00 | 0.6172
GS              | 2        |    17.40% |       5.00 | 0.6749
RDVT            | 2        |    16.53% |       5.00 | 0.6749
ARGX            | 1        |     7.99% |       5.00 | 0.6172
RKLB            | 1        |    23.97% |       5.00 | 0.6172
LLY             | 2        |     5.48% |       3.44 | 0.6720
SN              | 3        |     5.84% |       1.66 | 0.5828
AREC            | 2        |    31.10% |       5.00 | 0.6749
AIQ             | 2        |    10.94% |       5.00 | 0.6749
MDT             | 2        |     4.77% |       5.00 | 0.6749
APH             | 1        |     9.20% |       5.00 | 0.6172
CX              | 1        |    11.55% |       5.00 | 0.6172
DSGN            | 4        |   -28.19% |       0.41 | 0.1863
NYT             | 2        |     0.02% |       1.11 | 0.2542
META            | 2        |     2.14% |       1.53 | 0.4359
SNDL            | 2        |    -6.72% |       0.00 | 0.0349
HCA             | 1        |     5.64% |       5.00 | 0.6172
ANET            | 2        |     5.25% |       5.00 | 0.6749
APLD            | 4        |    29.14% |       5.00 | 0.8976
ABUS            | 4        |   -15.06% |       0.45 | 0.1925
CLS             | 1        |    22.45% |       5.00 | 0.6172
NEO             | 1        |   -13.87% |       0.00 | 0.0156
SCHD            | 3        |     7.07% |       5.00 | 0.7836
IMMX            | 1        |    19.96% |       5.00 | 0.6172
TK              | 2        |    15.60% |       5.00 | 0.6749
IREN            | 1        |    23.00% |       5.00 | 0.6172
ROST            | 1        |     6.25% |       5.00 | 0.6172
INTC            | 5        |   -13.13% |       0.66 | 0.2851
FUTU            | 3        |   -18.34% |       0.00 | 0.0811
XSD             | 2        |    12.82% |       5.00 | 0.6749
FTI             | 2        |    18.31% |       5.00 | 0.6749
FTXL            | 3        |    -2.39% |       0.51 | 0.1385
VERA            | 4        |     1.12% |       5.00 | 0.8976
CRWV            | 1        |   -21.33% |       0.00 | 0.0156
APP             | 3        |    20.16% |       2.69 | 0.7637
SPY             | 2        |     5.41% |       5.00 | 0.6749
CLSK            | 1        |    18.59% |       5.00 | 0.6172
AMAT            | 2        |    21.02% |       5.00 | 0.6749
DBI             | 3        |   -36.58% |       0.00 | 0.0811
SPHR            | 2        |    25.60% |       5.00 | 0.6749
AMRC            | 1        |    -0.12% |       0.00 | 0.0156
LABU            | 1        |    18.97% |       5.00 | 0.6172
BE              | 2        |    19.46% |       5.00 | 0.6749
CBRL            | 1        |    -0.06% |       0.00 | 0.0156
UTHR            | 3        |    26.81% |       5.00 | 0.7836
NVDA            | 3        |    14.83% |       5.00 | 0.7836
GLUE            | 2        |    16.01% |       5.00 | 0.6749
RY              | 0        |     0.00% |       0.00 | 0.0000
GLW             | 2        |    13.08% |       5.00 | 0.6749
LMT             | 2        |    10.27% |       5.00 | 0.6749
JAZZ            | 1        |    -7.70% |       0.00 | 0.0156
CVI             | 2        |    12.47% |       5.00 | 0.6749
REGN            | 1        |    -0.12% |       0.00 | 0.0156
TCMD            | 1        |     9.78% |       5.00 | 0.6172
HIT             | 2        |    10.41% |       2.12 | 0.6020
TNDM            | 2        |     3.04% |       1.44 | 0.3967
BLOK            | 1        |    10.77% |       5.00 | 0.6172
DIBS            | 3        |   -14.67% |       0.00 | 0.0811
OGI             | 1        |   -17.55% |       0.00 | 0.0156
ALAB            | 2        |    37.85% |       5.00 | 0.6749
PL              | 2        |    48.98% |       5.00 | 0.6749
AVDL            | 2        |    23.55% |       5.00 | 0.6749
PLTR            | 2        |     5.83% |       2.27 | 0.6232
APPN            | 3        |    -0.02% |       1.14 | 0.3506
[DONE] Detailed results archived in: output\opt_results_20260117_1300.csv
real 938.696    user 0.045      sys 0.108       pcpu 0.01

Based on the backtest data you provided, here is a comparative analysis of Vidya_RSI_v0.1 and adapt_RSI_v0.2.1.
The primary difference between these versions is their robustness and activity level. 
While Vidya_RSI shows higher per-trade returns, it suffers from a significant lack of data (low trade count), 
whereas adapt_RSI is a more mature, high-frequency system with deeper risk management.
1. High-Level Performance Comparison
Metric	Vidya_RSI_v0.1	adapt_RSI_v0.2.1	Winner / Insight
Avg Trades / Ticker	1.1	6.05	adapt_RSI: Higher statistical significance.
Best Score	0.3876	0.4234	adapt_RSI: Better overall efficiency.
Profit Factor (Diag)	5.00 (Cap)	3.74	Vidya_RSI: Highly skewed by low trade counts.
Ticker Coverage	94.7% (18/19)	78.9% (15/19)	Vidya_RSI: Operates on more tickers.

2. Key Strategy Differences
Vidya_RSI_v0.1 (The "Sniper" - Low Frequency)
Low Sample Size: Most tickers have only 1 trade. While the returns look impressive (e.g., 22.25% on IREN), a single trade is not enough to prove a strategy's edge. This is likely "cherry-picking" market moves.
Extreme Score Skew: The "PF Diag" (Profit Factor Diagnostic) is hitting a ceiling of 5.00 across almost every ticker. This usually happens in backtests when there are zero losing trades, which is unsustainable in live trading.
The EOSE Failure: This ticker shows the risk of low-frequency systems; with only 2 trades, it hit a -12.71% return, causing the score to plumment to 0.1369.

adapt_RSI_v0.2.1 (The "Machine" - Adaptive/High Frequency)
Statistical Significance: With an average of 6 trades per ticker, this version is much more reliable. It isn't just catching one lucky move; it's entering and exiting multiple times.
Complex Optimization: You are using Optuna (Bayesian optimization) with sophisticated penalties (Soft penalty, Tail-protection, Max-trades penalty). This suggests the model is better at filtering out "fake" profits that won't happen in real life.
Better Ticker Balance: Tickers like TERN (0.705 score) and STI (0.685 score) show high profit factors even with 6–7 trades, which is a very strong signal of a real edge.
Realism: This version includes a commission_rate_per_side (0.0006), making the results much closer to what you would see in a live brokerage account.

3. Critical Warnings
Overfitting Risk: Vidya_RSI looks overfitted. When a strategy only takes 1 trade over a period, it has likely "memorized" a specific price spike.
Ineligible Tickers: In the adapt_RSI results, tickers like IREN, PRAX, STX, and NFLX are marked eligible: False. This is because they didn't meet your min_trades gate: 5. This is a healthy filter that prevents you from trusting low-data results.

4. Summary Verdict
Use adapt_RSI_v0.2.1. Even though the raw returns on some individual tickers in Vidya_RSI look "sexier," the adapt_RSI version is a far superior system. 
It has higher trade frequency, integrated commissions, and a more rigorous scoring logic that accounts for risk (Max Drawdown) and consistency.

============================================================
BEST SCORE: 0.38762626197327854
BEST PARAMS: {'vl': 17, 'vs': 30, 'fp': 16, 'sp': 86, 'ts': 36}
============================================================
Ticker          | Trades   | Return     | PF Diag    | Score
------------------------------------------------------------
CIEN            | 1        |     8.14% |       5.00 | 0.6172
SATS            | 1        |    15.31% |       5.00 | 0.6172
ASML            | 1        |     6.72% |       5.00 | 0.6172
BITF            | 1        |    22.02% |       5.00 | 0.6172
XPEV            | 1        |    12.52% |       5.00 | 0.6172
METC            | 1        |    17.70% |       5.00 | 0.6172
STX             | 1        |     8.98% |       5.00 | 0.6172
BLBX            | 1        |    11.70% |       5.00 | 0.6172
AZTA            | 1        |     6.24% |       5.00 | 0.6172
EOSE            | 2        |   -12.71% |       0.77 | 0.1369
PRAX            | 2        |     0.12% |       5.00 | 0.6749
TERN            | 1        |    15.80% |       5.00 | 0.6172
SNDK            | 1        |    13.14% |       5.00 | 0.6172
MRNA            | 2        |    12.79% |       5.00 | 0.6749
NFLX            | 1        |     7.41% |       5.00 | 0.6172
IREN            | 1        |    22.25% |       5.00 | 0.6172
CLS             | 1        |    15.32% |       5.00 | 0.6172
STI             | 1        |    16.65% |       5.00 | 0.6172
NEM             | 0        |     0.00% |       0.00 | 0.0000
[DONE] Detailed results archived in: output\opt_results_20260117_1214.csv
real 91.875     user 0.046      sys 0.076       pcpu 0.13

1/17/2026,adapt_RSI v0.2 & adapt_RSI v0.2.1
adapt_RSI v0.2.1
Success! We have officially hit the "High-Density" sweet spot.
By forcing the model to prioritize frequency, we’ve achieved something very difficult in algorithmic optimization: 100% Coverage and an average trade count of 9.895. 
This is a statistically significant dataset that moves your strategy from "lucky timing" to "systemic edge."
1. Analysis of the ResultsCoverage (19/19): This is the first time every single ticker in your list met the criteria. The strategy is now "universal" across your entire watch list.Avg Trades (9.895): We hit the target of ~10 trades per ticker perfectly. This provides a much more reliable Profit Factor (PF) than the previous versions.The "Grinder" Stats: Notice that your Avg Profit Factor (raw) is 3.35. While lower than the previous 4.64, it is much more "real." A strategy that maintains a 3.35 PF over nearly 200 total trades (19 tickers × 9.8 trades) is a powerhouse.
2. The "Aggressive" Parameter ProfileThe Best Params reveal exactly how the model adapted to your high-frequency requirement:atrPeriod: 11 & slMultiplier: 1.26: These are very tight. The model is using a "tight leash" strategy—if a trade doesn't move in your direction immediately, it cuts it and looks for the next entry.time_stop: 14: The model is effectively "cycling" capital. If a trade hasn't hit TP or SL within 14 bars, it exits. This is why your trade count jumped up.smooth_len: 2: By reducing smoothing to the absolute minimum, the Adaptive RSI is reacting to price changes almost instantly.
3. Interpreting the "Negative" TickersYou have 6 tickers with a return <= 0 (CIEN, AZTA, STX, XPEV, MRNA, NFLX).Don't trim them yet. In a high-frequency system, it is normal for a portion of the portfolio to "tread water" or take small losses while the winners (TERN, SATS, STI) provide the bulk of the growth.The "Tail Protection": Your ticker_score for these losers is very low (e.g., 0.007 for NFLX). This means your scoring function is correctly identifying that these tickers aren't fitting the regime well and is preventing them from skewing the optimization.
4. Comparison Table: The Final EvolutionMetricVersion 3 (Robust)Version 4 (High-Density)Avg Trades7.119.90Coverage94.7%100%Avg Profit Factor4.643.35Stop Loss (SL Mult)1.541.26Lookback (Base Period)3616Strategy VerdictThis is a Production-Ready logic. It is fast, aggressive, and has been tested across 100% of your data with a high sample size of trades. The tight stops ($1.26 \times \text{ATR}$) mean your "Risk per Trade" is very controlled, which is exactly what you want when trading frequently.

adapt_RSI v0.2
After comparing the two sets of results, the Latest Version (adapt_RSI v0.2) is significantly better for actual trading,
despite the Earlier Version showing much higher "vanity" numbers (like a Profit Factor of 26.5).
Here is the breakdown of why the Latest Version is the superior model:
1. Statistical Reliability vs. "Luck"The biggest danger in the Earlier Version is its extremely low trade count.Earlier Version: Avg Trades/Ticker is 3.47.Latest Version: Avg Trades/Ticker is 6.05.
Why it matters: A strategy that only takes 3 trades over a backtest period is highly susceptible to "curve fitting." 
It likely found a few specific bars that worked perfectly. 
The Latest Version nearly doubles the trade density, which means the results are more likely to repeat in the future.
2. Profit Factor: Realistic vs. Theoretical
The Earlier Version shows an Avg Profit Factor of 26.53, 
but this is an illusion caused by several tickers (STI, SATS, XPEV, IREN, STX) having "inf" (infinite) Profit Factors.
An "inf" PF occurs when there are zero losing trades.
While this looks amazing, it usually indicates the model is too restrictive or "over-optimized" for that specific data window.
Latest Version: Shows an Avg PF of 5.17. While lower, this is a much more "pro" number. 
It shows the strategy can handle losses and still come out significantly ahead.
3. The "Eligibility" Gate
The models used different "min_trades" gates to determine if a ticker was successful:
Earlier Version: Used a gate of 3 trades.
Latest Version: Used a gate of 5 trades.
Winner: The Latest Version is better because it demands more evidence before calling a ticker "eligible." 
Even with this tougher requirement, it maintained a solid coverage.
4. Comparison Summary
Metric	Earlier Version	Latest Version (Winner)
Robustness	Very Low (Avg 3.4 trades)	Moderate (Avg 6.0 trades)
Profitability	Theoretical/Fragile (Many "inf" PFs)	Realistic (PF 5.17)
Min-Trades Gate	3 (Loose)	5 (Better)
Max Drawdown	Very low (due to few trades)	Normal/Expected

Key Takeaway
The Earlier Version is what we call a "Ghost Strategy." It looks perfect on paper because it barely traded, effectively "dodging" the market's volatility rather than trading through it.
The Latest Version is a "Working Strategy." It enters the market more often and maintains a Profit Factor above 5.0 even when things go wrong. In live trading, the Latest Version is far less likely to collapse the moment the market conditions change slightly.

=== OVERALL (ALL FILES) METRICS (BEST PARAMS) ===
Avg Profit Factor (raw):      5.172241776904312
Avg Profit Factor (diag):     3.747500
Avg Trades / Ticker:          6.053
Mean Ticker Score (eligible): 0.423350
Coverage (eligible/total):    15/19 = 0.789
Penalty enabled:              True (soft)
Soft penalty center/k:        0.0 / 6.0
Tail-protection ret floor/k:  0.02 / 6.0
Max-trades penalty:           max_trades=180, k=0.05
PF-floor penalty:             pf_floor=1.0, k=1.0
cooldown_bars:                1 (opt=False)  [loss-only]
time_stop_bars:               32 (0=disabled)  (opt=True)
num_neg (return<=0):          4/19
commission_rate_per_side:     0.000600
PF ROC baseline/k:            1.050 / 0.900
Trades ROC baseline/k:        7.000 / 0.200
weight_pf:                    0.450
score_power:                  1.000
min_trades gate:              5
loss_floor (scoring):         0.001
min_tp2sl_eff (constraint):   1.1000 (fixed)
threshold_mode:               fixed
Score (single fill 'next_open'): 0.423350

=== BEST PARAMS ===
atrPeriod: 30
basePeriod: 35
fastPeriod: 3
maxPeriod: 32
minPeriod: 8
slMultiplier: 2.4205183247623805
slowPeriod: 21
smooth_len: 3
time_stop: 32
tpMultiplier: 3.555808406836294
best_score (OPTUNA objective): 0.10556608861932634
fill_mode (final report): next_open
Saved per-ticker CSV to: output\per_ticker_summary.csv

=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===
ticker  profit_factor  profit_factor_diag  num_trades  ticker_score  total_return  gross_profit  gross_loss     maxdd  eligible
  TERN      18.022032            6.000000           7      0.705269      0.842193      0.729898    0.040500 -0.042226      True
   STI       9.236673            6.000000           6      0.685499      1.119380      1.125793    0.121883 -0.122978      True
  EOSE       4.710848            4.710848           7      0.660530      0.636080      0.745048    0.158156 -0.099765      True
  METC      13.770278            6.000000           5      0.643140      0.727889      0.665816    0.048352 -0.050064      True
  SNDK            inf            6.000000           5      0.578894      0.456955      0.416242    0.000000 -0.001799      True
  BITF       4.474280            4.474280           6      0.533473      0.378182      0.495777    0.110806 -0.093370      True
  CIEN       3.856291            3.856291           7      0.524570      0.368532      0.501352    0.130009 -0.103392      True
  MRNA       4.934014            4.934014           6      0.511139      0.390325      0.475746    0.096422 -0.096602      True
  BLBX       3.184065            3.184065           9      0.491827      0.328966      0.485608    0.152512 -0.150950      True
  XPEV       5.425910            5.425910           7      0.439736      0.259575      0.322614    0.059458 -0.062276      True
  SATS       4.066848            4.066848           6      0.435823      0.269111      0.372367    0.091561 -0.071153      True
   NEM       1.159914            1.159914          10      0.078432      0.007313      0.313855    0.270585 -0.175121      True
   CLS       0.830215            0.830215           6      0.031841     -0.062294      0.172406    0.207665 -0.192490      True
  ASML       0.593099            0.593099           7      0.022901     -0.089304      0.112505    0.189690 -0.115640      True
  AZTA       0.024456            0.024456           6      0.007176     -0.153695      0.003870    0.158233 -0.156451      True
  IREN      10.624710            6.000000           3      0.000000      0.206621      0.238639    0.022461 -0.024680     False
  PRAX       6.244155            6.000000           4      0.000000      0.137206      0.175779    0.028151 -0.029899     False
   STX       1.942565            1.942565           4      0.000000      0.039115      0.101231    0.052112 -0.055273     False
  NFLX       0.000000            0.000000           4      0.000000     -0.188873      0.000000    0.193820 -0.188873     False

1/15/2026, adapt_RSI v0.1
=== OVERALL (ALL FILES) METRICS (BEST PARAMS) ===
Avg Profit Factor (raw):      26.531672156129222
Avg Trades / Ticker:          3.474
Mean Ticker Score (eligible): 0.639731
Coverage (eligible/total):    16/19 = 0.842
Penalty enabled:              True (soft)
Soft penalty center/k:        -0.01 / 6.0
Tail-protection ret floor/k:  0.0 / 10.0
Max-trades penalty:           max_trades=60, k=0.08
PF-floor penalty:             pf_floor=1.0, k=3.0
cooldown_bars:                5 (opt=True)  [loss-only]
time_stop_bars:               10 (0=disabled)  (opt=False)
num_neg (return<=0):          1/19
commission_rate_per_side:     0.000600
PF ROC baseline/k:            1.400 / 1.100
Trades ROC baseline/k:        8.000 / 0.400
weight_pf:                    0.750
score_power:                  1.000
min_trades gate:              3
loss_floor (scoring):         0.001
min_tp2sl_eff (constraint):   1.1500 (auto)
Score (single fill 'next_open'): 0.639731

=== BEST PARAMS ===
atrPeriod: 25
slMultiplier: 3.8607698834699513
tpMultiplier: 2.9674264160922794
basePeriod: 44
minPeriod: 6
maxPeriod: 55
fastPeriod: 4
slowPeriod: 50
smooth_len: 5
best_score (OPTUNA objective): 0.31130463936129765
pf_cap_score_only: 5.0
cooldown (best): 5
fill_mode (report): next_open
Saved per-ticker CSV to: output\per_ticker_summary.csv

=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===
ticker  profit_factor  num_trades  ticker_score  total_return  gross_profit  gross_loss     maxdd  eligible
   STI            inf           3      0.757832      1.967133      1.461579    0.000000 -0.000600      True
  SATS            inf           3      0.755953      0.992360      0.790521    0.000000 -0.000600      True
  XPEV            inf           4      0.492585      0.174775      0.172452    0.000000 -0.000600      True
  IREN            inf           2      0.000000      3.418361      2.263064    0.000000 -0.000600     False
   STX            inf           2      0.000000      1.086523      0.926144    0.000000 -0.000600     False
  BITF     201.882768           2      0.000000      0.282491      0.287407    0.001424 -0.003220     False
  PRAX      39.762466           3      0.757837      2.448287      2.588415    0.065097 -0.066779      True
  TERN      25.021184           3      0.757735      1.474228      1.379467    0.055132 -0.056832      True
  METC      20.849440           4      0.763902      0.823667      0.755780    0.036249 -0.037983      True
  CIEN      18.699593           3      0.756478      1.046215      1.003982    0.053690 -0.055392      True
  EOSE      16.066154           3      0.716948      0.491514      0.484755    0.030172 -0.031336      True
   CLS      11.254127           3      0.753806      0.867283      1.068864    0.094975 -0.096352      True
  SNDK      10.247353           4      0.768227      1.096989      1.020740    0.099610 -0.101230      True
  ASML       8.679369           5      0.721402      0.428012      0.449704    0.051813 -0.053519      True
  BLBX       5.314900           4      0.754623      0.659704      0.752263    0.141538 -0.143083      True
   NEM       5.153222           5      0.719655      0.423592      0.541157    0.105013 -0.060514      True
  NFLX       4.467028           3      0.394620      0.120555      0.167988    0.037606 -0.038761      True
  MRNA       4.045807           4      0.363815      0.102877      0.149004    0.036829 -0.036696      True
  AZTA       0.000000           6      0.000274     -0.202543      0.000000    0.212548 -0.196850      True
