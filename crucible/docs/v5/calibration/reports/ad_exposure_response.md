# ad_exposure_response@1 held-out evaluation report

Domain: advertising_response. Dataset: criteo_uplift_v2.1. Identification: randomized_controlled_trial.
Evaluation plan: `docs/v5/calibration/plans/E5_ad_exposure_plan.json` (sha256 76162a59da04, commit 790e3ff5f); fit at commit 5880c2d72 in 524 s.

## Estimand

the complier effect of being displayed an advertiser's campaign during the test period on the log-odds of visiting the advertiser, as a function of the displayed individual's organic (no-ad) visit log-odds. Identified by the randomised eligibility (treatment) as an instrument for display (exposure): exclusion restriction (eligible users who were never displayed are unaffected) and one-sided non-compliance (no control user is displayed; checked in the conversion record: exposed_in_control = 0)

## Splits

| split | definition | rows | users |
|---|---|---|---|
| fit | validation rows with r % 10 in {1, 3, 5, 7, 9} (val-A): stratum edges, nuisance rates and candidate parameters | 1397959 |  |
| val | validation rows with r % 10 in {2, 4, 6, 8} (val-B): selection | 1118367 |  |
| test | the whole test split of criteo.splits() (2,795,919 users): confirmation, once | 2795919 |  |
| explore | validation rows whose rank r (0-based position in the ascending list of validation row indices of criteo.splits()) has r % 10 == 0; inspected before this plan; excluded from everything below | 279592 |  |

## Held-out evaluation

Every candidate on the validation and test splits (primary metric `excess_log_loss`, lower is better).

| candidate | complexity | params | val excess_log_loss | test auc | test brier | test ece | test excess_log_loss | test log_loss | test mean_obs | test mean_pred |
|---|---|---|---|---|---|---|---|---|---|---|
| B0_no_effect | baseline: no effect | 0 | 0.0044 | 0.8003 | 0.2292 | 0.2194 | 0.0041 | 0.7052 | 0.4127 | 0.1934 |
| B1_constant_difference | baseline: one difference | 1 | 0.0032 | 0.8003 | 0.2098 | 0.1695 | 0.0028 | 0.6061 | 0.4127 | 0.2432 |
| M1_constant_log_odds | one log-odds shift | 1 | 0.0001 | 0.8003 | 0.1711 | 0.0236 | 0.0000 | 0.5378 | 0.4127 | 0.3979 |
| M2_linear_log_odds | linear log-odds shift | 2 | 0.0001 | 0.8003 | 0.1709 | 0.0206 | 0.0000 | 0.5364 | 0.4127 | 0.3995 |

## Baseline comparison

Selected on validation: **M1_constant_log_odds**; best simple baseline on test: **B1_constant_difference**; kept: **M1_constant_log_odds**. selected on validation and confirmed on test: no simple baseline is better.

Test excess_log_loss: kept 0.0000, baseline 0.0028, difference -0.0027 (95% user-cluster bootstrap interval -0.0031 to -0.0022).

## Calibration

* ece: 0.0236
* brier: 0.1711
* mean_pred: 0.3979
* mean_obs: 0.4127
* itt_weighted_mse: 0.0000

## Further held-out checks

### strata

```json
{
 "edges_p0_hat": [
  0.0006562810807667601,
  0.001047718654041213,
  0.0026148043127936556,
  0.00437426652663452,
  0.009044754232677123,
  0.02392422160558378,
  0.10829046069955901
 ],
 "K": 8,
 "rule": "10 equal-mass bins of p0_hat on the fit split (edges stored); then, from the lowest bin upwards, a bin is merged into its upper neighbour while it has fewer than 300 treated exposed fit rows or its complier baseline moment estimate on the fit split is not strictly inside (0, 1); the top bin is merged downwards under the same condition"
}
```

### per_stratum

```json
{
 "fit": [
  {
   "stratum": 0,
   "rows": 419064,
   "exposed_rows": 559,
   "exposure_rate": 0.001570727704939222,
   "control_visit_rate": 0.0006489600810408686,
   "never_taker_visit_rate": 0.00037993172486188776,
   "complier_baseline": 0.1716561849755109,
   "complier_baseline_unclipped": 0.1716561849755109,
   "exposed_visit_rate": 0.1556350626118068,
   "itt": -2.516482076089137e-05,
   "wald_late": -0.01602112236370409,
   "complier_log_odds_shift": -0.11713611493171205
  },
  {
   "stratum": 1,
   "rows": 140020,
   "exposed_rows": 835,
   "exposure_rate": 0.007026667676487171,
   "control_visit_rate": 0.001179968848822391,
   "never_taker_visit_rate": 0.0006525534331090358,
   "complier_baseline": 0.07571166252947448,
   "complier_baseline_unclipped": 0.07571166252947448,
   "exposed_visit_rate": 0.058682634730538925,
   "itt": -0.00011965731919678208,
   "wald_late": -0.017029027798935573,
   "complier_log_odds_shift": -0.27304462886948944
  },
  {
   "stratum": 2,
   "rows": 139840,
   "exposed_rows": 3239,
   "exposure_rate": 0.027266604933075176,
   "control_visit_rate": 0.00166270783847981,
   "never_taker_visit_rate": 0.0013500532232520704,
   "complier_baseline": 0.012816629865086953,
   "complier_baseline_unclipped": 0.012816629865086953,
   "exposed_visit_rate": 0.03643099722136462,
   "itt": 0.0006438836254481302,
   "wald_late": 0.02361436735627767,
   "complier_log_odds_shift": 1.0688881538220087
  },
  {
   "stratum": 3,
   "rows": 139848,
   "exposed_rows": 1916,
   "exposure_rate": 0.016170823310967634,
   "control_visit_rate": 0.004072461732902682,
   "never_taker_visit_rate": 0.0038003242714615376,
   "complier_baseline": 0.02062924239194905,
   "complier_baseline_unclipped": 0.02062924239194905,
   "exposed_visit_rate": 0.07045929018789145,
   "itt": 0.0008057928984852572,
   "wald_late": 0.049830047795942425,
   "complier_log_odds_shift": 1.2805451737157454
  },
  {
   "stratum": 4,
   "rows": 139781,
   "exposed_rows": 2757,
   "exposure_rate": 0.02328586631530938,
   "control_visit_rate": 0.006827853902632933,
   "never_taker_visit_rate": 0.006303992528601448,
   "complier_baseline": 0.028800959870850178,
   "complier_baseline_unclipped": 0.028800959870850178,
   "exposed_visit_rate": 0.10591222343126587,
   "itt": 0.0017956025746724271,
   "wald_late": 0.07711126356041567,
   "complier_log_odds_shift": 1.3849294323258698
  },
  {
   "stratum": 5,
   "rows": 139814,
   "exposed_rows": 4205,
   "exposure_rate": 0.03541440326098857,
   "control_visit_rate": 0.017080229634198415,
   "never_taker_visit_rate": 0.01365557224181888,
   "complier_baseline": 0.1103579610168412,
   "complier_baseline_unclipped": 0.1103579610168412,
   "exposed_visit_rate": 0.17526753864447087,
   "itt": 0.0022987339576053187,
   "wald_late": 0.06490957762762967,
   "complier_log_odds_shift": 0.5383444592859576
  },
  {
   "stratum": 6,
   "rows": 139791,
   "exposed_rows": 10727,
   "exposure_rate": 0.08986119139168824,
   "control_visit_rate": 0.04980899206582427,
   "never_taker_visit_rate": 0.05030097748651584,
   "complier_baseline": 0.04482602870081489,
   "complier_baseline_unclipped": 0.04482602870081489,
   "exposed_visit_rate": 0.295982101239862,
   "itt": 0.02256918390361605,
   "wald_late": 0.25115607253904715,
   "complier_log_odds_shift": 2.192599727797999
  },
  {
   "stratum": 7,
   "rows": 139801,
   "exposed_rows": 18410,
   "exposure_rate": 0.15358689213878715,
   "control_visit_rate": 0.3210093307916123,
   "never_taker_visit_rate": 0.3083769478695408,
   "complier_baseline": 0.3906260431286363,
   "complier_baseline_unclipped": 0.3906260431286363,
   "exposed_visit_rate": 0.7110809342748506,
   "itt": 0.049217670801820435,
   "wald_late": 0.32045489114621456,
   "complier_log_odds_shift": 1.3453210944911798
  }
 ],
 "val": [
  {
   "stratum": 0,
   "rows": 333376,
   "exposed_rows": 413,
   "exposure_rate": 0.0014596889060108787,
   "control_visit_rate": 0.00039651856698189893,
   "never_taker_visit_rate": 0.0003539522306069573,
   "complier_baseline": 0.029515190765514624,
   "complier_baseline_unclipped": 0.029515190765514624,
   "exposed_visit_rate": 0.15012106537530268,
   "itt": 0.00017604705716764673,
   "wald_late": 0.12060587460978804,
   "complier_log_odds_shift": 1.7592388395381553
  },
  {
   "stratum": 1,
   "rows": 112949,
   "exposed_rows": 634,
   "exposure_rate": 0.006610639584593248,
   "control_visit_rate": 0.0011148272017837235,
   "never_taker_visit_rate": 0.0005772944831639936,
   "complier_baseline": 0.08189056406036245,
   "complier_baseline_unclipped": 0.08189056406036245,
   "exposed_visit_rate": 0.07570977917981073,
   "itt": -4.085894119523065e-05,
   "wald_late": -0.00618078488055172,
   "complier_log_odds_shift": -0.08518595517382277
  },
  {
   "stratum": 2,
   "rows": 111836,
   "exposed_rows": 2585,
   "exposure_rate": 0.02727023377500211,
   "control_visit_rate": 0.002229523586012673,
   "never_taker_visit_rate": 0.0014966325767024195,
   "complier_baseline": 0.028371760063950997,
   "complier_baseline_unclipped": 0.028371760063950997,
   "exposed_visit_rate": 0.0367504835589942,
   "itt": 0.00022848974844593122,
   "wald_late": 0.008378723495043216,
   "complier_log_odds_shift": 0.2674178777266598
  },
  {
   "stratum": 3,
   "rows": 111896,
   "exposed_rows": 1527,
   "exposure_rate": 0.016118476608680966,
   "control_visit_rate": 0.0036713286713286712,
   "never_taker_visit_rate": 0.0036155306890965464,
   "complier_baseline": 0.007077271124422397,
   "complier_baseline_unclipped": 0.007077271124422397,
   "exposed_visit_rate": 0.07269155206286837,
   "itt": 0.001057602252501763,
   "wald_late": 0.06561428093844598,
   "complier_log_odds_shift": 2.397703373237684
  },
  {
   "stratum": 4,
   "rows": 110928,
   "exposed_rows": 2184,
   "exposure_rate": 0.023192838255439805,
   "control_visit_rate": 0.006801503490245212,
   "never_taker_visit_rate": 0.006185925660176337,
   "complier_baseline": 0.03272764613824215,
   "complier_
```

### conversion_per_stratum

```json
{
 "fit": [
  {
   "stratum": 0,
   "complier_conversion_baseline": 0.0001,
   "complier_conversion_baseline_unclipped": -0.0024453157903350686,
   "exposed_conversion_rate": 0.016100178890876567,
   "itt": 2.91299222975821e-05
  },
  {
   "stratum": 1,
   "complier_conversion_baseline": 0.004321879674831674,
   "complier_conversion_baseline_unclipped": 0.004321879674831674,
   "exposed_conversion_rate": 0.0023952095808383233,
   "itt": -1.3538070472717578e-05
  },
  {
   "stratum": 2,
   "complier_conversion_baseline": 0.0033744029673977854,
   "complier_conversion_baseline_unclipped": 0.0033744029673977854,
   "exposed_conversion_rate": 0.002778635381290522,
   "itt": -1.6244559402318605e-05
  },
  {
   "stratum": 3,
   "complier_conversion_baseline": 0.004223665543913017,
   "complier_conversion_baseline_unclipped": 0.004223665543913017,
   "exposed_conversion_rate": 0.0041753653444676405,
   "itt": -7.810539911156739e-07
  },
  {
   "stratum": 4,
   "complier_conversion_baseline": 0.009900661258046695,
   "complier_conversion_baseline_unclipped": 0.009900661258046695,
   "exposed_conversion_rate": 0.00544069640914037,
   "itt": -0.00010385414524261168
  },
  {
   "stratum": 5,
   "complier_conversion_baseline": 0.0001,
   "complier_conversion_baseline_unclipped": -0.004890747402212497,
   "exposed_conversion_rate": 0.013555291319857313,
   "itt": 0.0006532554538711905
  },
  {
   "stratum": 6,
   "complier_conversion_baseline": 0.003349785328893305,
   "complier_conversion_baseline_unclipped": 0.003349785328893305,
   "exposed_conversion_rate": 0.019390323482800412,
   "itt": 0.001441421869073924
  },
  {
   "stratum": 7,
   "complier_conversion_baseline": 0.04195008656213522,
   "complier_conversion_baseline_unclipped": 0.04195008656213522,
   "exposed_conversion_rate": 0.10380228136882129,
   "itt": 0.009499686372321747
  }
 ],
 "test": [
  {
   "stratum": 0,
   "complier_conversion_baseline": 0.0001,
   "complier_conversion_baseline_unclipped": -0.002207559305433717,
   "exposed_conversion_rate": 0.02834008097165992,
   "itt": 4.251794917153301e-05
  },
  {
   "stratum": 1,
   "complier_conversion_baseline": 0.0010220847947965503,
   "complier_conversion_baseline_unclipped": 0.0010220847947965503,
   "exposed_conversion_rate": 0.009380863039399626,
   "itt": 5.589788093764901e-05
  },
  {
   "stratum": 2,
   "complier_conversion_baseline": 0.001745471725092585,
   "complier_conversion_baseline_unclipped": 0.001745471725092585,
   "exposed_conversion_rate": 0.0032438808611029196,
   "itt": 4.283839930707139e-05
  },
  {
   "stratum": 3,
   "complier_conversion_baseline": 0.00019324747508039648,
   "complier_conversion_baseline_unclipped": 0.00019324747508039648,
   "exposed_conversion_rate": 0.004929942916450441,
   "itt": 7.724463348301169e-05
  },
  {
   "stratum": 4,
   "complier_conversion_baseline": 0.002774447620682214,
   "complier_conversion_baseline_unclipped": 0.002774447620682214,
   "exposed_conversion_rate": 0.008282716731087797,
   "itt": 0.00012651686632268204
  },
  {
   "stratum": 5,
   "complier_conversion_baseline": 0.003389127324513249,
   "complier_conversion_baseline_unclipped": 0.003389127324513249,
   "exposed_conversion_rate": 0.015153254505797268,
   "itt": 0.00043365442944325527
  },
  {
   "stratum": 6,
   "complier_conversion_baseline": 0.009572999410939772,
   "complier_conversion_baseline_unclipped": 0.009572999410939772,
   "exposed_conversion_rate": 0.021867945823927766,
   "itt": 0.0010939496312593437
  },
  {
   "stratum": 7,
   "complier_conversion_baseline": 0.04665210235334444,
   "complier_conversion_baseline_unclipped": 0.04665210235334444,
   "exposed_conversion_rate": 0.10560825517964152,
   "itt": 0.009081985822432394
  }
 ]
}
```

### negative_effect_strata_test

```json
[]
```

### negative_effects

```json
"no stratum's test interval lies entirely below 0: the floor at 0 agrees with the data"
```

### incremental_visit_conversion_ratio

```json
{
 "fit": {
  "rho": 2.7605437161313673,
  "ci95": [
   2.0258061665935276,
   3.471882678722655
  ],
  "conversions_per_incremental_visit": 0.1296876887603022,
  "conversions_per_control_visit": 0.0469790382244143,
  "counts": [
   [
    201480,
    7729,
    381
   ],
   [
    1130829,
    53881,
    3659
   ]
  ]
 },
 "test": {
  "rho": 2.441921880396867,
  "ci95": [
   2.075738537864644,
   2.9324412321666173
  ],
  "conversions_per_incremental_visit": 0.12040433688138343,
  "conversions_per_control_visit": 0.049307202596429914,
  "counts": [
   [
    403984,
    15232,
    790
   ],
   [
    2260794,
    107702,
    7417
   ]
  ]
 }
}
```

### served_conversion_ratio

```json
2.7605437161313673
```

### pooled_wald_late_fit

```json
0.2709750463314458
```

### kept_shift_at_organic_probability

```json
{
 "0.001": 1.390830496023541,
 "0.01": 1.390830496023541,
 "0.05": 1.390830496023541,
 "0.1": 1.390830496023541,
 "0.3": 1.390830496023541,
 "0.5": 1.390830496023541,
 "0.8": 1.390830496023541
}
```

### m2_center_logit

```json
-1.9675735623488122
```

### theta

```json
{
 "B0_no_effect": [],
 "B1_constant_difference": [
  0.04983303192943396
 ],
 "M1_constant_log_odds": [
  1.390830496023541
 ],
 "M2_linear_log_odds": [
  1.411800805065313,
  -0.019334540490061045,
  -1.9675735623488122
 ]
}
```

### kept_theta_ci95_fit_bootstrap

```json
[
 [
  1.2212788293907306,
  1.6066303500834458
 ]
]
```

### fit_split_metrics

```json
{
 "B0_no_effect": {
  "n": 42648,
  "log_loss": 0.6994139603478651,
  "brier": 0.23104086297621326,
  "ece": 0.2167971332846,
  "auc": 0.7818895808018285,
  "mean_pred": 0.19827301337650063,
  "mean_obs": 0.4146501594447571,
  "excess_log_loss": 0.0041557980318850255
 },
 "B1_constant_difference": {
  "n": 42648,
  "log_loss": 0.6197545781991134,
  "brier": 0.2119587355898548,
  "ece": 0.17225292706333561,
  "auc": 0.7818895808018285,
  "mean_pred": 0.2481060453059346,
  "mean_obs": 0.4146501594447571,
  "excess_log_loss": 0.002974394258491976
 },
 "M1_constant_log_odds": {
  "n": 42648,
  "log_loss": 0.5396658367648816,
  "brier": 0.17898979032327678,
  "ece": 0.06304502318952629,
  "auc": 0.7818895808018285,
  "mean_pred": 0.40861274431517935,
  "mean_obs": 0.4146501594447571,
  "excess_log_loss": 0.00024159685329471026
 },
 "M2_linear_log_odds": {
  "n": 42648,
  "log_loss": 0.5388591686019136,
  "brier": 0.17883882522017985,
  "ece": 0.06230098221283018,
  "auc": 0.7818895808018285,
  "mean_pred": 0.41073468458646184,
  "mean_obs": 0.4146501594447571,
  "excess_log_loss": 0.00024126021947702404
 }
}
```


## Identification

Randomised eligibility (treatment) is the instrument for display (exposure). Assumptions: exclusion restriction (eligibility affects visits only through display), one-sided non-compliance (no control user is displayed: exposed_in_control = 0 in the conversion record) and monotonicity (trivial here). Under them, each stratum's complier baseline and complier visit rate under display are identified, and so is their log-odds difference. The score that defines strata uses pre-treatment features only and was fitted on the train split, so strata are balanced between the arms on the fit, validation and test splits.

## Extrapolation limits

Criteo Uplift v2.1 (13,979,592 users of pooled randomised incrementality tests of Criteo display advertising; advertisers, countries and dates undisclosed; released 2018; sub-sampled non-uniformly by the publisher). The simulator may use this model for: the visit probability of an individual who is displayed (and, through ad_response@2, attends) an advertiser's display or feed campaign during a campaign period, as a log-odds shift of that individual's own organic visit probability. It may NOT infer: (1) effects for compliers unlike Criteo's - the displayed users were chosen by Criteo's bidder (exposure rises from about 0.1% to 15% of eligible users with the baseline score), so the effect is that of a targeted display on users a bidder reached; (2) effects of ad content, creative quality, price or relevance (not observed; ad_response@2 adds relevance and repetition only through a separate attention model and a declared assumption); (3) frequency, wear-in, wear-out, carry-over or repeated campaigns (no timestamps, one period per user); (4) negative effects (floored at 0); (5) absolute rates: the publisher's sub-sampling distorts levels; the log-odds shift is invariant to sub-sampling that depends on the outcome or on the arm only, not to sub-sampling that depends on exposure; (6) search advertising (a click on a sponsored result is a different decision). Strata are bins of a pre-treatment score; within-stratum heterogeneity is averaged, and the complier baseline of each stratum is a moment estimate (noisy where exposure is rare; clipped to [1e-4, 0.999]).

## Simulator feature mapping

| input | simulator quantity |
|---|---|
| organic_logit | log-odds of the displayed individual's organic (no-ad) probability of visiting the advertiser during the campaign period: in the data the stratum's complier baseline, in the simulator the individual's own organic visit probability (ad_response@2 input organic_visit_probability) |
