# Prior-work check: calendar information, shock origin and machine learning in single-stock VaR and ES forecasting

Search date: 20 November 2026. Each source below is labelled with how much of it was actually read. Statements about a source's contents are restricted to what was read; where only an abstract or a search-engine summary was available, this is said.

## What this changes in how the results are stated

The five claims checked are those of `docs/EARNINGS_CALENDAR_RESULTS.md` (Sections 7 to 11). None was found in full; each has a precedent that must be cited and that narrows the claim.

- **Calendar (C1).** The specification is not new. Li and Engle (1998) have a multiplicative announcement filter and a separate recursion for announcement-day shocks; Maheu and McCurdy (2004) try an earnings dummy with a one-time, non-propagating effect; Deng (2008) puts earnings dummies in a GARCH for daily stock returns. The claim is the VaR backtest, the size of the effect and the attribution. "Published comparisons omit the calendar" is too broad: Christensen, Siggaard and Veliyev (2023) include an earnings indicator. Say instead that the GARCH-benchmarked single-stock VaR comparisons checked do not include it, and name them.
- **Equation (C2).** The principle is Holzmann and Eulert (2014) and a Gaussian closed form is in Dimitriadis and Puke (2026). The claim is the Student-t form, its use before estimation, and the decay with the number of names.
- **Shock origin (C3).** Connolly and Stivers (2005) have the market-shock half in regression form. The claim is the opposite sign of the two errors and their cancellation.
- **Firm-level announcement variance (C4).** Persistence of a firm's announcement moves is in Barth and So (2014). The claim is the VaR and the comparison with flexible models. The result that the firm's history beats option-implied variance runs against Govindaraj, Jin, Livnat and Zhao, who find the option measure adds information; it should be stated as a result for this panel and this estimator until it is split by company size.
- **What the flexible models add (C5).** "Little, on equal inputs" is in Christensen et al. (2023) and Audrino and Chassot (2026) for realized-variance models. The claim is the decomposition.

Open: the full text of Deng (2008, chapter 3), which is restricted to its university and is the closest precedent for C1.

## 1. Verdict by claim

| Claim | Verdict | Closest papers | What they do and do not do |
|---|---|---|---|
| C1. Earnings calendar: VaR breached on about a quarter of announcement-reaction days; giving any model the date is worth 5 to 8 percent of quantile loss; ML-versus-GARCH VaR comparisons on single stocks omit it | PARTLY ANTICIPATED | Deng (2008, dissertation chapter 3); Li and Engle (1998); also Christensen, Siggaard and Veliyev (2023) | Deng adds earnings-announcement dummies to a GARCH for daily stock returns and reports better out-of-sample volatility forecasts than GARCH(1,1), and Li and Engle specify a multiplicative announcement filter with separate persistence for announcement-day shocks (Treasury futures, macro news); neither reports a VaR or quantile backtest, a breach rate on announcement days, or the size of the calendar's value relative to the gap between models. Christensen et al. include an earnings-announcement indicator in an ML comparison on 29 single stocks that has a VaR section, so the statement that such comparisons omit the calendar holds for the GARCH-benchmarked papers checked but not for every published single-stock ML VaR exercise. |
| C2. Closed-form expected pinball loss under a rescaled Student-t on a known subset of days, used as an ex-ante value-of-information equation, with decay in the number of portfolio names | PARTLY ANTICIPATED (principle and Gaussian analogue only) | Holzmann and Eulert (2014); Dimitriadis and Puke (2026, arXiv) | Holzmann and Eulert prove that a larger information set lowers the expected consistent score of an ideal quantile forecast and relate that score to expected shortfall, and compare portfolio-history with component-history information sets, without a closed form for any parametric family; Dimitriadis and Puke derive a closed-form expected quantile score for a misspecified forecast in a Gaussian simulation design. No source gives the Student-t mixture form, uses it to predict the value of a scheduled event before estimation, or derives the decay with the number of names. |
| C3. Shock origin: GARCH variance forecast too high after a large firm-specific shock, too low after a large market-wide shock, with the two errors cancelling when origin is ignored | PARTLY ANTICIPATED | Connolly and Stivers (2000 working paper; 2005); Li and Engle (1998) | Connolly and Stivers add the lagged squared market shock to a firm-level asymmetric GARCH for the 30 Dow stocks and find it positive and significant for 25 firms, which is the market-shock half of the claim in regression form; Li and Engle find lower persistence for announcement-day shocks in Treasury futures. Neither classifies large single-stock shocks as firm-specific versus market-wide, shows opposite-signed GARCH forecast errors for the two, or documents the cancellation. Patton and Sheppard (2015) explicitly leave idiosyncratic versus systematic jumps to future work. |
| C4. Firm-level announcement variance is persistent; a VaR from the firm's last eight announcement returns beats ML with calendar inputs and beats option-implied announcement variance | PARTLY ANTICIPATED | Barth and So (2011 working paper); Govindaraj, Jin, Livnat and Zhao (SSRN working paper); also Yu, Liu and He (2024) | Barth and So show that the average absolute announcement return over the prior three years predicts the absolute return at the next announcement; Govindaraj et al. run a regression of absolute announcement returns on an option-implied measure and on the standard deviation of past announcement-window returns and report that the option measure has incremental power; Yu et al. show that the mean of a ticker's own past post-earnings volatility beats transcript-based deep models. None builds a VaR or evaluates quantile loss, none uses a short eight-announcement window against ML models with calendar dummies, and none finds that the historical measure beats the option-implied one; Govindaraj et al. point the other way on incremental information. |
| C5. With the calendar in every model, flexible models beat the best parametric model by about 1 percent of quantile loss, half of which is a six-coefficient log-linear rescaling of GARCH volatility | PARTLY ANTICIPATED (the "ML adds little" pattern), decomposition NOT FOUND | Christensen, Siggaard and Veliyev (2023); Audrino and Chassot (2026); Dimitriadis and Puke (2026, arXiv) for recalibration | Christensen et al. report that in their VaR application ML is at best on par with HAR when both use the HAR inputs, and Audrino and Chassot report that a well-fitted HAR is not beaten by ML on over a thousand stocks; both concern realized-variance models, not GARCH-type VaR with an earnings calendar. Dimitriadis and Puke formalise linear recalibration of quantile forecasts. No source attributes a stated share of an ML gain to a low-dimensional log-linear rescaling of GARCH volatility. |

## 2. Sources on the supplied list

### Maheu and McCurdy (2004)
Maheu, J. M. and McCurdy, T. H. (2004), "News arrival, jump dynamics, and volatility components for individual stock returns", Journal of Finance 59(2), 755–793.
Read: the June 2003 CIRANO working-paper version (2003s-38), full text. The published PDF on the author's site did not download.
Content bearing on the claims. The model (GARJI) writes the conditional variance of a daily stock return as a GARCH component plus a jump component with time-varying intensity, Var(r_t | Φ_{t-1}) = σ_t² + (θ² + δ²)λ_t. A subsection on prescheduled earnings announcements reports that roughly 16 percent (IBM) and 23 percent (Intel) of inferred jump days coincide with earnings announcement days, and that only about 20 percent of scheduled announcements produce an inferred jump. The authors then let an earnings-announcement dummy (switched on for the announcement day and the day after) enter both the jump intensity and the GARCH variance, try propagating and non-propagating versions, and report that the "one-time effect" specification had the best likelihood but that the dummy effects were imprecisely estimated and did not change their conclusions. Single stocks are used (IBM, Intel and others, plus indices). There is no VaR or quantile backtest and no out-of-sample evaluation of the announcement dummy.
Relevance: C1 (a calendar dummy in a single-stock conditional variance, with a non-propagating effect, is present but treated as a robustness check and found imprecise); C3 (jump versus smooth components, not firm versus market origin).

### Li and Engle (1998)
Li, L. and Engle, R. F. (1998), "Macroeconomic announcements and volatility of Treasury futures", UCSD Department of Economics Discussion Paper 98-27.
Read: full text (eScholarship PDF).
Content. The paper defines what it calls a "filtered GARCH model": r_t = c + F_t ε_t, with ε_t | Φ_{t-1} ~ N(0, h_t), a multiplicative filter F_t = (1+ρ1 I_t)(1+ρ2 I_{t-1})(1+ρ3 I_{t+1})(1+ρ4 D4)(1+ρ5 D5) in which I_t is an announcement-day indicator and D4, D5 are Thursday and Friday dummies, and a GJR-type recursion for h_t. A second specification splits the recursion into two components, h_t^A and h_t^N, so that announcement-day and non-announcement-day shocks have their own ARCH, asymmetry and GARCH coefficients. The reported result is that announcement-day shocks have lower persistence, and that without the sign asymmetry announcement-day volatility would appear not to persist at all. In-sample and out-of-sample multi-step volatility forecasts are compared with implied and realized volatility using a squared-return regression. The asset is the Treasury bond futures contract and the events are macroeconomic releases. There is no VaR or quantile evaluation, no single stock and no earnings announcement.
Relevance: C1 (the two-multiplier idea and the announcement-specific recursion are both present in structure, for a different asset and event type); C3 (heterogeneous persistence by type of day, not by firm versus market origin).

### Deng (2008)
Deng, Qian (2008), "Three Essays on Empirical Asset Pricing", PhD dissertation, University of Illinois at Urbana-Champaign, Department of Finance; committee chair Neil D. Pearson. Handle hdl.handle.net/2142/87440.
Read: abstract only. The full text is restricted to members of the University of Illinois, and no working-paper version of the third chapter was located.
Content from the abstract. The third chapter estimates the conditional variance of daily stock returns with an extended GARCH model containing "event-related dummy variables" for earnings announcements, macroeconomic announcements and day-of-the-week effects. It reports better out-of-sample forecasting than GARCH(1,1), and that the estimated dependence on the random component (persistence) rises once the predictable components are included, so that models using only past returns understate persistence. The abstract does not say how the dummies enter (additively or multiplicatively), whether announcement-day returns are excluded from the recursion, how many stocks are used, or which loss function is used; it does not mention VaR, quantiles or expected shortfall.
Relevance: C1. This is the closest precedent for a calendar-aware GARCH on single stocks, and the persistence finding is in the same direction as an announcement-skipping recursion. Its exact overlap with C1 cannot be determined without the full text.

### Christensen, Siggaard and Veliyev (2023)
Christensen, K., Siggaard, M. and Veliyev, B. (2023), "A machine learning approach to volatility forecasting", Journal of Financial Econometrics 21(5), 1680–1727.
Read: CREATES Research Paper 2021-03 (full text) and the later arXiv version 2601.13014 (searched and read in the data, variable-importance and VaR sections).
Content. One-day-ahead realized variance is forecast for 29 Dow Jones stocks, 2001–2017, with HAR-family models and regularised regressions, tree ensembles and neural networks. The extended feature set contains an earnings-announcement indicator (EA), defined as a dummy equal to one if the company makes an earnings announcement on a given day, together with implied volatility, momentum, volume and macro indicators. In the HAR-X regression table of the working paper the EA coefficient is positive and significant. The accumulated-local-effect analysis gives EA a "low VI score", which the authors attribute to the infrequency of announcements while stating that earnings information is important for short-run prediction around the announcement. The arXiv version has a section "A Value-at-Risk analysis": VaR at 95 percent (99 percent also examined) is built by filtered historical simulation from each variance forecast, evaluated with the quantile loss and Kupiec and Christoffersen tests; the figure for Apple uses open-to-close returns. The text states that HAR is hard to beat in the VaR framework, that ML is at best on par with HAR on the HAR-only feature set, and that ML improves on the unregularised HAR-X models on the extended set. The paper does not isolate the contribution of the EA variable to forecast loss, does not report coverage on announcement days, and has no GARCH-type benchmark.
Relevance: C1 (an earnings indicator is included in a published single-stock ML comparison with a VaR section, although the returns appear to be open-to-close, which excludes the overnight reaction to most announcements); C5 (ML at best on par with the linear model in VaR on equal inputs).

### Corsi, Pirino and Renò (2010)
Corsi, F., Pirino, D. and Renò, R. (2010), "Threshold bipower variation and the impact of jumps on volatility forecasting", Journal of Econometrics 159(2), 276–288.
Read: accepted version (City Research Online), abstract and the individual-stock section.
Content. With a threshold-based jump measure inside a HAR model, lagged jumps have a positive and mostly significant effect on future realized volatility for the S&P 500, for six liquid individual stocks (2001–2005) and for bond yields, and forecasts improve "especially in periods following the occurrence of a jump". Jumps are not classified by origin (firm-specific or market-wide) or by cause (earnings or other), and no VaR is evaluated.
Relevance: C3 (evidence that large discontinuous moves do raise subsequent volatility, including in single stocks, which is the opposite sign from the firm-specific half of C3 unless origin is separated).

### Patton and Sheppard (2015)
Patton, A. J. and Sheppard, K. (2015), "Good volatility, bad volatility: signed jumps and the persistence of volatility", Review of Economics and Statistics.
Read: published PDF, searched for the passages on idiosyncratic and systematic jumps and on announcements.
Content. Realized semivariances and signed jump variation are used to forecast volatility for the S&P 500 ETF and 105 individual stocks. A footnote notes that positive semivariance behaves differently for the index and for the panel of stocks, conjectures that this may reflect idiosyncratic jumps that average out in the index, and leaves the analysis of idiosyncratic and systematic jumps for "future research". No earnings calendar is used and no VaR is evaluated.
Relevance: C3 (the question is posed and explicitly left open).

### Dubinsky, Johannes, Kaeck and Seeger (2019)
Dubinsky, A., Johannes, M., Kaeck, A. and Seeger, N. (2019), "Option pricing of earnings announcement risks", Review of Financial Studies 32(2), 646–687.
Read: the 2006 working paper by Dubinsky and Johannes ("Earnings announcements and equity options", Columbia version, full text) and the authors' 2017 conference slides. The published article was not accessible.
Content. Earnings announcements are modelled as jumps at known dates with random size; the announcement jump volatility is estimated from options by a term-structure estimator (two maturities, ex ante) and a time-series estimator (implied volatility before and after). In the 2006 paper (20 firms, 1996–2002) the authors state that the physical-measure announcement volatility cannot be estimated from the "single observation" after each announcement and therefore test predictive content cross-sectionally: the across-firm correlation between average implied announcement volatility and realized announcement-day volatility is 0.54. The 2017 slides (50 most actively traded firms per year) report a firm-level cross-sectional correlation of 85 percent, a pooled test in which analyst dispersion is not informative, and implied announcement volatility above subsequent realized volatility. Neither document read uses the firm's own past announcement returns as a competing forecast, and neither builds a VaR.
Relevance: C4 (option-implied announcement variance as a forecast; no horse race against a firm-history estimator in the versions read); C1 (the view of volatility as a sequence of predictable events).

### Holzmann and Eulert (2014)
Holzmann, H. and Eulert, M. (2014), "The role of the information set for forecasting, with applications to risk management", Annals of Applied Statistics 8(1), 595–621.
Read: arXiv version 1404.7653, full text searched and the theory and portfolio sections read.
Content. For ideal forecasts under nested information sets F ⊂ G, the expected score under any strictly consistent scoring function is weakly lower for the larger set (Theorem 1 and Corollary 2), and a Diebold–Mariano test can detect the difference. For the quantile, a particular consistent scoring function is shown to equal the conditional lower-tail expected shortfall in expectation, so that more information yields VaR forecasts with smaller average expected shortfall. An application compares univariate modelling of portfolio returns with multivariate modelling of the two component stocks (simulation from a DCC-GARCH and data). There is no closed-form expression for the expected score under any parametric family, no treatment of scheduled events, and no result on how the value of component-level information varies with the number of components.
Relevance: C2 (the ordering principle and the link to expected shortfall are established; the equation, its ex-ante use and the portfolio decay are not).

### Patell and Wolfson (1979, 1981)
Not accessed. They are described in Dubinsky and Johannes (2006) as documenting that implied volatilities rise before earnings announcements and fall afterwards and as developing a time-series estimator of the rise. No further statement is made here.

### Jones, Lamont and Lumsdaine (1998)
Jones, C. M., Lamont, O. and Lumsdaine, R. L. (1998), "Macroeconomic news and bond market volatility", Journal of Financial Economics 47(3), 315–337.
Read: abstract only (RePEc); the NBER working-paper scan could not be converted to text.
Content from the abstract. For daily Treasury bond returns around employment and PPI releases, announcement-day volatility "does not persist at all", and there is a risk premium on release dates. Li and Engle (1998) describe the same finding and qualify it with sign asymmetry. No stocks, no VaR.
Relevance: C1 and C3 by analogy (non-persistence of scheduled-announcement shocks in bonds).

### Andersen, Bollerslev and Diebold (2007)
Andersen, T. G., Bollerslev, T. and Diebold, F. X. (2007), "Roughing it up: including jump components in the measurement, modeling and forecasting of return volatility", Review of Economics and Statistics.
Read: NBER Working Paper 11775, searched and the HAR-RV-J results passage read.
Content. In HAR-RV-J regressions for the DM/$ rate, the S&P 500 and T-bond futures, the coefficient on the jump component offsets the coefficient on lagged realized volatility, so that realized volatility attributable entirely to jumps has essentially no predictive power for next-day volatility. No individual stocks, no distinction by origin of the jump, no VaR.
Relevance: C3 (non-persistence of jump variation at the index level).

### Muler and Yohai (2008); Harvey (2013); Franses and Ghijsels (1999); Charles and Darné
Muler, N. and Yohai, V. J. (2008), "Robust estimates for GARCH models", Journal of Statistical Planning and Inference 138, 2918–2940: abstract-level information only (two robust estimators with consistency and asymptotic normality; the documentation of the R package robustGarch describes the method as protecting against additive outliers). Harvey, A. C. (2013), Dynamic Models for Volatility and Heavy Tails, Cambridge University Press: catalogue and secondary descriptions only (Beta-t-EGARCH is described as resistant to observations that would be outliers under a Gaussian model). Franses and Ghijsels (1999) and the Charles and Darné papers on outliers in GARCH were not accessed and are not characterised here. From what was read, none of these uses a calendar of known event dates; they bound or remove the influence of large observations after the fact, which is a related mechanism to an announcement-skipping recursion but does not use the date.
Relevance: C1 (a mechanism that limits propagation of large shocks exists without calendar information); C3 (these methods treat all large shocks alike, irrespective of origin).

## 3. Closer or additional papers found

### Connolly and Stivers (2000, 2005)
Connolly, R. and Stivers, C. (2005), "Macroeconomic news, stock turnover, and volatility clustering in daily stock returns", Journal of Financial Research 28(2), 235–259; earlier version "Evidence on the economics of equity return volatility clustering", Econometric Society World Congress 2000 Contributed Paper 1575.
Read: the 2000 working paper in part (model section and results summary); the 2005 article at abstract level.
Content. The firm-level conditional variance for each Dow 30 stock is an asymmetric GARCH with an added term in the lagged squared market return shock: V_{i,t} = α1 + δ1 ε²_{i,t-1} + δ2 D⁻ ε²_{i,t-1} + δ3 V_{i,t-1} + δ4 ε²_{Mkt,t-1}. The coefficient δ4 is positive and significant for 25 of 30 firms; the authors relate this to Engle and Lee (1993), who conclude from a factor-GARCH that individual-stock volatility forecasts depend on both market and firm-specific shocks. The market-to-firm relation weakens following macroeconomic news announcements and does not vary systematically in the months when firms report earnings. The 2005 abstract states that the relation between the index return shock and next-period volatility decreases on days with important macroeconomic news.
Relevance: C3. This is the closest statement found that a univariate GARCH misses the effect of market shocks on a firm's next-day variance. It does not condition on large shocks, does not show that own (firm-specific) shocks are over-weighted, and does not show offsetting errors.

### Barth and So (2011 working paper; published 2014)
Barth, M. E. and So, E. C., "Risk premiums and non-diversifiable earnings announcement risk", working paper dated May 2011 (published in The Accounting Review in 2014 under a revised title).
Read: the 2011 working paper, searched and the validation regressions read.
Content. Implied announcement volatility (IAV) is extracted from option prices before each quarterly announcement. The validation regressions relate both IAV and the realized absolute announcement return |RET| to firm characteristics, including HISTRET, the average absolute market-adjusted announcement return over the three years before the quarter. HISTRET is significantly positively related to |RET| and to IAV. The paper's main result is that implied announcement volatility exceeds realized announcement volatility when announcement risk is non-diversifiable. There is no VaR, no out-of-sample forecast comparison and no regression of |RET| on IAV and HISTRET together in the passages read.
Relevance: C4 (persistence of the firm's announcement-return magnitude across its own past announcements is documented in a large sample).

### Govindaraj, Jin, Livnat and Zhao (working paper)
Govindaraj, S., Jin, W., Livnat, J. and Zhao, C., "Using option implied volatilities to predict absolute stock returns: evidence from earnings announcements and annual shareholders' meetings", SSRN 2501750.
Read: full text searched; abstract, variable definitions and the discussion of the main regression table read.
Content. The dependent variable is the absolute excess return in a three-day window around the earnings announcement (about 101,000 announcements). Regressors are an option-implied predicted event volatility (IVOL3), the 60-day pre-event return volatility (HIST) and STDXRET, the standard deviation of three-day excess returns around the firm's prior earnings announcements using all data since 1986. The option-implied measure remains significant after controlling for HIST and STDXRET; within quartiles of STDXRET absolute returns still rise with IVOL3. Straddle returns are negatively related to predicted volatility.
Relevance: C4. This is the closest horse race between option-implied announcement volatility and a firm-history announcement volatility. Its finding is that the option measure has incremental information beyond the firm-history measure; it does not test the reverse ordering, uses the full history instead of a short window, and does not evaluate quantile forecasts.

### Yu, Liu and He (2024)
Yu, D., Liu, Z. and He, H. (2024), "Same company, same signal: the role of identity in earnings call transcripts", arXiv 2412.18029.
Read: abstract, introduction and the sections defining the baselines.
Content. For post-earnings volatility prediction over windows of 3 to 30 days, the authors assemble a data set with 20 earnings events per ticker, observe that each ticker has a stable post-earnings volatility distribution, and propose "training-free baselines": PEV (mean of historical post-earnings volatility) and STPEV (the same using only the same ticker's past earnings). These baselines outperform transcript-based neural and large-language-model predictors on their data set and on two established benchmarks. The paper does not consider VaR, option-implied measures, or GARCH-type models with calendar inputs.
Relevance: C4 (a firm's own announcement history beating machine-learning predictors is shown for a different target and a different class of ML models).

### Milian (2023)
Milian, J. A. (2023), "The efficiency of weekly option prices around earnings announcements", Journal of Risk and Financial Management 16(5), 270.
Read: abstract only (RePEc); the publisher page refused automated access.
Content from the abstract. The option-implied earnings move is the at-the-money weekly straddle price over the stock price. Straddle returns are higher when the firm's historical earnings-announcement move is high relative to the implied move and lower when it is low. The abstract reports no association of straddle returns with historical earnings-announcement volatility or implied volatility taken separately.
Relevance: C4 (the firm's history carries information not fully reflected in the option-implied move; evidence is in straddle returns, not in forecast loss).

### Leung and Santoli (2014)
Leung, T. and Santoli, M. (2014), "Accounting for earnings announcements in the pricing of equity options", arXiv 1412.8414.
Read: abstract and the section comparing implied and historical announcement-jump distributions.
Content. For IBM, the historical announcement-jump volatility estimated from all announcement returns since 1994 is close to the jump volatility calibrated from options under an extended Heston model at five dates in 2012–2013, and below the simple term-structure estimator. One firm, no forecast evaluation.
Relevance: C4 (a firm-history estimate of announcement volatility set beside the option-implied one).

### Dimitriadis and Puke (2026)
Dimitriadis, T. and Puke, M. (2026), "Statistical inference for score decompositions", arXiv 2603.04275.
Read: abstract, application section (searched) and the appendix passage on the Gaussian quantile score.
Content. A scoring function is decomposed into miscalibration, discrimination and uncertainty through a linear recalibration of the forecast, with inference valid for quantiles and a stated connection to the Mincer–Zarnowitz regression. In a Gaussian simulation design the authors derive closed forms: the expected quantile score of the ideal forecast is σ φ(z_α), and for a misspecified forecast with Gaussian error of mean m and standard deviation s it is m(Φ(m/s) + α − 1) + s φ(m/s). The application covers variance and 1 percent VaR forecasts for S&P 500 E-mini futures from GARCH, GJR-GARCH with Student-t errors, HAR, MIDAS and historical simulation. No single stocks, no event calendar, no Student-t closed form, no portfolio result.
Relevance: C2 (a closed-form expected pinball loss under misspecification exists for the Gaussian case, in a simulation appendix); C5 (linear recalibration of a quantile forecast as a formal object, the nearest methodological precedent for attributing part of a gain to rescaling).

### Kim, Schonberger, Wasley and Yang (2024)
Kim, J., Schonberger, B., Wasley, C. and Yang, Y. (2024), "Forecasting market volatility: the role of earnings announcements", The Accounting Review 99(4).
Read: abstract only.
Content from the abstract. In an exponential GARCH for index returns that controls for macroeconomic announcements, earnings-announcement information aggregated across firms forecasts market volatility at daily and weekly horizons, and out-of-sample forecasts with that information are better. Market level, not single stocks; no VaR.
Relevance: C1 (earnings information inside a GARCH-family model, at the aggregate level).

### Léber and Egyed (2025)
Léber, D. and Egyed, B. (2025), "The sentiment augmented GARCH-LSTM hybrid model for Value-at-Risk forecasting", Computational Economics.
Read: open-access repository PDF, abstract and data description; text searched for earnings and announcement terms.
Content. One-day-ahead VaR at the 2.5 percent level is forecast for S&P 500 constituents, 2019–2024, with GARCH, GARCH-LSTM hybrids and versions augmented with news and Twitter sentiment, evaluated by standard VaR tests and loss functions. The word "earnings" occurs only in the literature review; no announcement dates enter any model and no announcement-day results are reported.
Relevance: C1 (a published ML-versus-GARCH VaR comparison on single stocks at the same 2.5 percent level that omits the earnings calendar).

### Goel, Pasricha and Kanniainen (2025)
Goel, A., Pasricha, P. and Kanniainen, J. (2025), "Time-series foundation AI model for Value-at-Risk forecasting", arXiv 2410.11773 (version 7).
Read: abstract and data section; text searched for earnings and announcement terms (none found).
Content. A fine-tuned foundation model is compared with GARCH, GAS and non-parametric models on the S&P 100 index and 91 constituents, 2005–2023, at the 1, 2.5, 5 and 10 percent levels. The abstract states that on the quantile score the foundation model performs comparably to the best econometric model, GAS. Inputs are past returns only.
Relevance: C1 (calendar omitted); C5 (ML roughly level with the best parametric model on quantile score).

### Ardia, Bluteau, Boudt and Catania (2018)
Ardia, D., Bluteau, K., Boudt, K. and Catania, L. (2018), "Forecasting risk with Markov-switching GARCH models: a large-scale performance study", International Journal of Forecasting 34(4), 733–747.
Read: published PDF, data section; text searched for earnings and announcement terms (none found).
Content. VaR, ES and left-tail forecasts from single-regime and Markov-switching GARCH models are compared on 426 S&P 500 stocks, indices and exchange rates. No event dates are used.
Relevance: C1 (a large single-stock VaR study without the calendar; not an ML paper).

### Chronopoulos, Raftapostolos and Kapetanios (2024)
Chronopoulos, I., Raftapostolos, A. and Kapetanios, G. (2024), "Forecasting Value-at-Risk using deep neural network quantile regression", Journal of Financial Econometrics 22(3), 636–669.
Read: 2022 working-paper version ("Deep quantile regression"), abstract and empirical section.
Content. The deep quantile estimator is applied to the S&P 500 index, with GARCH, RiskMetrics, CAViaR and stochastic-volatility quantities as covariates. It is not a single-stock study.
Relevance: C1 and C5 (index only; outside the single-stock claim).

### Audrino and Chassot (2026); Branco, Rubesam and Zevallos (2024)
Audrino, F. and Chassot, J. (2026), "HARd to beat: the overlooked impact of rolling windows in the era of machine learning", International Journal of Forecasting 42(2), 330–343 (arXiv 2406.08041): abstract read; arXiv text searched for earnings and VaR terms (no earnings term; no VaR evaluation). The abstract states that a HAR model with a suitable fitting scheme is not beaten by ML models on more than a thousand stocks. Branco, R., Rubesam, A. and Zevallos, M. (2024), "Forecasting realized volatility: does anything beat linear models?", Journal of Empirical Finance 78: abstract only; ten stock indices; no evidence that nonlinear ML models statistically outperform linear ones in general.
Relevance: C5 (the general finding that flexible models add little over a well-specified simple model is established for realized-variance forecasting; neither paper concerns quantile loss with a calendar or a recalibration decomposition).

### Bu, Hizmeri, Izzeldin, Murphy and Tsionas (2023)
Bu, R., Hizmeri, R., Izzeldin, M., Murphy, A. and Tsionas, M. G. (2023), "The contribution of jump signs and activity to forecasting stock price volatility", Journal of Empirical Finance 70, 144–164.
Read: Dallas Fed Working Paper 1902 (revised 2022), abstract and the co-jump passages.
Content. For SPY and 20 stocks, co-jumps account for about 9 percent of the jump component, so that most jumps are idiosyncratic and described as stock-specific reactions to news; the small number of systematic jumps is given as a reason no extended HAR model dominates. Jump origin is not used as a forecasting variable.
Relevance: C3 (documents the mix of origins; does not test differential persistence).

### Peng and Yao (2022)
Peng, W. and Yao, C. (2022), "Co-jumps, co-jump tests, and volatility forecasting: Monte Carlo and empirical evidence", Journal of Risk and Financial Management 15(8), 334.
Read: abstract only.
Content from the abstract. Jumps are split into idiosyncratic jumps and co-jumps; in HAR-type forecasts of sector ETF volatility the predictive content of co-jumps is higher than that of idiosyncratic jumps.
Relevance: C3 (the same qualitative ordering as C3, for sector ETFs and realized volatility; no GARCH forecast-error analysis and no single stocks in the forecasting target).

### Other items read or located
- Engle, R. F., Hansen, M. K., Karagozoglu, A. K. and Lunde, A. (2021), "News and idiosyncratic volatility: the public information processing hypothesis", Journal of Financial Econometrics 19(1), 1–38. Read: abstract, and the 2012 CREATES working paper (Engle, Hansen and Lunde, "And now, the rest of the news") in part. Firm-specific news counts for 28 large US firms, including an earnings-surprise category with contemporaneous and lagged effects, explain part of firm-specific volatility. News arrival is measured ex post; there is no ex-ante calendar forecast and no VaR.
- Hwang, S. and Satchell, S. (2001), "GARCH model with cross-sectional market volatility; GARCHX models", working paper WP01-16. Read: abstract. Cross-sectional market volatility improves the specification of daily return volatility in UK and US data but does not necessarily improve forecasts. Relevant to the market-shock half of C3.
- Hizmeri, R., Izzeldin, M., Nolte, I. and Pappas, V. (2022), "A generalized heterogeneous autoregressive model using market information", Quantitative Finance 22(8), 1513–1534. Abstract only: adding market realized measures improves individual-stock volatility forecasts by about 9.8 percentage points. Relevant to the market-shock half of C3.
- Xiao, X. and Zhou, C. (2018), "The decomposition of jump risks in individual stock returns", Journal of Empirical Finance 47, 207–228. Abstract only: a GARCH-jump model with systematic and idiosyncratic jumps and diffusive volatility, used for expected returns.
- Gao, C., Xing, Y. and Zhang, X. (2018), "Anticipating uncertainty: straddles around earnings announcements", Journal of Financial and Quantitative Analysis. Read in part: straddle returns around announcements are positive on average and higher for firms with more volatile past earnings surprises and higher historical return moments; no firm-history announcement-return forecast.
- Lundquist, D. P. and Eck, D. J. (2024), "Volatility forecasting using similarity-based parameter correction and aggregated shock information", arXiv 2406.08738. Abstract and application read: a GARCH forecast immediately after a news shock is adjusted additively using estimated shock effects from similar past events in other series; the application is an ETF around US presidential elections. Related in spirit to C4 (using analogous past events), not about earnings or VaR.
- Yee, B. (2025), "Beyond the event horizon: peak risk-adjusted performance in post-event markets", MPRA Paper 125993. Model section read: event volatility is the square root of a GJR-GARCH variance times a deterministic three-phase multiplier around earnings and FDA dates; the object is a return-to-variance ratio, with no VaR backtest.
- Adamski, Ł. and Ślepaczuk, R. (2026), "When the Fed speaks: dynamics and forecasts of the volatility surface", arXiv 2608.10693. Abstract read: an FOMC-date feature is added to a neural model of the implied-volatility surface. A calendar dummy as an ML input, for a different target.
- Rahimikia, E., Zohren, S. and Poon, S.-H., "Realised volatility forecasting: machine learning via financial word embedding", arXiv 2108.00480, and Zhang, Zhang, Cucuringu and Qian, "Volatility forecasting with machine learning and intraday commonality", arXiv 2202.08962. Texts searched: the first uses news text in which earnings-related phrases appear; the second contains no earnings term. Both target realized volatility, not VaR.
- Ormaniec, W., Pitera, M., Safarveisi, S. and Schmidt, T. (2022), "Estimating value at risk: LSTM vs. GARCH", arXiv 2207.10539. Text searched: simulated data and indices; no single-stock earnings treatment.
- A 2023 master's thesis at BI Norwegian Business School, "Forecasting realized volatility with earnings announcements and overnight returns" (abstract via search summary only): earnings announcements moderately improve realized-volatility forecasts for S&P 500 stocks and gradient boosting outperforms HAR.

### What was searched for and not found
No paper was found that reports VaR or ES breach rates of standard models on earnings-announcement reaction days for single stocks, or that expresses the value of the announcement date as a share of quantile loss or as a share of the gain of the best model over GARCH-t. No paper was found that adds an earnings-announcement dummy to a CAViaR or quantile-regression VaR model. No paper was found that gives a closed-form expected pinball loss under a rescaled Student-t on a subset of days or a rule for how the value of event information decays with the number of portfolio names. No paper was found that conditions GARCH forecast errors on whether a large shock was firm-specific or market-wide and reports opposite signs. No paper was found in which a firm-history announcement variance outperforms an option-implied one in forecast loss. These are statements about the searches run on the date above, which used a general web search engine and open repositories, not a citation database.

## 4. Not accessed or only partly accessed

- Deng (2008), dissertation chapter 3: full text restricted to the University of Illinois; abstract only. This is the main gap for C1.
- Dubinsky, Johannes, Kaeck and Seeger (2019), published article: not accessible; the 2006 working paper and 2017 slides were read instead, and the published version may contain predictive regressions not seen here.
- Maheu and McCurdy (2004), published article: the 2003 working paper was read instead.
- Milian (2023): publisher refused automated access; abstract only.
- Jones, Lamont and Lumsdaine (1998): abstract only.
- Patell and Wolfson (1979, 1981); Franses and Ghijsels (1999); Charles and Darné: not accessed.
- Muler and Yohai (2008); Harvey (2013): abstract or catalogue level only.
- Barth and So: the 2011 working paper was read; the 2014 published version was not.
- Connolly and Stivers (2005), Kim et al. (2024), Peng and Yao (2022), Xiao and Zhou (2018), Hizmeri et al. (2022), Branco et al. (2024), Bekaert and Wu (2000), Gaglianone et al. (2011), Bégin, Dorion and Gauthier (2020), Jeon, McCurdy and Zhao (2022): abstracts only; the last four were located as background and are not relied on above.
- Two non-academic items appeared only as search-engine summaries and could not be opened: a practitioner blog post describing an "event-aware" GARCH with a release-day variance multiplier for economic announcements (simulation only, per the summary), and a university course exercise that pairs single-stock VaR hit ratios with an earnings-day flag. Neither could be verified.
- Vendor risk-model documentation (Barra, Axioma, Northfield, Bloomberg) on earnings-date adjustments was searched for and not found.
