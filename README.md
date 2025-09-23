# Research-Internship
MS-ARCH-X volatility model

six-week internship at CREST–ENSAE under the supervision of RaphaëlThabut(PhD).The
project explores the idea that equity markets do not behave in the same way
at all times, but instead follow different such as stability, crisis, or recov-
ery. The main objective was therefore to study volatility and dependence
across assets in a regime-aware framework. The results do not constitute
a definitive proof that the chosen regime classification is optimal, but they
provide several consistent indications that regimes matter and that this type
of approach is useful.
An important part of the work focuses on a regime-aware GARCH model.
By estimating volatility parameters separately for each regime and using
them to standardize returns, I was able to build regime-specific correlation
matrices across indices and sectors. This shows the expected rise and ho-
mogenization of correlations during crises, and also highlights how standard-
ization clarifies the underlying structure beyond simple volatility effects. To
reduce biases, I also fitted GARCH models locally within regimes, which
confirmed the differences between periods.
The study also includes baseline ARCH(1) and GARCH(1,1) models fit-
ted by regime, together with standard residual checks, as well as comparisons
between raw and standardized correlations and rolling correlation plots with
regime markers. I also applied K-means clustering on return and volatility
features.
Finally, I worked on extending the analysis to MS-ARCH-X models with
exogenous variables. This involved setting up an estimation framework and
running initial simulations to test whether LASSO can identify relevant
volatility drivers. This part of the work is not finished, but it should even-
tually allow the regime-based analysis to be complemented.
