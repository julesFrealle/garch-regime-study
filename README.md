# Research Internship — MS-ARCH-X Volatility Modelling

This repository contains the material from my six-week research internship at **CREST–ENSAE**, supervised by **Raphaël Thabut (PhD)**.  
The project focused on modelling financial time series volatility and dependence across **different market regimes** (stability, crisis, recovery).  

The work explored several complementary directions:

- **ARCH/GARCH models:** implementation and estimation of volatility dynamics, both globally and within regimes.  
- **MS-ARCH-X model:** extension of Markov-switching ARCH to include exogenous variables, aiming to capture regime-specific volatility shifts.  
- **Regime-aware analysis:** estimation of regime-specific GARCH models, standardization of returns, and comparison of raw vs. standardized correlation matrices.  
- **Local GARCH fits:** regime-constrained rolling estimations to confirm structural differences across periods.  
- **Unsupervised methods:** use of clustering (e.g., K-means) on volatility and correlation features to validate the regime classification.  
- **Variable selection:** simulation experiments with LASSO to test its ability to identify relevant exogenous drivers of volatility.  

Together, these methods provided a deeper understanding of how correlations and volatility structures change across regimes, especially during crises when co-movements strengthen and become more homogeneous.  

---

## How to Use

All final materials are in the folder:

FINAL REPORT AND NOTEBOOK/


This folder contains:
- `ARCH_Jules.pdf` — the full technical internship report.  
- `Project_Jules.ipynb` — a cleaned Jupyter notebook reproducing the main results.  

Non organised/experimental code and functions can be found in the `Jules/` directory. Data examples are stored under `data/`. Code and Notebooks provided by **Raphaël Thabut (PhD)** in 'Raphaël_files/'.

---

## Acknowledgments

This work was carried out at **CREST–ENSAE (Paris-Saclay)**, in collaboration with École Polytechnique.  
Supervisor : **Raphaël Thabut (PhD)**.  
