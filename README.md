# Pricing at the Ceiling — replication code

Code for "Pricing at the Ceiling: Loan Prices, a Binding Rate Cap and What a Loan Guarantor Does Not Record".

## Data (not redistributed here; all public)

| File | Source | Size (bytes) | SHA-256 |
|---|---|---:|---|
| `FOIA_7a_FY2000_FY2009_asof_260630.csv` | SBA 7(a) FOIA, https://data.sba.gov/dataset/7a-504-foia | 318,425,094 | `66674e18a700fbba0378c25118c291ac2759784a550c643cab5747eced763d6a` |
| `FOIA_7a_FY2010_FY2019_asof_260630.csv` | same | 255,101,999 | `01a3e2c7988a6f4052e53f218a309feb2ec2fe42887bebdc0fa94ac8b1024ade` |
| `dprime.csv` | FRED series DPRIME (downloaded 2026-09) | 297,740 | `d65066a2c753b88b510912f19bd56eef7349fc1e4314df0645d97e4463457173` |
| `2020_Gaz_zcta_national.txt` | U.S. Census 2020 Gazetteer, ZCTA file | 6,662,135 | `036421adec9bdd8fdbf51dd39a6464617246547f32a32936a7b9ddb45bbfb3c9` |

SBA replaces its FOIA files every quarter. Later releases update loan statuses, so results from a later release will differ slightly. The fingerprints identify the release used in the paper (as of 30 June 2026).

FDIC total assets are downloaded at run time from the FDIC BankFind API by `28_paperB_extensions.py`.

Place the data in `~/sba_data`, or set the environment variable `SBA_DIR`.

## Environment

Python 3.11; pandas 3.0, numpy 2.4, scikit-learn 1.9, xgboost 3.0.5.

## Order of execution

| Script | Produces |
|---|---|
| `00_fetch_sba.sh all` | Downloads the SBA files |
| `07_residual_spread.py` | Loan-level sample and residual spread (Online Appendix A.1) |
| `08_bootstrap_and_decile.py` | Online Appendix Tables A1–A2 |
| `09_rate_cap.py` | Ceilings, `at_cap`, Section 2.2 |
| `10_cap_informativeness.py` | Tables 1–2 |
| `28_paperB_extensions.py` | Tables 3 and 5, Online Appendix Table A3, placebo (Section 5.4) |
| `32_paperB_more_robustness.py` | Table 6 (ceiling definitions and lender interactions) |
| `35_paperB_review_round1.py` | Table 4 Panel A, Table 6 (payment-relief and horizon rows), Sections 2.2, 3.1 and 5.6 |
| `36_paperB_rate_gradient.py` | Table 4 Panel B |

Comments in the scripts are in Chinese.

## Licence

Code: MIT. SBA data are U.S. Government Works.
