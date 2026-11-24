Run: experiment05_ft_supcon_w0.001_both_s42. Share = fraction of all test positive (admission, code) pairs, procedures included in the denominator.

| tier | codes | share of test positives | baseline AP | model fused AP | lift |
|---|---|---|---|---|---|
| narrow (original list) | 107 | 6.8% | 0.120 | 0.320 | 2.66× |
| broad extra codes only | 101 | 3.9% | 0.060 | 0.187 | 3.12× |
| BROAD = narrow + extra | 208 | 10.6% | 0.102 | 0.273 | 2.68× |
| dialysis-catheter related | 3 | 0.5% | 0.050 | 0.196 | 3.95× |
| no direct radiological sign | 3069 | 78.6% | 0.156 | 0.217 | 1.39× |
| all diagnoses | 3280 | 89.8% | 0.151 | 0.223 | 1.48× |

No-direct-sign tier by chapter:

| chapter | codes | share of test positives | baseline AP | model fused AP | lift |
|---|---|---|---|---|---|
| V codes (supplementary) | 336 | 14.3% | 0.147 | 0.230 | 1.57× |
| endocrine/metabolic | 141 | 11.0% | 0.200 | 0.284 | 1.42× |
| circulatory | 213 | 9.7% | 0.220 | 0.318 | 1.44× |
| symptoms/ill-defined | 184 | 6.4% | 0.036 | 0.068 | 1.88× |
| digestive | 244 | 5.5% | 0.112 | 0.188 | 1.67× |
| nervous/sense | 341 | 4.9% | 0.059 | 0.128 | 2.16× |
| mental | 134 | 4.8% | 0.153 | 0.181 | 1.18× |
| genitourinary | 137 | 4.4% | 0.149 | 0.221 | 1.48× |
| E codes (external causes) | 221 | 4.4% | 0.173 | 0.277 | 1.60× |
| blood | 10 | 3.7% | 0.204 | 0.307 | 1.50× |
| injury/poisoning | 355 | 2.6% | 0.051 | 0.165 | 3.25× |
| musculoskeletal | 173 | 2.1% | 0.022 | 0.031 | 1.43× |
| infectious | 155 | 1.9% | 0.023 | 0.043 | 1.89× |
| neoplasms | 228 | 1.3% | 0.007 | 0.034 | 4.71× |
| skin | 101 | 1.0% | 0.012 | 0.024 | 2.03× |
| respiratory | 44 | 0.2% | 0.004 | 0.007 | 1.79× |
| congenital | 18 | 0.2% | 0.003 | 0.003 | 1.05× |
| pregnancy | 31 | 0.1% | 0.001 | 0.014 | 19.51× |
| perinatal | 3 | 0.0% | nan | nan | nan× |

Top Broad extra codes by share of positives:

- 0.61%  4939  (category)
- 0.36%  V4581  Aortocoronary bypass status
- 0.25%  4168  Other chronic pulmonary heart diseases
- 0.21%  73301  Senile osteoporosis
- 0.21%  4241  Aortic valve disorders
- 0.18%  4932  (category)
- 0.18%  4240  Mitral valve disorders
- 0.16%  4049  (category)
- 0.12%  4254  Other primary cardiomyopathies
- 0.12%  V4501  Cardiac pacemaker in situ
- 0.12%  4259  Secondary cardiomyopathy, unspecified
- 0.12%  1985  Secondary malignant neoplasm of bone and bone marrow
- 0.11%  V4502  Automatic implantable cardiac defibrillator in situ
- 0.10%  5533  Diaphragmatic hernia without mention of obstruction or gangrene
- 0.10%  40291  Unspecified hypertensive heart disease with heart failure
- 0.08%  51919  Other diseases of trachea and bronchus
- 0.07%  V433  Heart valve replaced by other means
- 0.07%  4239  Unspecified disease of pericardium
- 0.07%  99731  Ventilator associated pneumonia
- 0.06%  4414  Abdominal aneurysm without mention of rupture
- 0.04%  73313  Pathologic fracture of vertebrae
- 0.03%  V5331  Fitting and adjustment of cardiac pacemaker
- 0.03%  73739  Other kyphoscoliosis and scoliosis
- 0.03%  51909  Other tracheostomy complications
- 0.03%  4940  Bronchiectasis without acute exacerbation

Top No-direct-sign codes by share of positives:

- 2.29%  4019  Unspecified essential hypertension
- 2.12%  2724  Other and unspecified hyperlipidemia
- 1.84%  V1582  Personal history of tobacco use
- 1.79%  E849  (category)
- 1.53%  285  (category)
- 1.46%  53081  Esophageal reflux
- 1.27%  5849  Acute kidney failure, unspecified
- 1.25%  311  Depressive disorder, not elsewhere classified
- 1.19%  42731  Atrial fibrillation
- 1.01%  4039  (category)
- 0.99%  V498  (category)
- 0.98%  30000  Anxiety state, unspecified
- 0.89%  2500  (category)
- 0.85%  2449  Unspecified acquired hypothyroidism
- 0.84%  V5867  Long-term (current) use of insulin
- 0.80%  V5861  Long-term (current) use of anticoagulants
- 0.68%  2761  Hyposmolality and/or hyponatremia
- 0.65%  5990  Urinary tract infection, site not specified
- 0.60%  32723  Obstructive sleep apnea (adult)(pediatric)
- 0.59%  99591  Sepsis
- 0.58%  287  (category)
- 0.56%  27800  Obesity, unspecified
- 0.55%  3051  Tobacco use disorder
- 0.54%  2762  Acidosis
- 0.51%  288  (category)
