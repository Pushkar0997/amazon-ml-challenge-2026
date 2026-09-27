# Preprocessing evaluation: BEFORE (eda/common.py) vs AFTER (src.preprocess)

Sampled 200,000 train Source-1 entities (seed=42, requested sample-s1=200000, max-scan-rows=unlimited). For each, compared its true S1-S2/S1-S3 match (when present) against one random same-country non-match.

| country/pair/kind | n | mean name Jaccard | name Jaccard=0 % | name_core exact-eq % | mean addr-component Jaccard | share addr number % |
|---|---|---|---|---|---|---|
| India|S1-S2|non-pair|BEFORE | 80,298 | 0.051 | 75.0% | 0.0% | 0.012 | 6.9% |
| India|S1-S2|non-pair|AFTER | 80,298 | 0.003 | 98.8% | 0.0% | 0.012 | 7.4% |
| India|S1-S2|pair|BEFORE | 69,724 | 0.462 | 36.9% | 15.7% | 0.635 | 83.7% |
| India|S1-S2|pair|AFTER | 69,724 | 0.509 | 37.2% | 38.0% | 0.638 | 85.0% |
| India|S1-S3|non-pair|BEFORE | 80,298 | 0.060 | 71.3% | 0.0% | 0.005 | 6.0% |
| India|S1-S3|non-pair|AFTER | 80,298 | 0.003 | 98.7% | 0.0% | 0.005 | 6.5% |
| India|S1-S3|pair|BEFORE | 70,545 | 0.591 | 19.0% | 21.0% | 0.433 | 81.3% |
| India|S1-S3|pair|AFTER | 70,545 | 0.640 | 19.5% | 47.0% | 0.434 | 82.8% |
| US|S1-S2|non-pair|BEFORE | 119,702 | 0.014 | 91.1% | 0.0% | 0.009 | 0.2% |
| US|S1-S2|non-pair|AFTER | 119,702 | 0.003 | 98.4% | 0.0% | 0.009 | 0.2% |
| US|S1-S2|pair|BEFORE | 103,912 | 0.708 | 8.0% | 30.4% | 0.530 | 76.5% |
| US|S1-S2|pair|AFTER | 103,912 | 0.760 | 8.4% | 53.1% | 0.653 | 80.2% |
| US|S1-S3|non-pair|BEFORE | 119,702 | 0.014 | 91.0% | 0.0% | 0.001 | 0.3% |
| US|S1-S3|non-pair|AFTER | 119,702 | 0.003 | 98.5% | 0.0% | 0.001 | 0.3% |
| US|S1-S3|pair|BEFORE | 105,329 | 0.712 | 8.0% | 32.1% | 0.243 | 78.6% |
| US|S1-S3|pair|AFTER | 105,329 | 0.755 | 8.4% | 52.0% | 0.317 | 81.9% |

## India pairs where the S2/S3 name is non-Latin (name_translit, AFTER only)

| kind | n | mean translit-token Jaccard | Jaccard=0 % | translit exact-eq % |
|---|---|---|---|---|
| pair | 30,866 | 0.095 | 70.9% | 1.2% |
| non-pair | 27,600 | 0.001 | 99.8% | 0.0% |

## Throughput

Metric computation (normalize() called twice per comparison, BEFORE+AFTER): 1,465 S1 entities/sec over 136.5s.

See AGENT_LOG.md for the CLI's own end-to-end preprocess throughput and the full-run projection.
