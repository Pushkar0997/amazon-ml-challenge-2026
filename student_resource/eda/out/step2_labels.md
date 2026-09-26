## Step 2 — Label structure (train)

GT rows: 2,206,821; unique S1 ids in GT: 2,206,821; total true pairs: 7,638,365 (S2: 3,693,619, S3: 3,944,746). Rows with a repeated id inside the list: 0. Ids in lists without S2-/S3- prefix: 0.

### Singletons and mean match counts (country = S1 record's country)

| country | S1 entities | singletons | singleton % | mean |matches| | mean #S2 | mean #S3 | only-S2 % | only-S3 % | both % |
|---|---|---|---|---|---|---|---|---|---|
| all | 2,206,821 | 123,247 | 5.58% | 3.461 | 1.674 | 1.788 | 6.5% | 7.5% | 80.5% |
| US | 1,323,633 | 73,896 | 5.58% | 3.459 | 1.672 | 1.787 | 6.5% | 7.5% | 80.5% |
| India | 883,188 | 49,351 | 5.59% | 3.465 | 1.676 | 1.788 | 6.5% | 7.4% | 80.5% |

### Match-set size distribution (count (% of S1 entities in that country))

| country | counted | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|---|
| all | S2+S3 | 123,247 (5.6%) | 119,157 (5.4%) | 375,212 (17.0%) | 530,841 (24.1%) | 484,115 (21.9%) | 574,249 (26.0%) |
| all | S2 only count | 287,745 (13.0%) | 789,108 (35.8%) | 652,779 (29.6%) | 333,957 (15.1%) | 119,078 (5.4%) | 24,154 (1.1%) |
| all | S3 only count | 266,276 (12.1%) | 716,417 (32.5%) | 668,375 (30.3%) | 372,443 (16.9%) | 145,116 (6.6%) | 38,194 (1.7%) |
| US | S2+S3 | 73,896 (5.6%) | 71,689 (5.4%) | 225,285 (17.0%) | 318,876 (24.1%) | 290,446 (21.9%) | 343,441 (25.9%) |
| US | S2 only count | 172,744 (13.1%) | 474,144 (35.8%) | 391,281 (29.6%) | 199,890 (15.1%) | 71,172 (5.4%) | 14,402 (1.1%) |
| US | S3 only count | 159,723 (12.1%) | 430,004 (32.5%) | 400,693 (30.3%) | 223,397 (16.9%) | 86,903 (6.6%) | 22,913 (1.7%) |
| India | S2+S3 | 49,351 (5.6%) | 47,468 (5.4%) | 149,927 (17.0%) | 211,965 (24.0%) | 193,669 (21.9%) | 230,808 (26.1%) |
| India | S2 only count | 115,001 (13.0%) | 314,964 (35.7%) | 261,498 (29.6%) | 134,067 (15.2%) | 47,906 (5.4%) | 9,752 (1.1%) |
| India | S3 only count | 106,553 (12.1%) | 286,413 (32.4%) | 267,682 (30.3%) | 149,046 (16.9%) | 58,213 (6.6%) | 15,281 (1.7%) |

Max |matches| for one S1: 11 (S2 max 5, S3 max 6). Count with ≥10 matches: 571.

### Is the S1 → S2/S3 mapping many-to-one?

|  | S2 | S3 |
|---|---|---|
| distinct ids in GT lists | 3,693,619 | 3,944,746 |
| ids appearing in >1 S1 list | 0 | 0 |
| max #S1 lists for one id | 1 | 1 |

### S2/S3 records that match no S1 (by the record's own country)

| source | country | records | unmatched | unmatched % |
|---|---|---|---|---|
| S2 | US | 3,016,817 | 803,743 | 26.64% |
| S2 | India | 2,017,799 | 537,254 | 26.63% |
| S3 | US | 3,170,056 | 804,608 | 25.38% |
| S3 | India | 2,115,547 | 536,249 | 25.35% |

### Id integrity

| check | count |
|---|---|
| GT S1 ids not in train_source1 | 0 |
| train_source1 ids missing from GT | 0 |
| GT S2 ids not in train_source2 | 0 |
| GT S3 ids not in train_source3 | 0 |
| train S1 ids also in test S1 | 0 |
| train S2 ids also in test S2 | 0 |
| train S3 ids also in test S3 | 0 |
| S2 ids whose numeric part also exists as an S3 id (train) | 26,801 |

### Country disagreement within true pairs

Pairs with a differing country: 0 / 7,638,365 (0.000%).

