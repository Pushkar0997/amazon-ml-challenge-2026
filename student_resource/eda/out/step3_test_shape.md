## Step 3 — Test shape vs train

All country labels in all files are within {US, India, France}; France appears only in test.

| split | country | S1 rows | S2 rows | S3 rows | S1 country share | S2 country share | S3 country share | S2/S1 | S3/S1 | (S2+S3)/S1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | US | 1,323,633 | 3,016,817 | 3,170,056 | 60.0% | 59.9% | 60.0% | 2.279 | 2.395 | 4.674 |
| train | India | 883,188 | 2,017,799 | 2,115,547 | 40.0% | 40.1% | 40.0% | 2.285 | 2.395 | 4.680 |
| train | all | 2,206,821 | 5,034,616 | 5,285,603 | 100.0% | 100.0% | 100.0% | 2.281 | 2.395 | 4.677 |
| test | US | 663,106 | 1,871,330 | 1,945,701 | 38.3% | 38.3% | 38.3% | 2.822 | 2.934 | 5.756 |
| test | India | 809,986 | 2,312,565 | 2,405,000 | 46.8% | 47.3% | 47.3% | 2.855 | 2.969 | 5.824 |
| test | France | 259,452 | 703,378 | 731,615 | 15.0% | 14.4% | 14.4% | 2.711 | 2.820 | 5.531 |
| test | all | 1,732,544 | 4,887,273 | 5,082,316 | 100.0% | 100.0% | 100.0% | 2.821 | 2.933 | 5.754 |

Reference (train GT): matched S2 per S1 = 1.674, matched S3 per S1 = 1.788; the remaining S2/S3 rows are unmatched (step 2).

