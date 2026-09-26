## Step 1 — Inventory

Parsing: raw bytes split on `\n`, fields split on `\t`, no quote handling. `empty` = zero-length/whitespace-only field; `null-like` = whole field is one of NULL/<NULL>/N/A/None/nan/-/… (case-insensitive). Every column is a string; `entity_id` checked against `^S[123]-\d+$`.

### Files

| file | MB | raw `\n` count (incl. header) | data rows | well-formed rows | malformed (≠ header field count) | bad id format | id prefix ≠ file | duplicate ids | `\r` bytes | rows containing `"` | country counts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train_source1 | 200.3 | 2,206,822 | 2,206,821 | 2,206,821 | 0 | 0 | 0 | 0 | 0 | 4 | India:883,188, US:1,323,633 |
| train_source2 | 466.6 | 5,034,617 | 5,034,616 | 5,034,616 | 0 | 0 | 0 | 0 | 0 | 6 | India:2,017,799, US:3,016,817 |
| train_source3 | 480.4 | 5,285,604 | 5,285,603 | 5,285,603 | 0 | 0 | 0 | 0 | 0 | 0 | India:2,115,547, US:3,170,056 |
| train_ground_truth | 121.1 | 2,206,822 | 2,206,821 | 2,206,821 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| test_source1 | 166.9 | 1,732,545 | 1,732,544 | 1,732,544 | 0 | 0 | 0 | 0 | 0 | 134 | France:259,452, India:809,986, US:663,106 |
| test_source2 | 485.9 | 4,887,274 | 4,887,273 | 4,887,273 | 0 | 0 | 0 | 0 | 0 | 349 | France:703,378, India:2,312,565, US:1,871,330 |
| test_source3 | 482.6 | 5,082,317 | 5,082,316 | 5,082,316 | 0 | 0 | 0 | 0 | 0 | 330 | France:731,615, India:2,405,000, US:1,945,701 |

### Columns (empties / null-like; % of well-formed rows)

| file | column | dtype | empty | null-like |
|---|---|---|---|---|
| train_source1 | entity_id | str | 0 (0.00%) | 0 (0.000%) |
| train_source1 | business_name | str | 0 (0.00%) | 0 (0.000%) |
| train_source1 | business_address | str | 0 (0.00%) | 0 (0.000%) |
| train_source1 | country | str | 0 (0.00%) | 0 (0.000%) |
| train_source2 | entity_id | str | 0 (0.00%) | 0 (0.000%) |
| train_source2 | business_name | str | 0 (0.00%) | 6 (0.000%) |
| train_source2 | business_address | str | 168,967 (3.36%) | 0 (0.000%) |
| train_source2 | country | str | 0 (0.00%) | 0 (0.000%) |
| train_source3 | entity_id | str | 0 (0.00%) | 0 (0.000%) |
| train_source3 | business_name | str | 0 (0.00%) | 18 (0.000%) |
| train_source3 | business_address | str | 175,916 (3.33%) | 0 (0.000%) |
| train_source3 | country | str | 0 (0.00%) | 0 (0.000%) |
| train_ground_truth | source1_entity_id | str | 0 (0.00%) | 0 (0.000%) |
| train_ground_truth | matched_entity_ids | str | 123,247 (5.58%) | 0 (0.000%) |
| test_source1 | entity_id | str | 0 (0.00%) | 0 (0.000%) |
| test_source1 | business_name | str | 0 (0.00%) | 0 (0.000%) |
| test_source1 | business_address | str | 0 (0.00%) | 0 (0.000%) |
| test_source1 | country | str | 0 (0.00%) | 0 (0.000%) |
| test_source2 | entity_id | str | 0 (0.00%) | 0 (0.000%) |
| test_source2 | business_name | str | 0 (0.00%) | 49 (0.001%) |
| test_source2 | business_address | str | 129,408 (2.65%) | 0 (0.000%) |
| test_source2 | country | str | 0 (0.00%) | 0 (0.000%) |
| test_source3 | entity_id | str | 0 (0.00%) | 0 (0.000%) |
| test_source3 | business_name | str | 0 (0.00%) | 61 (0.001%) |
| test_source3 | business_address | str | 136,098 (2.68%) | 0 (0.000%) |
| test_source3 | country | str | 0 (0.00%) | 0 (0.000%) |

### Malformed row examples

None — every row in every file has exactly the header's field count.

