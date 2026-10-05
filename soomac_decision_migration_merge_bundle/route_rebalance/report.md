# Route rebalance report

## BEFORE

total = 2789
task = 2528
general = 229
mixed = 32

## GENERATED

general raw = 300
mixed raw = 250
review pass = 550
review reject = 0
validator fail = 0
dedup removed = 0
supplement final = 550

## AFTER

total = 3339
task = 2528
general = 529
mixed = 282

percentage:
task = 75.71%
general = 15.84%
mixed = 8.45%

## Validation

- Decision schema violations rejected = 0
- exact duplicates removed = 0
- near duplicates rejected = 0
- original base rows preserved byte-for-value = 2789
- schema fields added = 0

## Hard eval

- total = 495
- general/task minimal pairs = 110
- mixed boundary = 110
- reference mixed = 35
- unsupported mixed = 35
- query vs general = 70
- commit mixed = 25
- exact train leakage = 0
