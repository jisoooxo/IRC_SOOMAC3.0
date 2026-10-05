# SOOMAC Decision migration + merge bundle

실행:
```bash
./run_migration_merge.sh /home/roma/IRC_SOOMAC3.0
```

기본 legacy 입력:
`/home/roma/IRC_SOOMAC3.0/decision_adapter_v225_root_cause_review_20260930/dataset_contract/decision_train.jsonl`

출력:
- migrated_legacy_pass.jsonl
- migrated_legacy_review_required.jsonl
- migrated_legacy_fail.jsonl
- merged_raw.jsonl
- dedup_removed.jsonl
- final_validator_fail.jsonl
- final_train.jsonl
- manifest.json
- migration_report.md

원본 legacy 파일은 overwrite하지 않는다.
