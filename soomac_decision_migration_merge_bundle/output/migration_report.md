# Decision dataset migration + merge report

legacy original = 2111
migration pass = 2111
migration review required = 0
migration fail = 0

supplement = 685

merged raw = 2796
dedup removed = 7
validator fail = 0
final train = 2789

route task = 2111
route general = 0
route mixed = 0

mentions added rows = 1432
mentions absent rows = 679

Notes:
- Legacy v2.2.5 is a task-only Decision dataset; route is migrated to task when a legacy semantic target exists.
- Existing semantic target fields are preserved exactly.
- mentions only use literal current-message surfaces from a conservative lexicon.
- Reference-only expressions are not emitted as mentions.
- Exact full-context duplicates are removed.
- Same surface with different state/context is preserved.
- Minimal pairs are preserved.
