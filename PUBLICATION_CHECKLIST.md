# Private main synchronization checklist

- User requested current project/code synchronization and explicitly selected main.
- Existing repository visibility must remain private.
- Append to the existing main history; no force push or replacement history.
- Use a source-only allowlist, not the mixed local workspace.
- Preserve 160 library source files byte-for-byte against runtime-v4.
- Include matching synthetic tests and the portable C-pilot policy/adapter.
- Exclude raw data, prepared arrays, predictions, weights, cloud controllers,
  credentials, SSH details, chat attachments and local environments.
- Retain the MIT license and upstream notices.
- Record fresh local CPU verification separately from historical/cloud results.
- Verify remote branch head and every published blob after the update.

This is a private source update, not authorization to publish private research
data, change repository visibility, certify source rights or rerun experiments.
