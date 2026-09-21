# Public release provenance

This repository is a clean-history public snapshot prepared from source commit
`56c78e55fa128f4f6a4b80533f5d077515408926`, whose Git tree was
`15518ce43bfdce97426aede1f364db7bc8d88923`. The deterministic source `git archive` used for the
initial inventory had SHA-256
`8e4dd48fbdb09e09eab7cf649e30b9c7f0cbacc459df7114eddd78dd4294e308`.

The public snapshot intentionally does not retain the source repository's commit history, tags,
commit messages, author email metadata, or internal release-review documents. Files were selected
and rewritten under an explicit disposition review before the new initial commit.

## Confirmatory-code custody

`frozen/PREREGISTERED_V1.sha256` records the exact scoring and derivation files carried from the
historical preregistration snapshot. `python scripts/verify-frozen.py` checks their paths and bytes.
This proves that the files in the public snapshot match the recorded snapshot. It does not provide
an independent timestamp or third-party attestation of when the original freeze occurred. The
preregistration had internal custody, as described in `PREREGISTRATION.md`.

## Evidence boundary

The synthetic evidence and analyses are included for offline verification. Third-party corpora are
not redistributed, so the external-data replay remains a separate gate. See
`THIRD_PARTY_NOTICES.md` for the data and licensing boundary and `docs/ERRATA.md` for substantive
claim changes.
