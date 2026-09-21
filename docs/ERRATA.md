# Errata and claim changes

This file records substantive corrections to the analyses and claims. The current README, paper,
tests, and committed evidence contain the corrected values. Dates identify when a correction was
made in the source project; this public repository intentionally starts with a clean history.

## Benchmark and baseline corrections

- The original unprotected baseline retained a confidence floor. It was not an all-controls-off
  baseline, so the associated improvement figures were withdrawn and recomputed.
- Early deterministic repeats at temperature zero were described as replication. Current model-run
  figures use ten draws per cell at temperature 0.7 and report instability.
- Run counts once treated three configurations over the same cells as three times as many samples.
  Current counts use distinct cells.
- The first control-space sweep covered 82 of 328 configurations. The current exhaustive analysis
  covers all 328.
- Wilson intervals over correlated draws were replaced by cell-clustered intervals. The effective
  clustering unit is the nine approval-authority pairs, not all 18 scenarios.

## Third-party benchmark corrections

- Mind2Web-SC and EICU-AC comparisons were recomputed against policies that use the same evaluation
  set and metric as the published results. EICU-AC's full-set no-model floor is 84.49% with the
  recorded resampling interval, rather than the earlier 64.8% validation-split base rate.
- The EICU-AC reproduction on 128 validation items must not be compared with published figures on
  316 records. It is retained only as a split-specific reproduction.
- BeaverTails sampling now shuffles before capping instead of taking the first rows. The winning
  floor policy changed from response-only to prompt-only, and the floor moved from 66.4% to 67.4%.
- R-Judge is evaluated on F1, its reported metric. An always-positive classifier reaches 69.04% F1;
  an accuracy majority is not the relevant floor.
- The claim that none of five papers reported a trivial baseline was withdrawn. R-Judge reports a
  random baseline; four of the five examined papers do not report a comparable no-model floor.
- A claim that one reproduced LLaMA-Guard3 configuration scored below the Mind2Web-SC floor was
  narrowed after the benchmark rules were supplied to the model. It then matched, but did not
  exceed, the 70.5% floor.

## Adaptation-headroom corrections

- BeaverTails and ToxicChat cannot support the original headroom analysis because their proposed
  grouping annotations partially encode the label. The implementation now refuses retained units
  that contain only one class.
- EICU-AC's recorded positive gap is a nominal association, not evidence of held-out adaptive
  benefit. It is sensitive to minimum group size and cross-validation shuffle seed, and its
  exchangeability, dependence, fitting, and selection assumptions remain unresolved.
- Null concentration is a diagnostic, not an automatic veto for a permutation result. The legacy
  field remains in the evidence without controlling the verdict.
- Headroom is reported with its permutation reference and sensitivity information. A positive
  maximum-minus-maximum gap is expected under selection noise and is not sufficient by itself.

## Statistical and implementation corrections

- The preregistered H1 comparison was in-sample and incapable of failing at the deriver's fixpoint.
  Its equality is an algebraic identity, not confirmatory evidence.
- Factorial and vocabulary families now include simultaneous bands. Several individually
  significant effects do not survive the family-wise comparison.
- A dominance-resampling check and an earlier paired-policy check were incapable of failing. Both
  were replaced with tests that demonstrably reject constructed counterexamples.
- The majority baseline in `floors.py` can now be fitted on held-out training labels. When no
  training set is supplied, the result explicitly warns that an evaluation-label majority requires
  the answer key.
- `floors.py` now validates empty benchmarks, unhashable labels and units, policy name collisions,
  prediction lengths, and minimum permutation counts with named errors.
- CI now compares regenerated evidence with committed files, verifies declared run completeness,
  checks generated figures, and verifies the frozen confirmatory implementation against
  `frozen/PREREGISTERED_V1.sha256`.

## Scope that remains unchanged

The internal synthetic result concerns five model configurations on two small scenario surfaces.
The external datasets are not redistributed, so their full replay is a separate gate:
`make verify-external` fails when the corpora are absent. None of the external analyses establishes
a deployable adaptive policy improvement or a result about safety benchmarks in general.
