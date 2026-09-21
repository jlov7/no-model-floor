# Limitations

Everything wrong with this work, as far as I can tell. `PAPER.md` section 5 has the short version.

## The numbers

**Always-abstain scores 7778.** The 18 cells are 14 refusals, 2 acts, 2 asks. A policy that ignores
its input entirely scores 7778 of 10000. Any claim on this benchmark has to beat that number to mean
anything, and none of the five unguarded models does.

**Two trials at temperature zero are one trial.** Earlier versions of this work reported identical
results across trials as replication. They were deterministic repeats of a fixed-seed pipeline. All
current figures are ten independent draws per cell at temperature 0.7. The instability that reveals
is substantial: `qwen3.5:2b` gives a different answer on all 18 cells across draws, `qwen3.8-27b` on
14, `gemma4:12b` on 2.

**Run counts are not sample sizes.** Earlier counts multiplied by three configurations over the same
cells, inflating apparent n threefold. Current figures count distinct cells.

**The unprotected baseline was not unprotected.** It retained a confidence floor of 7000, which is a
safety control and is correct on most cells. Every improvement figure measured against it was wrong.
Fixed; the affected figures were regenerated.

## The constructs

**Unsafe execution is nearly definitional.** It is computed on the post-scaffold decision against the
same fields the controls read. Once a control is enabled, zero is close to a logical identity rather
than a measurement. The model's unsafe proposals persist regardless.

**The label rule is the control set.** The function defining correct answers, the three controls and
the coverage taxonomy are one conditional expressed three times. Different defensible labelling, for
instance treating unresolved approval under expired authority as ask rather than refuse, would break
the one-control-one-failure correspondence and much of the structure.

**The oracle rule wins with no model.** A three-clause conditional over two fields present verbatim
in the observation scores 10000 with zero unsafe executions. Full marks on this benchmark does not
require a model at all.

## The scope

Five models, four from one family, two runtimes, mixed quantization, one machine, one synthetic task,
one action, three controls, 18 cells. Nothing here supports a claim about models in general, about
scale, or about agent safety outside this setting.

Authority status is handed to the model as a literal enum field. There is no inference, ambiguity,
indirection or adversarial pressure. Calling this an authority failure in the sense that matters for
real agents would be a stretch.

Execution happens in a no-effect runtime. No action ever had a consequence.

## The process

The original run was not preregistered. The scenario set, prompt, provider configuration, deriver
rule and promotion criterion all changed as problems surfaced, and each change moved results
favourably.
Specifically: the scenario set was replaced because the original made every model look perfect; the
baseline was switched to all-controls-off because that created the headroom the deriver then found;
the deriver was widened from unsafe-only to any-repairable failure after a model stalled; a
safety-credit rule was added after candidates missed the improvement floor.

Each change is defensible on its own. Together they mean the original results are exploratory,
produced by the same process that reports them.

A prospectively specified confirmatory run was then done against frozen scoring and derivation
code, on a surface written after the freeze by a separate context with no access to any result. All
three hypotheses held. Its custody limits are real: the hypotheses were committed to a local
repository that had no remote at the time, so no external service timestamped them; the same author
specified and accepted the surface, ran the experiment, analysed it, and reported it. This is
internal evidence of good faith, not third-party verification.

The preregistration text said no file under `src/`, `experiments/` or the single-file
reproduction could change. The
intent was that the scoring and derivation logic stay frozen, which it did and which
`scripts/verify-frozen.py` confirms against the public checksum manifest. New files were added to implement the new surface and its runner. The
written rule was broader than the intent and should have said so.

## The statistics

**Multiple comparisons are only partly handled, and the boundary is stated here rather than
hidden.** Two families get a simultaneous adjustment: the seven-vocabulary sweep's 21 contrasts and
the factorial's 36 intervals. Everything else in this repository — per-model cell-clustered
intervals, the floor intervals, headroom null means, McNemar tests, rank intervals — is read at a
nominal 95% each with no family-wise correction, across roughly two dozen interval or test
statements in the documents. The defensible parts: most headline claims rest on identities,
paired decision counts or exact arithmetic rather than interval overlap; the exploratory sections
are labelled as such; and where an adjusted and unadjusted reading disagree, the adjusted one is
now quoted as governing. What does not follow is that every unadjusted interval carries 95%
confidence in any family-wise sense.

**Null concentration is not an inference-validity test.** The former
`null_zero_fraction >= 0.90` veto was removed after a mixed-label positive control
showed it could suppress a clear policy-by-group association. The legacy concentration
field remains diagnostic only. Valid permutation inference depends on exchangeability
and the sampling, dependence, and selection design, not the fraction of zero replicates.
The EICU-AC result remains limited by the reported group-size and fitting sensitivities;
no held-out adaptive benefit is established.

## What is not claimed

Not a self-improving system. Host code selects among three pre-authored controls; no weights change
and no control is authored.

Not an isolated evaluator. An earlier version of this project included a separate evaluator that owns
scoring and never trusts a self-reported result. That is scoring independence, not containment: the
candidate ran as the same operating system user on the same machine. It is not part of this
repository.

Not novel machinery. Held-out evaluation, ablation against fixed baselines and outcome-derived
scoring are standard practice.

Not a result about adaptive safety in general. It is a negative result about one benchmark,
established on that benchmark.

**The same seeds are reused in every cell.** Draws use seeds 1000 to 1009 in every cell, arm and
vocabulary rather than fresh seeds throughout. If a runtime honours them, the draws are one fixed
seed panel replicated across cells rather than independent samples, which induces cross-cell
correlation that the cell-clustered intervals assume away. Disclosed rather than corrected:
re-running to vary seeds would change the frozen artefacts.
