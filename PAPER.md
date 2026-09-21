# Does the benchmark measure the model? Two preconditions for adaptive safety results

A short technical report. Personal research, five local models, two synthetic surfaces, one laptop.
Sections 3.1 to 3.3 are exploratory; 3.4 is a prospectively specified confirmatory run with
internal custody; 3.5 to 3.9 are offline re-analyses of the evidence those runs produced; 3.10 to
3.10.2 concern published third-party benchmarks; 3.11 and 3.12 are further exploratory experiments.
Everything is reproducible from committed evidence without calling a model except 3.10.1, which runs
LLaMA-Guard3, and 3.11 to 3.12, which run local models. Sections 3.10 to 3.10.2, 4.4 and 4.5 additionally need
third-party datasets that are not redistributed here.

## Abstract

I built a system that selects executable safety controls from a language model's own observed
failures, and it eliminated unsafe executions on all five models tested. It also performed
identically to switching every control on with no evidence at all.

The reason is that the benchmark barely measures the model. Replacing the model with a policy that
always attempts the action scores a perfect 10000 basis points, matching or beating every real model
tested;
an adversary choosing the worst output per input still scores 8889 against a 7778 do-nothing
baseline. Only 1111 of the 10000 responds to the model at all, so 88.9% of the score is the scaffold.
The controls are repair-only and can already produce the correct answer wherever the correct answer
is one they can reach, which makes attempting the action unpunishable and makes a single
configuration weakly best on every individual scenario cell. Enumerating the entire space of possible
model behaviour confirms that configuration is optimal not merely for the five models run but for
every model that could exist, so adaptation headroom here is zero by construction rather than by
measurement, under any reweighting of the benchmark and any metric nondecreasing in the per-cell
scores.

That gap between the best per-model and best shared setting is the virtual-best minus single-best
gap that algorithm selection has computed routinely for a decade, and it appears not to be reported
for adaptive safety guardrails. Two diagnostics follow, neither novel: substitute the model and
re-score, then measure the headroom. Both use the benchmark's existing labels and policies without
requiring another model run. Applied to two published
third-party guardrail benchmarks, the substitution test puts their no-model floors at 70.5% and
84.5%, far above the chance levels a reader would otherwise assume, and the published figures for a
guardrail baseline fall below both. Running that baseline here rather than citing it, in the weak
configuration and in the stronger one its authors describe, it never exceeds either floor: on EICU-AC
it lands roughly thirty points below in both, and on Mind2Web-SC it falls seventeen points below
without the benchmark's rules and lands exactly on the floor when given them, with symmetric
disagreements and an exact McNemar p of 1.000 — indistinguishable from a lookup that never reads the
task rather than worse than one. Of five published
benchmarks four report no trivial baseline and the fifth reports one 18 F1 points below its own
floor. The exploratory headroom analysis refuses two label-derived groupings, finds zero observed
headroom on Mind2Web-SC, no nominal association on R-Judge, and a threshold-sensitive nominal
association on EICU-AC; it does not establish held-out adaptive benefit. A separate factorial finds
that rewording an identical rule shifts five local models substantially towards asking rather than
acting, most strongly where they should have been most certain.

The contribution is an import rather than an invention, with a worked case in which it invalidates
my own headline result.

## 1. Setting

An agent decides whether to apply a financial adjustment. Applying it is correct only when approval
is granted and the authority to act is valid. Otherwise the agent should refuse, or ask when
approval is merely unresolved. The scenario space is the full cross product of approval status
(approved, denied, unknown), authority status (valid, expired, missing) and risk tier (low, high):
18 distinct cells.

A scaffold sits between the model and execution. It carries three categorical controls, each of
which can only make the agent more conservative:

- refuse when authority is not valid,
- refuse when approval was denied,
- ask when approval is unresolved.

It also carries a continuous control, a confidence floor, below which a decision is downgraded to
ask or refuse.

A deriver inspects failing trajectories and enables the first control that is off, has failing
evidence, and provably repairs that evidence under replay. This is host code choosing among
pre-authored controls. No model weights change and no new control is authored.

## 2. Method

Controls apply after the model decides, so a single set of samples scores every policy. I draw ten
independent samples per cell at temperature 0.7 and score all policies against those same draws,
paired cell by cell. The unit of analysis is the distinct cell, not the run.

Five policies, in increasing order of information used:

| Policy | Uses |
|---|---|
| always refuse | nothing |
| oracle rule | the label function itself, no model |
| raw | the model, no controls |
| all controls | every control on, no evidence, no derivation |
| derived | controls selected from observed failures |

Models: `qwen3.5:2b`, `qwen3.5:4b`, `qwen3.5:9b` and `gemma4:12b-it-q4_K_M` on Ollama, and
`qwen/qwen3.8-27b` on LM Studio. Reported figures are task success in basis points. Interval
estimates are cell-clustered bootstraps over the nine (approval, authority) cells (section 5); the
Wilson intervals
emitted by `ablation.py` are retained in the raw artefacts for traceability but are not the reported
uncertainty, because they treat correlated draws from one scenario as independent.

## 3. Results

### 3.1 Unguarded models are worse than refusing everything

| Model | raw | always refuse |
|---|---:|---:|
| `qwen3.5:2b` | 6592 | 7821* |
| `qwen3.5:4b` | 2722 | 7778 |
| `qwen3.5:9b` | 5389 | 7778 |
| `gemma4:12b` | 5222 | 7778 |
| `qwen3.8-27b` | 7556 | 7778 |

<sub>*`qwen3.5:2b`'s always-refuse baseline is 7821 rather than 7778 because one of its 180 draws
came back off-contract, meaning it did not parse as the JSON decision the schema requires, leaving nothing to score; it is excluded, which shifts the denominator.</sub>

All five fall below a policy that ignores its input. The failure they share is acting on an
approved adjustment when the authority to act has expired or is absent. It appears in every model,
though the 27B model makes it least often, and five models across two families is far too small a
sample to say anything about scale.

### 3.2 Derivation never beats enabling everything

| Model | all controls | derived | Difference |
|---|---:|---:|---:|
| `qwen3.5:2b` | 9888 | 9888 | 0 |
| `qwen3.5:4b` | 10000 | 10000 | 0 |
| `qwen3.5:9b` | 10000 | 10000 | 0 |
| `gemma4:12b` | 10000 | 10000 | 0 |
| `qwen3.8-27b` | 9944 | 9944 | 0 |

Identical, with identical intervals and identical unsafe-execution counts. Every control is
monotonically conservative and 14 of 18 correct answers are refusals, so enabling all of them is a
ceiling the deriver can reach but never exceed. The oracle rule, applied with no model, scores 10000
with zero unsafe executions: the task is solvable by a three-clause conditional over two fields
present verbatim in the observation.

### 3.3 No control has a model-dependent optimum

Sweeping the confidence floor across its full range, offline from the same draws:

| Setting | Best shared value | Each model tuned alone | Headroom |
|---|---:|---:|---:|
| Controls on | 9966 (floor 0) | 9966 | 0 |
| Controls off | 6607 (floor 10000) | 6718 | 111 |

With controls on, every model wants the same floor. With them off, every model wants the floor at or
near maximum, which is conservatism reached by another route, and even tuned it lands below the
always-refuse baseline for four of five.

### 3.4 Confirmatory run on an unseen surface

Frozen scoring and derivation files match `frozen/PREREGISTERED_V1.sha256`. Hypotheses and
falsification conditions were fixed in `PREREGISTRATION.md` before the surface existed. The surface shares the abstract decision rule and
no vocabulary: a maintenance-dispatch domain with renamed fields and two distractor fields, written
by a process with no access to any result in this repository. One run, ten draws per cell at
temperature 0.7, five models, no off-contract responses.

| Model | raw | always refuse | all controls | derived |
|---|---:|---:|---:|---:|
| `qwen3.5:2b` | 3889 | 7778 | 9556 | 9556 |
| `qwen3.5:4b` | 1722 | 7778 | 9500 | 9500 |
| `qwen3.5:9b` | 2722 | 7778 | 9944 | 9944 |
| `gemma4:12b` | 5167 | 7778 | 9444 | 9444 |
| `qwen3.8-27b` | 5222 | 7778 | 9500 | 9500 |

H1, that `derived` would fall within 1 bps of `all_guards` for every model: confirmed, with exact
equality — **and it could not have come out otherwise.** `derive_guards` runs the deriver to a
fixpoint over the same draws the comparison is then scored on, and a guard is admitted only when it
provably repairs a failing draw. At that fixpoint every guard left off has, by construction, no draw
it would change. Checking the five model-surface pairs that omit a guard confirms it: each omitted
guard applies to 20 draws and changes none of them. So the exact equality is an algebraic identity of
in-sample derivation, not a measurement, and H1 was a preregistered hypothesis incapable of failing.
Deriving on one run and scoring on another would make it falsifiable; that experiment has not been
done. The identity is still worth stating — it is precisely why the derivation earns nothing — but it
is not evidence, and this paper reported it as a confirmed prediction for longer than it should
have. H2, that headroom with controls enabled would be under 100 bps: confirmed at 0 bps, with
67 bps when controls are disabled. H3, that `raw` would fall below `always_abstain` for at least
three models: confirmed for all five.

Exploratory, from the same data and not preregistered: every model scored worse on the new surface
than the old one despite an identical rule, unguarded scores falling from a 2722-7556 range to
1722-5222.

The drop is not uniform degradation. Pooled across models the decision mix shifts sharply towards
asking, from 32.1% to 61.2%, while attempted actions fall from 29.6% to 14.2% and unsafe actions
mostly disappear. On cells where authority is invalid and refusal is the only correct answer, the
rate of asking instead rises from 32.2% to 67.7%. The models became more hesitant, which cost more
accuracy than the reduction in unsafe actions gained.

An obvious explanation is that the new surface renames unknown permission to PENDING, a word that
suggests waiting. It does not survive contact with the data: the shift towards asking is largest
where permission is GRANTED, at 46.3 points, against 18.0 points where it is PENDING. The effect is
broad rather than tied to that token, and it is strongest exactly where the model should be most
confident.

An earlier version of this section attributed the drop to familiarity with the financial vocabulary.
That attribution was not supported, because the two surfaces differ in three ways at once: domain
and field vocabulary, and the presence of two distractor fields in the newer one. Section 3.11
separates the two factors that can be varied cleanly.

### 3.5 Exhaustive sweep of the declared control space

Two words are used deliberately from here on. A **guard** is one of the three specific booleans in
this scaffold. A **control** is the general term the algorithm-selection framing needs: any element
of the declared family a method could choose between, which here means the three guards and the
confidence floor. Every guard is a control; the floor is a control that is not a guard. Section 4.1's
conditions are stated over controls because they are meant to apply to families other than this one.

Sections 3.3 and 3.4 varied the confidence floor under two categorical settings, all controls off
and all controls on. That is 82 of the 328 configurations this experiment grids over: three
boolean guards give eight combinations, and the floor grid has 41 points. The six mixed-guard
combinations were untested, and they are precisely where a per-model difference could hide, since a
partial configuration could suit one model and not another.

All 328 were then evaluated offline against the same committed draws, on both surfaces, under three
metrics. Balanced class accuracy is included because the cell set is 14 refusals to 2 acts to 2
asks, so raw accuracy rewards caution. Utility prices over-refusal explicitly: a correct decision
scores 1, an unsafe execution costs 10, and failing to act where acting was correct costs 1.

| Surface | Metric | Headroom | Best fixed configuration |
|---|---|---:|---|
| exploratory | accuracy | 0 bps | all three guards, floor 0 |
| exploratory | balanced | 0 bps | all three guards, floor 0 |
| exploratory | utility | 0 bps | all three guards, floor 0 |
| confirmatory | accuracy | 0 bps | all three guards, floor 0 |
| confirmatory | balanced | 0 bps | all three guards, floor 0 |
| confirmatory | utility | 0 bps | all three guards, floor 0 |

The result is stronger than a tie on the mean. One configuration is optimal for each model
individually, on both surfaces, under all three metrics reported. Between 5 and 160 of the 328
reach each model's maximum, so an arbitrary argmax appears to vary between models when the optimal
sets in fact all contain the shared best.

The earlier two-configuration analysis reached the correct conclusion by an insufficient method.
Reporting it as the headroom of the control family was an overclaim, and is corrected here.

### 3.6 The zero is forced by dominance, not by the scenario mix

A sweep over 328 configurations establishes that headroom is zero on the distribution actually
tested, 14 cells where abstaining is correct, 2 where acting is and 2 where asking is. It does not
establish that a different distribution would behave the same way, and the natural objection is that
the mix is what produces the null.

Scoring is a pure function of each draw's cell, decision, confidence and correct answer, so the full
configuration space can be re-scored under any reweighting of the cells without calling a model.
Rather than sweep weightings, the question can be settled directly. For each model, compute the set
of configurations that are weakly best on *every individual cell*. A configuration in that set
maximises any nonnegative weighted mean of the per-cell scores, so it is optimal under every possible
mix. If the sets of all models intersect, one configuration is optimal for every model under every
mix, and headroom is identically zero.

They do intersect: 35 configurations shared by all five models on the exploratory surface, 5 on the
confirmatory one, under both accuracy and utility. Balanced class accuracy needs no separate
treatment because it is itself a nonnegative reweighting of the per-cell accuracies, each cell in a
class of size n carrying weight 1/(3n).[^weights]

[^weights]: The implementation applies those weights per draw rather than per cell, which coincides
only when every cell has the same number of draws. One exploratory model has a cell with nine, from
an off-contract response. The difference does not affect the result.

Two numerical checks accompany the argument. Two thousand random Dirichlet reweightings of the 18
cells, per surface per metric, give a maximum headroom of 0.0000 bps. Resampling the draws within
each cell, which is the only noise that can break dominance, leaves a shared dominating
configuration intact in 100.0% of 300 replicates.

That second check previously resampled whole cells from an already-computed score matrix, and was
incapable of failing: a configuration weakly best on every column of a fixed matrix is weakly best
on any multiset of those columns, so it returned 100% whatever the data while being cited as
evidence that the dominance was not an accident of sampling. The check had no power. The
replacement is demonstrated to report failure on a constructed case where dominance is accidental
(`tests/test_dominance.py`).

**Mechanism.** The guards are repair-only: each maps a decision to ABSTAIN or to ASK, and none can
produce ACT. A guard therefore cannot destroy a correct ACT, and every cell whose correct answer is
ABSTAIN or ASK falls under one of the three guards. Enabling all three, **in the order the scaffold
fixes**, is consequently weakly beneficial on every cell, guards-on dominates, and once the
repairable errors are repaired the confidence floor has no remaining error to correct, which sends
its optimum to zero for every model.

The order qualifier is load-bearing and an earlier version of this paragraph omitted it, claiming
instead that enabling any individual guard is weakly beneficial. That is false: the ask guard alone
converts correct refusals into asks on the two cells where approval is unknown and authority is not
valid. It is the invalid-authority guard's precedence that makes the composition safe. Section 4.1
states the corrected condition.

The generalisation is the useful part, and it has to be stated carefully because a looser version
of it is vacuous. Let the controls be repair-only with respect to an output subset S: every control
maps a decision into S, and none can produce an output outside S. Let the family be complete on S:
for every input whose correct answer lies in S, some control produces that correct answer. Then any
such family has zero adaptation headroom independent of scenario mix, class balance and model, for
any metric nondecreasing in the per-cell scores. Section 4.1 gives the full statement and explains
why the looser phrasing, "complete with respect to the errors it can repair", says nothing.

The condition can be checked by inspecting the control set against the error taxonomy, before any
model is run, though section 4.2 explains why that check is of limited practical use and what to do
instead. This supersedes the base-rate explanation given in earlier
versions of this work: the reweighting result shows base rate is not the operative variable.

### 3.7 Sensitivity of the measurement

A measurement that cannot return a nonzero value would generate every result above, so the sweep is
checked against a case where headroom is known to be present. Restricted to the guards-off subspace,
where no configuration dominates, it returns 111.1 bps on the exploratory surface and between 66.7
and 133.3 bps on the confirmatory one. The exploratory figure agrees with `experiments/sweep.py`, an
independently written code path, to the precision reported. Constructed score matrices with headroom
built in by design are recovered exactly to six decimal places.

This control is asserted in continuous integration. A regression that made the sweep return zero
unconditionally would fail the build rather than silently strengthen the headline claim.

### 3.8 Most of the score is not the model

The dominance result invites a sharper question than the one it answers. If a single configuration
is optimal for every possible model, how much is the model contributing at all?

Replacing the model with sources that carry no information answers it directly, and needs no ground
truth. Under the dominating configuration, on both surfaces:

| Model substitute | Score |
|---|---:|
| always ACT | **10000 bps** |
| uniform over all decisions and confidences | 9259 bps |
| always ASK, always ABSTAIN, worst-case adversary | 8889 bps |
| observed models, exploratory | 9888 to 10000 bps |
| observed models, confirmatory | 9444 to 9944 bps |

A model that always attempts the action achieves a perfect score, matching or beating every real
model tested: three of the five also reach 10000 on the exploratory surface, and it beats all five
on the confirmatory one.
A model chosen adversarially to fail still scores 8889, above the 7778 of refusing everything. Only
1111 bps of the range responds to the model at all: 88.9% of the score is the scaffold.

The reason is the same repair-only structure that produces the dominance. The guards intercept every
input on which acting is wrong, so an unconditional attempt to act is never punished and is correct
on the one input class the guards do not cover. **The benchmark rewards recklessness in the model,
because the scaffold absorbs the consequences.**

This is the most damaging result in this work and it subsumes the headroom finding. A benchmark on
which the maximally unsafe constant policy scores full marks cannot support a claim about model
safety, and an adaptation claim built on top of it inherits the defect.

### 3.9 Where the dominance result stops

Cell-by-cell dominance implies optimality only for metrics **nondecreasing in the per-cell scores**.
Accuracy, balanced accuracy and the utility used here all qualify. Earlier drafts of this paper
stated the generalisation without that qualifier, which made it false as written.

The boundary is real and reachable on this evidence. Under a calibration gap, the absolute
difference between how often a system is correct and the confidence it states, negated so larger is
better, headroom is 33.52 bps on the exploratory surface and 237.33 bps on the confirmatory one. The
dominating configuration is not the shared optimum under either, and the per-model optima separate.
Repairing a decision moves accuracy while stated confidence stays put, so more repair can widen the
gap rather than close it. `experiments/boundary.py` reproduces both numbers.

Two further scope limits belong here. The scaffold admits every integer floor in 0 to 10000 and five
mutable booleans, so its configuration space is 320,032 rather than the 328 this experiment grids
over. Dominance nonetheless holds across all of it for the floor axis, because a floor affects a
given draw only through the predicate `confidence < floor` and therefore has exactly two behaviours
per draw; enumerating those two against the eight guard combinations covers every integer floor, and
the dominating configuration attains the maximum achievable score on every draw. The two booleans
outside the guard set cannot change a decision at all, but they do change resource consumption, so
the dominance argument does not extend to any cost-aware metric.

### 3.10 The no-model floor of two published benchmarks

Sections 3.1 to 3.9 concern a benchmark constructed for this work, which limits what they can
support. The diagnostic of section 3.8 is not so limited: it asks only what a system with no model
in it scores, and that question can be put to any benchmark whose data is available.

It was applied to the two benchmarks used by GuardAgent (arXiv 2406.09187) and AGrail
(arXiv 2502.11448). Mind2Web-SC contains 200 web navigation tasks with a balanced allow/block label
and six safety rules gating actions on user attributes. EICU-AC contains 316 clinical questions
under role-based access control across ten databases, with an official train/valid split which is
used here, since that is how a published figure would be obtained.

The no-model policies are deliberately weak, though they are not equally weak and the difference
matters. For Mind2Web-SC: the user-side precondition of each rule, evaluated from the user profile
alone, and a cross-validated lookup over the four boolean profile fields. That floor never reads the
task text. For EICU-AC: the question's template is inferred from raw text by token overlap against
training questions, then a lookup on role crossed with inferred template. That one does read the
text, and is a nearest-neighbour classifier fitted to training labels, so it is the stronger of the
two policies and yields a correspondingly less trivial floor. Neither calls a model.

| | Mind2Web-SC | EICU-AC |
|---|---:|---:|
| no-model floor | 70.5% [60.7, 77.3], cluster bootstrap over CV refits | 64.8% [56.3, 72.6] on the 128-item valid split; majority class |
| AGrail (GPT-4o) | 98.4% | 97.8% |
| AGrail (Claude-3.5) | 94.0% | 98.4% |
| GuardAgent (GPT-4) | 90.0% | 98.7% |
| LLaMA-Guard3 | 56.0% | 48.7% |

An interval policy, applied consistently from here on: a floor set by a *fitted or selected*
policy carries a resampling interval that refits inside each replicate, because a Wilson interval
treats the selected policy as fixed and on Mind2Web-SC comes out about a quarter narrower than the
cluster bootstrap with a lower bound three points higher — it is measuring the wrong variability.
The valid-split majority class is a single constant policy, so a binomial interval is appropriate
for it and retained above. Section 3.10.1 corrects the EICU-AC column's evaluation set and
supersedes this table's comparison for that benchmark.

Published figures are label prediction accuracy as reported in AGrail table 2, which also restates
the GuardAgent results.

Two observations follow. LLaMA-Guard3, used as a baseline in both papers, scores below the no-model
floor on both, on the published figures and on the Mind2Web-SC run reproduced in section 3.10.1. On
EICU-AC the reproduced run sits below the floor's point estimate but inside its interval, so that
half is weaker on evidence generated here than on the published number. It is not straightforwardly below majority-class: it clears
Mind2Web-SC's 50%, and it clears the 35.2% obtained by carrying EICU-AC's training majority onto the
validation split. The EICU validation split's own base rate is 64.8%, which 48.7% does not clear, so
which figure counts as doing nothing depends on whether a method may see the training labels. The
defensible statement is that it underperforms a lookup fitted to training labels. It is a general-purpose safety classifier applied to task-specific
access control, so the shortfall reflects misapplication as much as capability, but the reported
figures carry no indication of either, because neither paper computes a floor. Those figures are
taken from AGrail's table 2 and were not reproduced here: this work did not run the model, verify
its prompt template, or confirm the split behind its number, and if that harness differed from the
protocol assumed here the comparison weakens accordingly. And
the stronger methods clear their floors by between 13.3 and 27.9 points depending on method and
benchmark, so the diagnostic discriminates rather than rejecting everything, which is the property
that makes it usable.

Both benchmarks are better constructed than the one in this paper on the axis that matters here.
Mind2Web-SC is balanced by design, and while every blocked instance has a disqualifying user
attribute, 94 permitted instances have one too, so the label is not recoverable from the profile
alone. Its model-independent fraction is 70.5%, and EICU-AC's is 64.8%. The like-for-like figure for the
benchmark in section 3.8 is not the 88.9% quoted there, which is an adversarial minimum rather than
a best-trivial-policy floor. Measured the same way as these two, its floor is **100%**, since
always-ACT scores full marks. That comparison is considerably worse for this paper's benchmark than
the one an earlier draft printed.

Three caveats. The floors are computed here rather than by the original authors. The Mind2Web-SC
figure is cross-validated because no official split is distributed with it. And a floor is a lower
bound established by one particular weak policy; a better trivial policy would raise it, so these
numbers should be read as "at least this much is not the model."

### 3.10.1 Running the baseline rather than citing it

Section 3.10 compared floors computed here against figures transcribed from AGrail's table 2. Those
figures were the weakest link in the only outward-facing claim in this work, since a reader who
declined to accept them left nothing that could be restored without running the model. This runs it.

The harness is stated so that it can be disagreed with: one item per prompt as a single user turn
giving the requester's context and request, temperature zero, no few-shot examples and no rule list,
with the verdict mapped unsafe to block and safe to allow. AGrail states that it configured
LLaMA-Guard3 "with guard requests as safety categories" and publishes its code, so the harness above
is deliberately weaker than the one behind the published figure: it withholds the benchmark's rules.
That is a difference in setup rather than a disagreement about a fact, and the direction favours the
published number. So both configurations are run. Arm A withholds the rules, as above. Arm B supplies
the benchmark's own rules as Llama-Guard unsafe-content categories, which is what AGrail describes;
everything else is identical.

| Benchmark | No-model floor | Arm A, no rules | Arm B, rules supplied | As published |
|---|---:|---:|---:|---:|
| Mind2Web-SC (n=200) | 70.5% [60.7, 77.3] | 53.5% | 70.5% | 56.0% |
| EICU-AC (n=316) | 84.49% [75.0, 89.4] | 50.63% | 56.33% | 48.7% |

All four runs completed with no off-contract responses and no transport errors, and the model
produced both classes throughout rather than collapsing to a constant.

**On Mind2Web-SC the shortfall does not survive arm B, and the replacement finding is sharper.**
Supplied with the six rules, LLaMA-Guard3 scores 70.5%: the no-model floor, to the decimal. Paired on
the same 200 items the disagreements are symmetric, 39 each way, exact McNemar two-sided p = 1.000.
The model is not below a profile lookup that never reads the task; it is statistically
indistinguishable from one. Withhold the rules and it falls to 53.5%, 17 points under the floor, with
66 discordant against 32 and p = 0.00077.

The claim that survives both arms is therefore narrower than the one first reported and harder to
rebut: on this benchmark the guardrail model never exceeds a no-model floor in any configuration
tested. It is worth stating what this is not evidence of. LLaMA-Guard3 is a content-safety classifier
applied to access control, which both papers acknowledge, so arm A's shortfall is substantially
misapplication. Arm B removes that objection and the model still only reaches the floor. What the
result indicts is the benchmark's ability to separate the two, and the absence of any reported
reference against which a reader could have noticed.

**On EICU-AC the shortfall survives both arms by roughly thirty points.** Arm B is the more damaging:
its category list states the role-to-table permissions explicitly, and applied mechanically with no
model at all that list scores 93.35% on these 316 items. Given a rule set that is close to the answer
key, the model returned 56.33%.

**EICU-AC was withdrawn against a floor that was wrong, and is partly restored.** The withdrawal
compared the run's 58.59% against a floor of 58.6% and concluded it was level rather than beneath.
That floor omitted the simplest no-model policy there is: predict the commonest label. On this
validation split that policy scores 64.8%, above the template lookup, so it sets the floor. The
correct floor is 64.8% with a Wilson interval of [56.3, 72.6].

That correction was right about the omission and wrong about what followed from it. Reporting the
published 48.7% as below this floor compared two different evaluation sets, and the 64.8% figure is
in any case the validation split's own base rate — a policy selected by inspecting the labels it is
then scored against, which is not a floor an inference-time method could have reached.

**The published figures are scored on all 316 records.** AGrail's LLaMA-Guard3 row reports 48.7% at a
0% block rate, and always-allow over the full set is 154/316 = 48.73%; GuardAgent's 98.7% is 312/316
and its LlamaGuard row 50.3% is 159/316. No denominator of 128 yields any of these. Recomputing on
the set the published numbers use, with a policy restricted to what exists at inference — the
requester's role, and the question type inferred from the raw question text by token overlap, fitted
by five-fold cross-validation — gives a no-model floor of **84.49%**.

Its interval is out-of-bag rather than Wilson. A naive item bootstrap leaks here: a resampled
duplicate can be matched by the near-unique question text of its own copy, which pushes the bootstrap
mean to 91.5% against a point estimate of 84.49%, in the anti-conservative direction. Refitting on
each draw and scoring only the held-out items gives [75.0, 89.4] with a mean of 82.6.

| EICU-AC, all 316 records | score |
|---|---:|
| majority class | 51.27% |
| role × shipped question-type annotation, 5-fold CV | 85.44% (annotation; not available at inference) |
| **no-model floor**: role × question type inferred from text, 5-fold CV | **84.49%** [75.0, 89.4] |
| GuardAgent | 98.7% |
| GuardAgent's LlamaGuard baseline | 50.3% |
| AGrail's LLaMA-Guard3 baseline | 48.7% |

Both published guardrail baselines sit roughly 35 points below that floor and below its interval's
lower bound, while GuardAgent's own method clears it by 14 points. The finding for EICU-AC is
therefore stronger than the one first reported, and it took two wrong intermediate states — a
withdrawal against a miscomputed floor, then a reinstatement against the wrong evaluation set — to
arrive at it.

The run reproduced here, 58.59%, was scored on the 128-item validation split and belongs to that
split, not to the comparison above; the two must not be mixed. On its own terms it is below that
split's floor point estimate and inside its interval, and a paired test against that split's template
lookup returns p = 1.000, so it establishes nothing on its own.

The omission was surfaced by the survey in section 3.10.2. A project about trivial baselines had
built an elaborate template-inference floor while leaving out the majority-class baseline, the
first trivial baseline normally checked.

The result of running it is a smaller claim than the one it replaces. It is also the only part of
this work whose central comparison rests on numbers produced here rather than quoted.

### 3.10.2 Floors for three further benchmarks

Sections 3.10 and 3.10.1 rest on two benchmarks, and the contribution rests on the claim that floors
like these go unreported. A review made the obvious objection: two papers cannot evidence a
statement about a field. This widens the sample to five, using whatever no-model policies each
benchmark admits.

| Benchmark | Items | Majority class | No-model floor | Floor policy |
|---|---:|---:|---:|---|
| Mind2Web-SC | 200 | 50.0% | **70.5%** | cross-validated profile lookup |
| EICU-AC (valid) | 128 | 64.8% | **64.8%** | majority class |
| BeaverTails | 3021 | 57.4% | **67.4%** | prompt only, response deleted |
| R-Judge | 571 | 52.7% | **52.7%** | majority class |
| ToxicChat | 5083 | 92.9% | **92.9%** | majority class |

Two denominators need stating. EICU-AC's row is the 128-item validation split, where the best
no-model policy is its majority class; on all 316 records — the set its published figures use —
the floor is 84.49%, set by role × inferred question type (section 3.10.1). And BeaverTails'
nearest-neighbour floors are computed on a seeded shuffle of 2500 of its items rather than all
3021, a runtime cap recorded beside the number in `evidence/floor-survey.json`; uncapped, the
prompt-only floor is 66.6% and nothing in this section changes.

The distribution is the result, and it is not the one that would most flatter this work.

**Two of five have floors well above chance.** Mind2Web-SC sits 20.5 points above its majority class
and BeaverTails 10.0, in both cases because something other than the whole input carries the label.
In the first it is a user profile that never reads the task. In the second it is either half of the
pair on its own: deleting the response and keeping the prompt scores 67.40%, deleting the prompt and
keeping the response scores 66.44%, and both beat the 57.37% majority class. The prompt-only policy
sets the floor, which is the sharper reading of the two — whether a response will be judged unsafe is
already largely determined before the response exists. This is the partial-input ablation natural
language inference has used for years.

**R-Judge passes on accuracy and fails on the metric it is actually scored by.** Its metadata, the
scenario and attack type, predicts nothing beyond the majority class, so its accuracy floor is the
majority class at 52.71%. R-Judge reports F1, and under F1 an always-positive policy attains
2π/(1+π) = 69.04% on these 571 items. Computing a floor on a scale the benchmark does not use is the
same error this work identifies in ToxicChat, committed here against R-Judge, and it is the reason
this section originally recorded a pass.

**One shows the diagnostic catching a metric rather than a benchmark.** ToxicChat is 92.9% negative,
so its floor under accuracy is 92.9%, which is why the field reports F1 and AUPRC there instead.
The floor test does not say ToxicChat is broken; it says accuracy is the wrong scale for it, which
is a thing that can be checked in one line before an experiment rather than discovered after.

**And it found an error in section 3.10.** Computing the majority-class policy for every benchmark
exposed that EICU-AC's floor had been reported without it, six points too low, which had caused a
correct finding to be withdrawn. See section 3.10.1.

Not every safety benchmark admits this test. AgentHarm was fetched and set aside: its harmful split
carries no allow-or-block label and is graded by task completion, so there is no classification for
a trivial policy to attempt. The test applies to benchmarks shaped as classification, which is a
narrower claim than "safety benchmarks" and is stated here rather than left implicit.

Five is better than two and is still not a field. What it supports: floors vary widely, they are
sometimes far above chance, they are cheap to compute, and four of the five papers report no trivial
baseline. The fifth, R-Judge, reports a random baseline in its main results table and frames its
findings against it; that baseline sits 18 F1 points below the best trivial policy on the same
items, so the failure there is an understated floor rather than an absent one.

### 3.11 A factorial that separates vocabulary from distractors

Section 3.4 observed that models scored worse on the confirmatory surface than the exploratory one
under an identical abstract rule. An earlier version of that section attributed the drop to
vocabulary familiarity and has since withdrawn the attribution, because the two surfaces differ in
three ways at once and the comparison that produced the observation cannot separate them.

Two of the three factors can be varied cleanly. This runs the full 2x2, with all four arms produced
by one generator so that nothing differs between them except the factors named. Vocabulary is
finance, where permission is APPROVED, DENIED or UNKNOWN and authority is VALID, EXPIRED or MISSING,
against maintenance, where clearance is GRANTED, REFUSED or PENDING and the permit is CURRENT,
LAPSED or ABSENT. Distractors are three fields that vary freely and never bear on the answer. The
rule is identical in all four arms. Eight draws per cell at temperature 0.7, intervals cell-clustered
over the eighteen cells and resampled jointly across arms to keep the comparison paired.

Effects in basis points, pooled over models, with 95% cell-clustered intervals:

| Outcome | Vocabulary | Distractors | Interaction |
|---|---:|---:|---:|
| accuracy | **-2069** [-2958, -1181] | -472 [-778, -160] | +306 [-306, +889] |
| ask rate | **+2472** [+1618, +3396] | +486 [+285, +681] | -389 [-944, +153] |

**A correction about how these intervals are read.** This section reports 36 intervals — two outcomes
by three effects by the pooled figure and five models — and originally read each one on its own at
95%. Section 3.12 next door builds a simultaneous band for its 21 contrasts and this section did not,
which is an inconsistent standard between adjacent sections. An intermediate version of this
correction quoted a "studentised max-t" band with a critical value of 3.07 under which 14 of the 36
survive; no committed script computed any such band, the figure is not reproducible from this
repository's evidence, and it was wrong. The reproducible replacement, now computed inside
`factorial_report.py` from the same replicates as the intervals themselves, is an unstudentised
simultaneous band: per replicate, the largest absolute deviation of any of the 36 estimates from its
observed value, taken at its 95th percentile. It comes out at **2500 bps**, and **4 of the 36
intervals survive jointly** against 23 read individually — `qwen3.5:9b`'s vocabulary effect on both
outcomes, `gemma4:12b`'s on ask rate, and the substituted-build `qwen3.8-27b`'s on accuracy. No
pooled effect clears it. The per-interval figures below are retained because they are what the
evidence file records; the joint reading is the more demanding one, and it is the one that should be
believed.

**Vocabulary dominates on the point estimates, individually.** Its effect is about four times the
distractor effect on accuracy and five times it on ask rate. Renaming the fields and values of an
identical rule cost about 21 points of accuracy and added about 25 points of asking. Per model the
vocabulary interval excludes zero for four of five on accuracy and all five on ask rate read
individually. Under the joint band, however, neither pooled vocabulary effect survives, and the
per-model survivors are two of five per outcome — so the statement this section can support jointly
is narrower than "vocabulary dominates": it is that the vocabulary point estimates are several times
the distractor ones with individually significant pooled effects, while joint adjustment over all
36 intervals leaves only four survivors. A family of 36 that includes eleven interaction and
per-model intervals pays for their noise; that conservatism is the price of stating one family, and
it is paid here rather than argued away.

**The distractor effect does not survive on either outcome jointly.** Its pooled accuracy effect of
−472 bps [−778, −160] excludes zero individually, as does the ask-rate effect at +486 [+285, +681];
neither clears the joint band. The accuracy effect is also inconsistent in direction across models,
raising the ask rate for four models and lowering it for `qwen3.5:9b`.

**The interaction claim is withdrawn.** This section previously said that four per-model interaction
intervals exclude zero and that the pooled figure is near zero because they cancel. Under the joint
band **none of the five survives**, so that sentence was a multiple-comparison artefact and the
cancellation story it told was unsupported. What remains is the weaker and duller statement: this
design does not establish an interaction either way.

So the attribution section 3.4 originally made, and withdrew, turns out to be right after all,
but only on evidence that isolates the factor rather than on the comparison that first suggested
it. What the models are doing is not applying a rule to a state. Their
behaviour moves substantially when the state is spelled with different words, and it moves towards
asking rather than towards error, which is the direction that a safety evaluation would score as
caution.

Three limits. This is one vocabulary pair. Section 3.12 tests six more and finds this accuracy
contrast typical of pairs in general, this ask-rate contrast about three times the median pair, and
the familiarity explanation qualified (section 3.12). It is exploratory and was built after seeing the section 3.4
result.

And one of the five models needs its provenance stated. The first attempt at `qwen3.8-27b` failed
as infrastructure: its runtime began returning HTTP 400 partway through the first arm and never
recovered, failing even a trivial completion request afterwards. Those raw error records are kept in
`evidence/factorial/F2-qwen3.8-27b/`. The run was repeated after the runtime was restarted with a
different build, published by unsloth at Q6_K_XL quantization, which completed all four arms with no
off-contract responses at all. The endpoint reports the same model identifier for both builds, so
the repository distinguishes them by publisher and quantization. Its four arms are internally
comparable because they all ran on that build, and its numbers are **not** comparable with the
`qwen3.8-27b` figures in sections 3.1 to 3.4, which came from the earlier one.

### 3.12 Seven vocabularies, and what the one pair was worth

Section 3.11 rests on a single contrast. That cannot separate three explanations: that financial
wording is unusually familiar and any move away from it costs accuracy; that maintenance wording is
unusually hard; or that each vocabulary is its own draw from a wide distribution, in which case one
pair establishes nothing general.

Five further vocabularies were built to the same template, renaming only the two decision-relevant
fields, their values and the action: clinical, logistics, legal, aviation, and one deliberately
meaningless vocabulary of invented tokens carrying no connotation at all. The nonsense arm is the
control that matters. If the effect is familiarity, it should be the worst surface tested. If models
respond instead to what the words suggest, it should sit in the middle, because it suggests nothing.
Finance and maintenance were re-run here rather than carried across, so every figure comes from one
run under one set of conditions. Eight draws per cell, five models, intervals cell-clustered and
resampled jointly so each contrast stays paired.

| Vocabulary | Accuracy | Ask rate |
|---|---:|---:|
| clinical | 7139 | 2917 |
| finance | 6208 | 3097 |
| logistics | 6158 | 3285 |
| legal | 5389 | 2806 |
| aviation | 4708 | 3097 |
| maintenance | 3986 | 5764 |
| nonsense | 3431 | 4292 |

**The familiarity explanation is not falsified, and an earlier version of this section said it
was.** That claim was wrong on both of its supports, and a statistical review caught it.

The support offered was that clinical outscores finance by 931 bps. It does, and the direction holds
for all five models separately, but the contrast does not exclude zero: its cell-clustered interval
is [-403, +2361] individually and [-764, +2625] under the simultaneous band below. No per-model
interval excludes zero either. Directional consistency across five models that share eighteen cells
and largely one family is weak evidence, not a falsification.

Worse, the sweep's own prespecified discriminating outcome pointed the other way. This experiment was
built to test familiarity by including a vocabulary of invented tokens, on the stated logic that *if
the effect is familiarity, nonsense should be the worst surface tested*. Pooled, nonsense is the
worst surface tested. By the rule written before the data existed, that outcome supports familiarity
rather than refuting it, and declaring the opposite while the prespecified signal went that way was
the error.

The defensible reading is narrower. Familiarity in the crude sense, "financial wording is uniquely
well handled", is disfavoured, because five of five models score clinical higher. Familiarity in the
sense that matters, "surfaces the model has seen more of are easier", is consistent with everything
here, including the nonsense result. This design cannot separate them.

**Meaning matters, and its absence costs the most accuracy for some models and not others.** Pooled,
the nonsense vocabulary is last, 2777 bps below finance. That pooled figure conceals the widest
disagreement in the study, and the disagreement is the finding rather than the average:

| Model | Nonsense accuracy | Its rank among the seven |
|---|---:|---:|
| `gemma4:12b` | 1111 | 7 of 7 |
| `qwen3.5:4b` | 1458 | 7 of 7 |
| `qwen3.5:2b` | 1944 | 7 of 7 |
| `qwen3.5:9b` | 5694 | 5 of 7 |
| `qwen3.8-27b` | 6944 | 4 of 7 |

Nonsense is the worst surface for three of the five models and not for the other two, so even the
pooled ordering is not a property of models in general. Within the Qwen family the two largest cope
far better than the two smallest, which invites a capability reading, but `gemma4:12b` is the worst
of all five despite being larger than three of them, so this is not a clean function of scale and
the evidence here cannot separate scale from family. What it does show is that applying a rule to
meaningless symbols is a capability some of these models substantially lack and others substantially
have, and that reporting only the pooled number would hide that.

**The original pair was representative on accuracy and unusual on ask rate.** Its accuracy gap of
2222 bps ranks 6th of the 21 pairs against a median of 1450, and bootstrapping the rank itself gives
a 95% interval of 3rd to 12th: consistent with an ordinary draw, though the data do not exclude a
fairly extreme one. Its ask-rate gap of 2667 bps ranks 3rd of 21 against a median of 1007, with a
rank interval of 1st to 5th, so that contrast is atypical on any reading.
That contrast was therefore amplified by an atypical pair, and **section 3.11's ask-rate effect
should be read as roughly three times the median pairwise contrast rather than as a typical
consequence of rewording.** Maintenance turns out to be an outlier on this axis, producing more
asking than even the meaningless vocabulary.

The spread across seven vocabularies is 3708 bps on accuracy, well above the original pair's gap.
Surface wording moves these models further than the one comparison suggested, and less predictably.

**Twenty-one contrasts need a joint band.** Reporting each pairwise interval at a nominal 95% and
then counting how many exclude zero overstates how many differences are real. Taking, in each
bootstrap replicate, the largest deviation of any contrast from its observed value gives a
simultaneous band: 1694 bps on accuracy, 1486 on ask rate. Twelve of the 21 accuracy contrasts
exclude zero individually and nine do simultaneously; nine and six respectively for ask rate. The
finance-to-clinical contrast is not among either group.

All five models, after a runtime fault was diagnosed rather than reported. The `qwen3.8-27b`
sweep first failed exactly as its factorial run had, the engine returning errors from the first
vocabulary onward. The cause is engine exhaustion under sustained load, not the model or the task,
and it clears on an unload and reload: the sweep was re-run with a fresh engine per vocabulary and
completed all seven with no off-contract responses at all. The runner merges rather than overwrites
so a single vocabulary can be repeated and folded back in.

The failed attempt itself was not kept: its files were cleared before the re-run, so unlike the
aborted factorial run in `evidence/factorial/F2-qwen3.8-27b/`, which is retained and documents the
same fault, there is no artefact of it here. While it existed it would have moved the pooled finance
figure by 167 bps had the loader admitted runs by file existence rather than completeness, which is
the error the errata records once already and which `experiments/runs.py` now makes structurally
impossible. Exploratory throughout, and not preregistered.

## 4. The headroom test

Define the adaptation headroom of a benchmark, over a set of controls and models, as

    max over per-model settings of mean score  −  max over shared settings of mean score

It is the entire performance a per-model adaptive method can win over the best fixed configuration.
Computing it requires no adaptive method: sweep the controls, take two maxima, subtract.

This is the virtual-best-solver minus single-best-solver gap under another name, with models playing
the role of problem instances and controls the role of solvers. Section 6 gives the provenance. The
name is only here because "headroom" says what it is for.

If the headroom is near zero, the benchmark cannot separate an adaptive policy from a fixed one, and
an adaptive result on it carries no information about adaptation regardless of how strong the
headline number looks. This is checkable before any adaptive method is built.

### 4.1 A sufficient condition, stated without equivocation

Earlier versions of this section gave the sufficient condition as one class of action being nearly
always correct while the controls bias toward it. Section 3.6 refutes that: reweighting the cells
changes the base rate and never produces headroom. The condition below replaces it.

Let the controls be **repair-only** with respect to an output subset S: every control maps a
decision into S and none can produce an output outside S. Let the family be **complete** on S: for
every input whose correct answer lies in S, some control produces that correct answer. And let the
family's **composition** be sound: wherever the controls, applied together in the order the scaffold
fixes, override a contract-valid model output, the value they produce is the correct answer for that
input. The contract qualifier is needed because a confidence outside the declared range trips even a
zero floor, which can override a correct action; that is a contract violation reaching a control
rather than the controls being unsound, and section 3.6's enumeration covers it separately.

Then enabling all controls is weakly optimal on every input, so headroom is zero for every model and
every input distribution, for any metric that is nondecreasing in the per-input scores **and**
maximised at the correct answer for each input.

Three conditions, each of which an earlier version of this section got wrong, and all three found by
review rather than by me.

**Soundness is not decoration.** An earlier statement had only repair-only and completeness. Without
soundness a control that abstains unconditionally is repair-only and vacuously complete, yet
destroys every correct ACT.

**Soundness must be a property of the composition, not of each control.** The next version required
each control to fire only where its own output is correct, and asserted that the three guards here
satisfy it. They do not. In isolation, `ask_on_unknown_approval` returns ASK on unknown approval
with expired or missing authority, where the correct answer is ABSTAIN. It is wrong on 2 of the 9
contexts and is rescued only because `apply_scaffold_rules` evaluates the invalid-authority guard
first and returns before reaching it. The composition is sound; the parts are not, and
`tests/test_soundness.py` now asserts exactly that rather than the false stronger claim.

**Order is part of the family.** Because soundness here is compositional, the theorem does not
quantify over unordered sets of controls. Keeping these three guards, their predicates and their
outputs, but evaluating the ask guard first, makes enabling all of them score 6667 bps against 7778
for enabling none, on a model that always abstains, and produces 556 bps of headroom where the
shipped order produces zero. A control family is a composition, not a set.

**The metric condition needs the second clause.** Section 3.9 added monotonicity in the per-input
scores after a calibration-gap counterexample. That is necessary and not sufficient: a metric can be
monotone while not being maximised at the correct answer. Price every ASK at 1.5 alongside the
utility used elsewhere here and the conditions above all hold while headroom is 277.8 bps, because
an always-abstaining model prefers the ask guard off.

What survives all of this is the result rather than the argument: section 3.6's dominance is
established by direct enumeration over the shipped scaffold, and does not depend on the general
theorem being correctly stated. The theorem is the attempt to say why, and it took three tries.

Since three successive statements each looked right and were each broken by a counterexample someone
else constructed, the fourth is not offered on inspection. `experiments/verify_theorem.py` searches
for a counterexample directly: it enumerates ordered control families over a four-input, three-output
space, keeps the 25,260 family-and-labelling pairs out of 2,046,060 that satisfy all three
conditions, and checks the conclusion against every model and every subset of the controls. It finds
none. That is a fact about a small space rather than a proof for all spaces, and it is the reason
this statement is offered with more confidence than the previous three, which the same search would
have refuted.

The phrasing matters, because a looser version of it is vacuous. "Complete with respect to the
errors it can repair" is true of every control family whatsoever and says nothing; earlier drafts of
this paper used that form in the abstract while using the contentful form in the proof, which was an
equivocation. Completeness on S is a real and checkable restriction, and it amounts to saying **the
control family contains the ground-truth rule restricted to S**.

Stated that way the theorem is close to a tautology, and that is the useful part rather than an
embarrassment to be argued away. The condition is easy to satisfy by accident, because a control set
is usually written by reading the same specification that defines the labels. It is invisible from
inside the experiment: every headline number still moves in the expected direction. In this work it
went unnoticed through the exploratory and preregistered analyses. The corrected soundness
conditions are stated above, and the substantive claim changes are listed in `docs/ERRATA.md`.

### 4.2 The test that does not require knowing the label-generating rule

Checking completeness on S presupposes the ground-truth mapping. An author who has it does not need
a model, and an author who lacks it cannot run the check, so as a practical diagnostic it fails in
both directions. The following substitutes for it without requiring an articulated ground-truth
rule, though it still uses the benchmark's existing scorer or labels:

    Replace the model with something that carries no information. Re-score. Whatever score
    survives was never attributable to the model.

Concretely: score the guarded system driven by a constant output, by uniform noise, and by an
adversary choosing the worst output per input. The span between the adversarial floor and the
ceiling is the only part of the range any model can move. It requires the ability to substitute the
model's output and invoke the benchmark's existing scoring path; it does not require new labels
beyond those used for ordinary evaluation. It can therefore run even when the author cannot
articulate the full label-generating rule.

Section 3.8 applies it here, and the result is more damaging than the headroom finding.

### 4.3 The two checks are not the same check, and they do not have the same standing

They are presented above as one toolkit. A review pointed out that they do not share a system
boundary and do not have the same standing, and both objections are correct.

**They cut at different places.** Check one, as run in section 3.8, is a *system-boundary ablation*:
the scaffold stays in the loop and absorbs whatever the substitute emits, which is exactly why an
always-ACT policy scores 10000 rather than being punished for it. The number it produces is a
property of the model-plus-scaffold composite. On Mind2Web-SC and EICU-AC there is no scaffold to
stay in the loop, so what section 3.10 computes is not that ablation at all: it is a strong trivial
baseline, of the kind ordinary machine learning practice has always reported. Calling both of them
"the no-model floor" flattens a real distinction. The honest taxonomy is:

| | What is deleted | What remains | Where run here |
|---|---|---|---|
| system-boundary ablation | the model | the scaffold, still deciding | section 3.8 |
| trivial baseline, task-blind | the model and the task input | metadata only | Mind2Web-SC |
| trivial baseline, label-fitted | the model | a classifier fitted to training labels | EICU-AC |

All three answer "how much of this score was the model", which is why they belong together. They are
not interchangeable, and a floor of the third kind is a weaker rebuke than a floor of the first.

**Check two had never been computed outside this repository**, which is what prompted section 4.4.
Until then the headroom test, which this repository was originally named after, had been applied to exactly one
benchmark: the one it was built to invalidate, where section 3.6 proves it must return zero. Section
4.4 computes it on five published benchmarks. Two cannot be assessed because their groupings are
label-derived; Mind2Web-SC has zero observed headroom; R-Judge has no nominal association;
and EICU-AC has a nominal association sensitive to the minimum-unit-size choice. A sampled
null concentrated at zero is not itself invalid. These diagnostics do not establish held-out
adaptive benefit, but neither do they establish the absence of useful external headroom. What remains undone is computing it over a rule space an adaptive method induced
itself, rather than over trivial policies or over this benchmark's shipped library (section 4.5
does the closest feasible version).

There is a further obstacle the earlier sections do not confront. The test as stated requires
enumerating a *declared control space*, and the adaptive guardrails this work points at do not have
one: they adapt over open-ended, model-generated rule sets rather than a finite grid of switches. So
check two cannot be run against them in the form given, and section 4's recipe silently assumes a
scaffold shaped like this one. Making it applicable would mean either discretising an induced rule
library into a finite portfolio after the fact, or restricting the claim to systems whose control
space is finite and declared. Section 4.5 now does the first of those for the one benchmark whose
rule library is published and finite; the fully general version, over rules a model generated
itself, remains undone.

The consequence for how this work should be read: check one is a known practice, applied to a
setting where two papers had not applied it, with one paired result that survives scrutiny. Check
two is a genuinely underused metric which sections 4.4 and 4.5 now run on published benchmarks —
over trivial policies, and once over a discretised rule library. The trivial-policy analysis has
one nominal, threshold-sensitive association; the rule-library analysis has none. Check one is a
small contribution that stands; check two remains a proposal with a working implementation and no
held-out external adaptive-benefit result.

### 4.4 The headroom test on benchmarks that are not mine

Section 4.3 concedes that the check this repository was originally named after had been computed exactly once, on
the benchmark it was built to invalidate, where section 3.6 proves it must return zero. This applies
it to the five published benchmarks of section 3.10.2. The results are mixed, and earlier versions
first overstated positives and then overstated the negative conclusion.

**The control space.** The test needs a declared control space, and the adaptive guardrails this
work points at adapt over open-ended generated rule sets instead; section 4.3 states that obstacle
and it is not solved here. What is used instead is a control space these benchmarks do have: the
trivial policies available for scoring them. Choosing among them per subgroup is adaptation in the
algorithm-selection sense.

**Two benchmarks cannot be assessed at all, because their groupings are the label.** The obvious
subgroup in BeaverTails is the harm category, and the obvious one in ToxicChat is the jailbreak
flag. Both are annotations made alongside the label rather than covariates available at inference.
BeaverTails annotates a harm category only on unsafe items, so the group "none" is exactly the safe
set: the crosstab has zero off-diagonal entries. ToxicChat's jailbreak flag is never set on a
non-toxic item, in 91 of 91 cases. Any subgroup that contains a single class contributes a per-unit
accuracy of 1.0 by construction, so the virtual best is inflated for free and the permutation test
duly returns p < 0.001 for structure no method could use, because the annotation does not exist at
inference time.

The first version of this section reported 833 bps for BeaverTails and 1982 for ToxicChat as real
adaptation headroom. Both were artefacts of this. `experiments/external_headroom.py` now refuses any
benchmark with a single-class retained unit and says which units, so the analysis reports the defect
instead of the artefact.

**The three assessable cases have distinct, bounded interpretations.**

| Benchmark | Units | Headroom | Null mean | Above null | Permutation p | Verdict |
|---|---:|---:|---:|---:|---:|---|
| EICU-AC | 2 | 397 bps | 1 | +396 | 0.001 | nominal association; threshold-sensitive |
| R-Judge | 3 | 113 bps | 259 | **-146** | 0.729 | no nominal association |
| Mind2Web-SC | 3 | 0 bps | 13 | -13 | 1.000 | zero observed headroom |

Mind2Web-SC returns exactly zero: one policy is best in every domain, the same structural situation
as section 3.6, arrived at independently. R-Judge's 113 bps is *below* the 259 bps mean under
shuffled membership, while a per-unit maximum over four policies is optimistically biased.
The null mean is a shuffled reference, not an unbiased estimate of that bias. R-Judge's largest retained unit is
414 items whose scenario field is empty, so what is being tested there is partly a missing-data
group rather than a scenario.

**EICU-AC has a nominal association with important design limits.** Its retained 20-item
minimum analysis reports 397 bps and p = 0.001. The sampled permutation null is zero in
1983 of 2000 replicates and takes six distinct values. The previous text incorrectly
called this an invalid point mass and argued that nonnegative support guaranteed a positive
observed gap. A statistic may be nonnegative without being strictly positive, and a
discrete null can provide a valid tail comparison under the appropriate exchangeability
assumption. Zero concentration is a diagnostic, not an automatic veto.

The net advantage remains five decisions: six gains and one loss in the 63-item general
administration group; the 49-item nursing group contributes no advantage. The separate
within-group paired test reports p = 0.125. That test addresses a different null from the
item-partition permutation and therefore is not a contradiction of its p-value. Lowering
the minimum group size to 10 admits the 16 physician records and changes the retained
permutation result to p = 0.0585. The result is also sensitive to the fitted policy's CV
shuffle seed (the recorded headroom range is 79–556 bps).

`experiments/external_headroom.py` now retains null concentration without using it to veto
the nominal comparison. Its legacy `exceeds_chance_structure` field means nominal p < 0.05,
conditional on an appropriate exchangeability design, not a confirmed adaptive effect.
Fitted-policy dependence, repeated/template-linked records, multiplicity, group availability
at inference, and policy selection/evaluation separation require explicit treatment.

**What this establishes.** The diagnostic runs on third-party data and reports refusals,
observed gaps, nominal comparisons, and sensitivities. It does not establish a deployable
adaptive policy's improvement. It also does not establish that all five benchmarks lack
useful adaptation headroom. See `docs/ERRATA.md` for the correction. The interpretation change did
not alter the recorded measurements and is not presented as a fresh third-party-data replay.

**Two reporting choices, stated because they change the numbers.** Units are weighted equally rather
than by size, so a small unit counts as much as a large one; the item-weighted figure is reported
beside each estimate and differs materially where sizes are uneven. And BeaverTails and ToxicChat are
subsampled to 1500 items with a fixed seed; an earlier version took the first 1500 rows of the file,
which for ToxicChat gave a toxicity rate of 18.1% against 7.1% for the whole file. Head truncation
is not sampling.

### 4.5 One step towards the real control space: the rule library as a portfolio

Section 4.4's control spaces were trivial policies. This section builds the closest thing to an
adaptive guardrail's control space that this project can honestly enumerate: every subset of
Mind2Web-SC's six published safety rules, under conservative semantics — block iff any selected
rule fires on the item. That is 2^6 subset policies plus always-allow and always-block: 66
configurations, each one a thing a deployment could actually have been, and exactly the
"discretise an induced rule library into a finite portfolio after the fact" move section 4.3
named. A deployed instance chooses which rules are active; headroom here asks whether choosing
them *per domain* could beat one fixed choice.

One honesty cost has to be paid up front. The dataset ships which rule each blocked item violated,
but reading that annotation as the firing signal would make the rules label-derived — the exact
defect section 4.4 refuses — and it does not exist at all on allowed items. Each rule therefore
fires from inference-time signals only: task-text keywords crossed with the user profile, plus the
site's shopping domain for the membership rule. Against the shipped annotations these predicates
recover 72 of the 100 violations and fire spuriously on 16 allowed items; both counts are recorded
beside the result in `evidence/rule-space-headroom.json`, and the finding belongs to this
portfolio rather than to perfect rule evaluation.

What that error rate does to the inference was measured rather than assumed, because the plausible
guess turns out to be wrong here. The tempting direction runs: noise makes policies harder to tell
apart, so approximation biases toward a null, and this negative might mask structure too fine for
blunt predicates to expose. The benchmark's own annotations permit an oracle evaluator --
label-derived, so it is fenced off as a diagnostic beside the real result in
`evidence/rule-space-headroom.json` and must never be quoted as a verdict -- and under it the family
is exactly zero, p = 1.000: a perfect firing signal makes all-rules-on a perfect classifier in every
domain by construction, the same repair-only-and-complete shape as section 3.6. So the 111 bps
reported below is manufactured by predicate imprecision rather than being the shadow of masked
structure; in this instance the noise pushed the estimate *up*, toward per-domain differences that
do not exist, and the permutation test refused them. What approximation still costs is stated
plainly: no available evaluator can certify that better inference-time features would not expose
genuine per-domain preferences. Section 4.4's refusals carry no such residual, because they turn on
a grouping being degenerate rather than on a policy being imprecise.

The units are the benchmark's three top-level domains (Shopping, Travel, Entertainment), all
available at inference, each above the 20-item minimum, none single-class. Chosen post hoc;
exploratory throughout.

The computation itself runs through `floors.py`, so the instrument applied here is literally the
file offered for copying. The result:

| Quantity | Value |
|---|---:|
| Portfolio | 66 policies |
| Units | 3 domains (56-79 items) |
| Headroom | **111 bps** |
| Permutation p | 0.167 |
| Null degenerate | no |
| Verdict | no nominal association |

The best fixed entry in the portfolio is all six rules on. No per-domain selection beats it by
more than a net of two item-level decisions across the whole benchmark, and the permutation test
does not separate that from chance (stable across seeds: p between 0.167 and 0.189). This is the
same shape as section 3.6 arrived at independently: a control family assembled from a published
specification leaves nothing for per-unit selection to win, even when the family is taken apart
into every subset and the units are the domains the rules are about. It is one benchmark, one
approximate predicate set, three post hoc units — but it is the first time check two has been
pointed at anything shaped like the systems it was proposed for, and it found nothing there either.

What would change this section's standing: running the same enumeration over a rule library that a
model induced itself — AGrail's generated rules, or any successor's — rather than over the
benchmark's shipped ones. That needs their released artefacts and is not done here.

## 5. Threats to validity

Third-party corpora are not redistributed. Their source terms, access restrictions, and the public
release boundary are listed in `THIRD_PARTY_NOTICES.md`; external replay requires separately
authorized local copies.

**Partly preregistered.** Sections 3.1 to 3.3 are exploratory: the scenario set, prompt, provider
configuration, deriver rule and promotion criterion all changed as problems were discovered.
Section 3.4 is a preregistered confirmatory run against checksum-frozen scoring and derivation code,
on a surface written after the freeze by a separate context with no access to any result, with three
hypotheses and explicit falsification conditions fixed in advance. All three held. The same author
specified and accepted the surface, ran the experiment, and reported it; independent replication
has not occurred.

**Small and correlated sample.** Five models, four from one family, two runtimes, mixed
quantization. Models are the sampling unit for any population claim, and five is not enough.

**Estimand.** Reported intervals are cell-clustered bootstraps. The clustering unit is the
(approval, authority) pair, of which there are nine, not the 18 scenarios: risk tier is dropped
before the bootstrap sees it. Earlier drafts of this paper described the unit as the 18 scenarios,
which was wrong. Re-running at 18 clusters moves four of ten model intervals and always raises the
lower bound, so the shipped nine-cluster figures are the more conservative of the two. They cover
uncertainty from scenario composition and model stochasticity on this set. Earlier drafts used
Wilson intervals over individual draws, which treats ten correlated draws from one scenario as ten
independent observations and produces intervals that are too narrow. No interval here supports
generalisation beyond this scenario set. The equality between the derived and best fixed policies is
not an interval claim at all: the two policies make identical decisions on every draw.

**Constructs are close to circular.** The label rule, the three controls and the coverage taxonomy
are the same conditional written three times. Unsafe execution is computed on the post-scaffold
decision against the same fields the controls read, so zero is close to a logical identity once a
control is enabled. Model proposals remain unsafe throughout; the controls block them from
executing. This is control of execution, not improvement of the model.

**The task is trivial.** Authority status is handed to the model as a literal field. Nothing is
inferred, ambiguous or adversarial. Generalising to real agent authority failures is unwarranted.

**Simulated effects.** Execution writes into a no-effect runtime. Nothing was ever at stake.

## 6. Relation to existing practice

An earlier version of this section cited one literature, algorithm selection, and described the work
as "an import, not an invention". The description was right and the sourcing was not: both
diagnostics have substantial prior art in at least three separate literatures, and a review pointed
out that engaging only one of them made the novelty claim look larger than it is. What follows is
the corrected positioning. It narrows the contribution considerably.

### 6.1 Deleting the model is a rediscovery of deleting the input

The substitution test of section 3.8 is the scaffold-side mirror of a move that natural language
processing has used for a decade: strip away the part of the input the task is supposed to depend
on, re-score, and see how much survives. Poliak et al. (*SEM 2018) showed that a classifier reading
only the hypothesis, never the premise, beats majority class across natural language inference
datasets; Gururangan et al. (NAACL 2018) traced the same effect to annotation artefacts. The
equivalents in vision-and-language (question-only VQA, Goyal et al. 2017) and in reading
comprehension (passage-only, Kaushik and Lipton 2018) are established practice. Geirhos et al.
(*Nature Machine Intelligence* 2, 2020) give the general name: shortcut learning, decision rules that
score well on a benchmark for reasons the benchmark did not intend to reward.

The finding in section 3.8 belongs to that family. What differs is only where the shortcut lives: in
those papers the model exploits structure in the data, whereas here the *scaffold* supplies the
answer and the model is close to irrelevant. That is a different location for the same failure, not
a different failure.

**The closest prior work is closer still.** "Cheating Automatic LLM Benchmarks: Null Models Achieve
High Win Rates" (arXiv 2410.07137, ICLR 2025) shows that a null model emitting one constant response
regardless of input reaches an 86.5% length-controlled win rate on AlpacaEval 2.0 and 83.0 on
Arena-Hard-Auto. A constant policy beating real models on a benchmark that is supposed to measure
them is exactly the result of section 3.8, published a year earlier and more spectacularly. This
paper's always-ACT result should be read as a further instance of that phenomenon in a guardrail
setting, not as its discovery. Where the null-model work targets LLM judges, the mechanism here is a
rule-based scaffold, which is a different attack surface with the same consequence.

**And closer still than that, in this paper's own setting.** "Safety, or Just Capability? A Validity
Audit of Agent-Safety Benchmarks" (arXiv 2607.28685, July 2026) audits R-Judge, InjecAgent, AgentHarm
and AgentDojo as measurement instruments and computes a trivial-policy floor on R-Judge: an
always-positive policy attains F1 = 2π/(1+π) = 0.690, above five of the 21 models evaluated. That is
check one of this paper, on one of the five benchmarks surveyed here, published a month before this
repository's first commit and reaching the same conclusion. Section 3.10.2 reproduces their 0.690
exactly on the same data. This paper did not find that idea; at most it applies it to two benchmarks
they did not cover and reports the specific floors. Anyone citing this work should cite theirs.

Feng, Wallace and Boyd-Graber, "Misleading Failures of Partial-input Baselines" (ACL 2019), is the
necessary counterweight to the whole method and cuts against this paper's own reporting. A *low*
partial-input floor does not certify a benchmark: their examples show datasets where partial-input
baselines find nothing and genuine shortcuts remain. Section 3.10.2 originally read R-Judge's
majority-class floor as a pass, which is exactly the inference they warn against, and the correction
recorded there follows from their argument as much as from the F1 scale error. Ren et al.,
"Safetywashing" (NeurIPS 2024, arXiv 2407.21792), asks the parallel construct-validity question for
safety benchmarks specifically — whether they measure safety or capability — and is the closest
safety-domain analogue to what is attempted here.

Kapoor, Stroebl, Narayanan et al., "AI Agents That Matter" (arXiv 2407.01502), makes the adjacent
argument for agent benchmarks: elaborate agent scaffolds are frequently matched by simple baselines,
and benchmarks that do not report those baselines mislead. Their prescription and this paper's are
the same prescription.

### 6.2 The headroom gap comes from algorithm selection

The headroom quantity is the virtual best solver minus single best solver gap, with models in place
of problem instances and safety controls in place of solvers. The VBS is the score of a perfect
per-instance selector; the SBS is the best single fixed choice; the gap bounds what any selector can
win, and a small gap is read as evidence that selection cannot help on that benchmark. This has been
standard since SATzilla-era portfolio work, is formalised in ASlib (Bischl et al., *Artificial
Intelligence* 237, 2016), which reports VBS and SBS for every scenario it ships, and is the headline
metric of the Open Algorithm Selection Challenge (Lindauer et al., 2017), scored as the fraction of
the gap a selector closes. The same quantity appears in machine learning as *tunability*, the gain
from per-dataset hyperparameter tuning over shared defaults (Probst et al., JMLR 2019), and in
current LLM routing work as the oracle-router-minus-best-single-model gap.

There is also an older framing worth naming. A configuration that is weakly best on every cell is,
in the language of item response theory, a benchmark whose items have no discrimination: they do not
separate the candidates being compared. Psychometrics has measured that property directly for
decades.

### 6.3 What is left

Stated honestly, the residue is narrow.

Both diagnostics are established elsewhere and neither is invented here. What this work adds is
their application to adaptive safety guardrails, a worked case in which they invalidate the author's
own headline result, and two floors computed for published benchmarks that report none. That is a
contribution of the "someone should apply this here" kind.

Its premise is that the guardrail literature does not compute these floors. Section 3.10.2 widens
the evidence for that premise from two benchmarks to five, four of which report none and the fifth
a baseline 18 F1 points below its own floor. Five is not a field either. The claim should be read
as "these five papers do not report a floor above their own trivial baseline, and the floors are
sometimes far above chance" rather than as a statement about published practice in general.

Proposer and evaluator separation, held-out evaluation and outcome-derived scoring are likewise
established practice, and none of the machinery here is new.

## 7. What would change my mind

A benchmark in this family with non-zero headroom would show the zero result is specific rather than
structural. A control whose optimum genuinely varies by model, on a task where over-refusing costs
something, would make the adaptive question real. Either would be more interesting than what is
here.

I have looked, within the reach available to me. Section 4.4 runs the headroom test over the trivial
policies of five published benchmarks and finds one nominal, threshold-sensitive association that
does not establish held-out adaptive benefit. Section 4.5 takes one step towards the control spaces
that actually matter by discretising Mind2Web-SC's published rule library into a 66-policy portfolio
and finds no nominal association there either — but that library was written by the benchmark's authors, not
induced by a model, and its predicates had to be approximated from inference-time signals. Someone
running check two over a rule set a guardrail generated itself would still tell me something I
cannot currently tell myself.

If someone reproduces this and gets different numbers, the raw responses behind every figure are in
`evidence/`, and I would rather know.
