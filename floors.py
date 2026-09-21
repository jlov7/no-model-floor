"""Run both checks on your own benchmark. One file, no dependencies, no imports from this project.

The rest of this repository argues that two things should be measured before an adaptive safety
result is reported. A reviewer pointed out the obvious gap: every runnable target here operates on
the author's benchmark, so a reader convinced by the argument was left with prose. This is the
argument as something you can execute.

Copy this file. It imports nothing but the standard library and nothing from this project, so it
works wherever your benchmark lives.

    from floors import Benchmark, no_model_floor, adaptation_headroom

    bench = Benchmark(
        items=my_items,                              # any list of objects you like
        correct=lambda item: item.label,             # the right answer
        policies={                                   # things that do not use your model
            "always allow":  lambda items: [0] * len(items),
            "always block":  lambda items: [1] * len(items),
            "metadata only": my_lookup,              # fitted to training labels, if you like
        },
        unit=lambda item: item.category,             # optional: what adaptation would adapt to
    )

    print(no_model_floor(bench))
    print(adaptation_headroom(bench))

**Check one, `no_model_floor`.** Scores every policy that does not consult the system under test,
and reports the best. Whatever that reaches was never evidence about your model. Compare your
headline against it rather than against chance: the gap to the floor is what your model earned.
Include the majority-class policy. Omitting it is how the floor in this project's own section 3.10
came out six points too low and caused a correct finding to be withdrawn.

**Check two, `adaptation_headroom`.** Asks whether the best policy differs across your units. If one
policy is best everywhere, an adaptive method has nothing to win on this benchmark and a strong
adaptive result on it carries no information about adaptation.

The headroom number is a maximum minus a maximum, so it is never negative. That alone does not
justify reading anything from the gap: an interval for a nonnegative quantity can lie entirely
above zero, and a per-unit maximum manufactures headroom out of selection noise, so a positive gap
is the default even where no unit differs. A permutation p-value is reported instead, testing
whether your units carry more policy-relevant structure than a random partition of the same sizes.
Report the effect size and the p-value together. Inference requires an appropriate
exchangeability assumption; neither quantity establishes held-out adaptive benefit.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Benchmark", "FloorResult", "HeadroomResult", "adaptation_headroom", "no_model_floor"]

DEFAULT_SEED = 20260819
DEFAULT_ROUNDS = 2000
MIN_UNIT_SIZE = 20
# Conservative usability floor, not a mathematical significance boundary. Randomized permutation
# p-values lie on a 1/(rounds+1) grid; this tool requires 100 draws before printing its nominal
# binary verdict so very coarse grids are rejected. The default uses substantially more draws.
MIN_ROUNDS = 100


@dataclass
class Benchmark:
    """Your benchmark, described in the four terms these checks need.

    items:    whatever your examples are. Nothing is assumed about them.
    correct:  the right answer for one item.
    policies: name -> function taking all items and returning one prediction each. These must not
              consult the system you are evaluating. Fitting them to training labels is allowed and
              worth saying out loud when you report the floor.
    unit:     optional. What adaptation would adapt to: a category, a user type, a model, a domain.
              Required for check two, ignored by check one.
    train:    optional. Held-out items, in the same shape as `items`, used to fit the built-in
              majority-class baseline. Supply it whenever you have it.

              Without it the majority baseline is computed on the labels of `items` themselves,
              which is the conventional test-set majority and is *not* a score any method could have
              reached at inference: it requires reading the answer key. That distinction is not
              pedantic. This project reported a 64.8% "floor" for one benchmark which was that
              benchmark's evaluation-split base rate, then drew a published comparison against it;
              the honest figure fitted on training labels was 35.2%, and the conclusion did not
              survive the correction. `FloorResult.majority_fitted_on` records which happened, and
              printing a result whose floor is set by an unfitted majority says so out loud.
    """

    items: Sequence[Any]
    correct: Callable[[Any], Any]
    policies: dict[str, Callable[[Sequence[Any]], Sequence[Any]]]
    unit: Callable[[Any], Hashable] | None = None
    train: Sequence[Any] | None = None
    _predictions: dict[str, Sequence[Any]] = field(default_factory=dict, repr=False)

    def predictions(self) -> dict[str, Sequence[Any]]:
        if not self._predictions:
            pending: dict[str, Sequence[Any]] = {}
            for name, policy in self.policies.items():
                guesses = policy(self.items)
                try:
                    count = len(guesses)
                except TypeError:
                    # A generator gets through the annotation and then fails several frames away,
                    # with a message that does not say which policy produced it.
                    raise ValueError(
                        f"policy {name!r} returned {type(guesses).__name__}, which has no length. "
                        "Policies must return a sequence with one prediction per item; a generator "
                        "cannot be scored twice and this check needs to read it more than once."
                    ) from None
                if count != len(self.items):
                    raise ValueError(
                        f"policy {name!r} returned {count} predictions for {len(self.items)} items"
                    )
                pending[name] = guesses
            # Publish only after every policy validates. A failed call must not
            # make the next call silently score a truncated policy set.
            self._predictions = pending
        return self._predictions

    def truth(self) -> list[Any]:
        return [self.correct(item) for item in self.items]


@dataclass
class FloorResult:
    scores: dict[str, float]
    floor: float
    floor_policy: str
    majority_class: float
    majority_fitted_on: str = "evaluation labels"

    @property
    def floor_requires_the_answer_key(self) -> bool:
        """True when an unfitted majority sets the floor, so the floor is not reachable at inference."""
        return self.majority_fitted_on == "evaluation labels" and self.floor_policy.startswith(
            "majority class"
        )

    def __str__(self) -> str:
        lines = [
            f"no-model floor: {self.floor:.1%}  ({self.floor_policy})",
            f"majority class: {self.majority_class:.1%}  [fitted on {self.majority_fitted_on}]",
        ]
        for name, value in sorted(self.scores.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:32s} {value:6.1%}")
        lines.append("")
        if self.floor_requires_the_answer_key:
            lines.append(
                "WARNING: the floor here is the majority class of the labels it is scored on, which\n"
                "requires reading the answer key. No method could reach it at inference time. Pass\n"
                "Benchmark(train=...) to fit the baseline on held-out labels instead, and say which\n"
                "one you are quoting when you report this number."
            )
            lines.append("")
        lines.append("Compare your reported score against the floor, not against chance.")
        return "\n".join(lines)


@dataclass
class HeadroomResult:
    headroom: float
    permutation_p: float
    units: int
    best_shared_policy: str
    per_unit_best: dict[str, str]
    dropped_units: dict[str, int]
    null_zero_fraction: float = 0.0
    # Net, not literal: per unit, decisions the unit-best policy gets right that the shared best
    # does not, minus the reverse, summed over units. A gap reported as five can be six flips one
    # way and one the other; read it as the size of the imbalance, not a count of discordant items.
    items_driving_the_gap: int = 0

    @property
    def null_is_degenerate(self) -> bool:
        """Legacy name for a zero-concentration diagnostic, not a validity test.

        A discrete null can concentrate at zero and still assign a meaningful
        tail probability to a positive observation. This flag does not veto
        inference; exchangeability and the analysis design determine validity.
        """
        return self.null_zero_fraction >= 0.90

    @property
    def exceeds_chance(self) -> bool:
        return self.permutation_p < 0.05

    def __str__(self) -> str:
        verdict = (
            "nominal association; requires exchangeability"
            if self.exceeds_chance
            else "not distinguishable from chance"
        )
        lines = [
            f"adaptation headroom: {self.headroom * 10000:.0f} bps over {self.units} units",
            f"permutation p: {self.permutation_p:.6g}  ({verdict})",
            f"best single policy: {self.best_shared_policy}",
        ]
        differing = sorted(u for u, p in self.per_unit_best.items() if p != self.best_shared_policy)
        if differing:
            lines.append(f"units preferring something else: {', '.join(differing)}")
        if self.dropped_units:
            lines.append(f"units too small to use (<{MIN_UNIT_SIZE} items): {self.dropped_units}")
        lines.append(f"driven by: a net of {self.items_driving_the_gap} item-level decisions")
        lines.append("")
        if self.null_is_degenerate:
            lines.append(
                f"WARNING: {self.null_zero_fraction:.1%} of sampled null gaps equal zero.\n"
                "This concentration does not by itself invalidate the p-value. Check the\n"
                "exchangeability assumption, grouping, dependence, and policy-selection design."
            )
            lines.append("")
        lines.append("Report headroom and its nominal p-value; neither proves held-out adaptive benefit.")
        return "\n".join(lines)


def _accuracy(predictions: Sequence[Any], truth: Sequence[Any], indices: Sequence[int]) -> float:
    if not indices:
        return 0.0
    return sum(predictions[i] == truth[i] for i in indices) / len(indices)


def _require_items(bench: Benchmark) -> list[Any]:
    if len(bench.items) == 0:
        raise ValueError("Benchmark has no items; there is nothing to score")
    return bench.truth()


def _label_counts(labels: list[Any]) -> Counter:
    """Counter over labels, with a message that names the cause when they are unhashable."""
    try:
        return Counter(labels)
    except TypeError as exc:
        raise ValueError(
            f"Benchmark.correct produced an unhashable value ({exc}); labels must be hashable "
            "so they can be counted for the majority baseline"
        ) from None


def no_model_floor(bench: Benchmark) -> FloorResult:
    """Check one: the best score reachable without consulting the system under test."""
    truth = _require_items(bench)
    everything = list(range(len(truth)))
    scores = {
        name: _accuracy(guesses, truth, everything) for name, guesses in bench.predictions().items()
    }
    if bench.train is not None:
        if not bench.train:
            raise ValueError("Benchmark(train=...) was given an empty sequence")
        train_truth = [bench.correct(item) for item in bench.train]
        # The best constant a policy could choose knowing only training labels, scored honestly on
        # the evaluation labels. This can be far below the evaluation base rate when the splits
        # differ, which is exactly when quoting the evaluation base rate would mislead.
        constant = _label_counts(train_truth).most_common(1)[0][0]
        majority = sum(y == constant for y in truth) / len(truth)
        fitted_on = "training labels"
    else:
        majority = max(_label_counts(truth).values()) / len(truth)
        fitted_on = "evaluation labels"

    # The built-in baseline is added under a name no user policy holds. A single rename step was
    # not enough: a user supplying both "majority class" and "majority class (built in)" had the
    # second silently overwritten by the built-in rate, so the loop keeps going until the name is
    # genuinely free.
    name = "majority class"
    while name in scores:
        name += " (built in)"
    scores[name] = majority

    best = max(scores, key=lambda n: scores[n])
    return FloorResult(scores, scores[best], best, majority, fitted_on)


def adaptation_headroom(
    bench: Benchmark, *, rounds: int = DEFAULT_ROUNDS, seed: int = DEFAULT_SEED
) -> HeadroomResult:
    """Check two: whether the best policy differs across units, tested against a permutation null."""
    if bench.unit is None:
        raise ValueError("adaptation_headroom needs Benchmark(unit=...) to group by")
    if rounds < MIN_ROUNDS:
        # This is a conservative interface rule rather than a claim that p < 0.05 is impossible for
        # every smaller count. For example, 99 draws can attain 0.01, but only on a coarse grid.
        raise ValueError(
            f"rounds={rounds} is below this tool's conservative minimum of {MIN_ROUNDS}; "
            f"adjusted p-values lie on a 1/{rounds + 1} grid, which is too coarse for the "
            "nominal verdict this interface prints. "
            f"Use at least {MIN_ROUNDS}; "
            f"{DEFAULT_ROUNDS} is the default."
        )

    truth = _require_items(bench)
    predictions = bench.predictions()
    if not predictions:
        raise ValueError("adaptation_headroom needs at least one policy")
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, item in enumerate(bench.items):
        key = bench.unit(item)
        try:
            groups[key].append(index)
        except TypeError as exc:
            raise ValueError(
                f"Benchmark.unit returned an unhashable value ({exc}) at item {index}; "
                "units must be hashable so items can be grouped by them"
            ) from None

    kept = {u: idx for u, idx in groups.items() if len(idx) >= MIN_UNIT_SIZE}
    dropped = {str(u): len(idx) for u, idx in groups.items() if len(idx) < MIN_UNIT_SIZE}
    if len(kept) < 2:
        raise ValueError(
            f"need at least two units of {MIN_UNIT_SIZE}+ items; got {len(kept)}. "
            "Either group more coarsely or accept that this benchmark cannot answer check two."
        )

    # A unit containing one class scores 1.0 there by construction, which inflates the per-unit best
    # for free. The permutation null does not cancel it, because shuffling items across units breaks
    # the very purity that produced the inflation, so the p-value comes back significant for
    # structure no method could exploit at inference time. This project published two such results
    # before catching it; refusing here is what stops a copy of this file doing the same.
    try:
        single = sorted(str(u) for u, idx in kept.items() if len({truth[i] for i in idx}) == 1)
    except TypeError as exc:
        raise ValueError(
            f"Benchmark.correct produced an unhashable value ({exc}); labels must be hashable "
            "so units can be checked for single-class purity"
        ) from None
    if single:
        raise ValueError(
            f"units {single} each contain a single class, so per-unit accuracy there is 1.0 by "
            "construction and the headroom measured would be annotation redundancy rather than "
            "structure a method could use. This usually means the unit is derived from the label. "
            "Group by something available at inference time instead."
        )

    def score(name: str, indices: Sequence[int]) -> float:
        return _accuracy(predictions[name], truth, indices)

    per_unit_best_score = {u: max(score(n, idx) for n in predictions) for u, idx in kept.items()}
    shared = {n: sum(score(n, idx) for idx in kept.values()) / len(kept) for n in predictions}
    best_shared = max(shared, key=lambda n: shared[n])
    observed = sum(per_unit_best_score.values()) / len(kept) - shared[best_shared]

    sizes = [len(idx) for idx in kept.values()]
    flat = [i for idx in kept.values() for i in idx]
    rng = random.Random(seed)
    at_least = 0
    zero_replicates = 0
    for _ in range(rounds):
        shuffled = flat[:]
        rng.shuffle(shuffled)
        blocks, cursor = [], 0
        for size in sizes:
            blocks.append(shuffled[cursor : cursor + size])
            cursor += size
        virtual = sum(max(score(n, b) for n in predictions) for b in blocks) / len(blocks)
        single = max(sum(score(n, b) for b in blocks) / len(blocks) for n in predictions)
        at_least += (virtual - single) >= observed
        zero_replicates += abs(virtual - single) < 1e-12

    driving = 0
    for idx in kept.values():
        unit_best = max(predictions, key=lambda n: score(n, idx))
        driving += round((score(unit_best, idx) - score(best_shared, idx)) * len(idx))

    return HeadroomResult(
        null_zero_fraction=zero_replicates / rounds,
        items_driving_the_gap=driving,
        headroom=observed,
        permutation_p=(at_least + 1) / (rounds + 1),
        units=len(kept),
        best_shared_policy=best_shared,
        per_unit_best={
            str(u): max(predictions, key=lambda n: score(n, idx)) for u, idx in kept.items()
        },
        dropped_units=dropped,
    )


if __name__ == "__main__":
    # A worked example on a benchmark with a floor that is not chance, so the output is not
    # mistakable for a trivial one. Everything is synthetic and self-contained.
    rng = random.Random(0)
    items = []
    for _ in range(600):
        premium = rng.random() < 0.5
        # The label depends mostly on a metadata field, which is exactly what a floor detects.
        label = (
            1 if (premium and rng.random() < 0.85) or (not premium and rng.random() < 0.2) else 0
        )
        items.append({"premium": premium, "label": label, "region": rng.choice(["north", "south"])})

    def by_premium(rows):
        return [1 if r["premium"] else 0 for r in rows]

    bench = Benchmark(
        items=items,
        correct=lambda r: r["label"],
        policies={
            "always 0": lambda rows: [0] * len(rows),
            "always 1": lambda rows: [1] * len(rows),
            "metadata only": by_premium,
        },
        unit=lambda r: r["region"],
    )
    print(no_model_floor(bench))
    print()
    print(adaptation_headroom(bench))
