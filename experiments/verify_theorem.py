"""Try to break the sufficient condition in PAPER section 4.1, by exhaustion over small families.

That condition has been stated wrongly three times. Each version looked right, survived a round of
review, and was then broken by a counterexample someone else constructed. Stating it a fourth time
and hoping is not a method.

So this searches for a counterexample instead. It enumerates small control families over a small
input space, keeps only those satisfying all three conditions, and checks the conclusion against
every model and every subset of the controls. If the conditions are sufficient, nothing is found. If
a fourth error is hiding in the statement, this is what finds it rather than a reader.

The conditions under test, from section 4.1:

  repair-only   every control maps a decision into S; none produces an output outside S. Note
                that `controls_over` only ever generates controls whose output is in S, so this
                condition holds by construction and the filter for it never rejects a family. The
                search therefore exercises the other two conditions; repair-only is enforced by the
                generator rather than tested by it.
  complete      for every input whose correct answer lies in S, some control produces it.
  sound         wherever the ordered composition overrides the model, its output is correct.

The conclusion under test: enabling all controls is weakly optimal among all subsets, for every
model, under any metric nondecreasing in per-input scores and maximised at the correct answer. The
0/1 correctness metric is the extreme case and is what is checked; a counterexample there is a
counterexample for the family, and its absence over this space is what the claim rests on.

Deliberately small and complete rather than large and sampled: exhaustive over 4 inputs, 3 outputs
and families of up to 3 controls means the absence of a counterexample is a fact about that space
rather than a probability.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

OUTPUTS = ("ACT", "ASK", "ABSTAIN")
S = ("ASK", "ABSTAIN")  # the reachable subset; ACT is the output no control can produce
INPUTS = (0, 1, 2, 3)


def controls_over(inputs):
    """Every control: a firing predicate over inputs, and an output drawn from S."""
    out = []
    for size in range(1, len(inputs) + 1):
        for fires_on in itertools.combinations(inputs, size):
            for value in S:
                out.append((frozenset(fires_on), value))
    return out


def compose(family, model_output, input_):
    """Apply an ordered family: first control whose predicate holds wins, else the model stands."""
    for fires_on, value in family:
        if input_ in fires_on:
            return value
    return model_output


def is_repair_only(family):
    return all(value in S for _, value in family)


def is_complete(family, correct):
    for input_ in INPUTS:
        if correct[input_] in S and not any(
            input_ in fires_on and value == correct[input_] for fires_on, value in family
        ):
            return False
    return True


def composition_is_sound(family, correct):
    """Wherever the composition overrides, its output must be the correct answer."""
    for input_ in INPUTS:
        for model_output in OUTPUTS:
            got = compose(family, model_output, input_)
            if got != model_output and got != correct[input_]:
                return False
    return True


def score(family, model, correct):
    return sum(compose(family, model[i], i) == correct[i] for i in INPUTS)


def search(max_controls: int) -> dict[str, Any]:
    every_control = controls_over(INPUTS)
    assignments = list(itertools.product(OUTPUTS, repeat=len(INPUTS)))
    models = list(itertools.product(OUTPUTS, repeat=len(INPUTS)))

    families_tested = qualifying = 0
    counterexamples = []

    for size in range(1, max_controls + 1):
        for family in itertools.permutations(every_control, size):
            if not is_repair_only(family):
                continue
            for correct in assignments:
                families_tested += 1
                if not is_complete(family, correct):
                    continue
                if not composition_is_sound(family, correct):
                    continue
                qualifying += 1
                # The conclusion: no subset of the controls beats the whole family, for any model.
                for model in models:
                    full = score(family, model, correct)
                    for take in range(size):
                        for subset in itertools.combinations(family, take):
                            if score(subset, model, correct) > full:
                                counterexamples.append(
                                    {
                                        "family": [[sorted(f), v] for f, v in family],
                                        "correct": list(correct),
                                        "model": list(model),
                                        "subset": [[sorted(f), v] for f, v in subset],
                                        "full_score": full,
                                        "subset_score": score(subset, model, correct),
                                    }
                                )
                                break
                        if counterexamples:
                            break
                    if counterexamples:
                        break
                if counterexamples:
                    break
            if counterexamples:
                break
        if counterexamples:
            break

    return {
        "inputs": len(INPUTS),
        "outputs": list(OUTPUTS),
        "reachable_subset": list(S),
        "max_controls_per_family": max_controls,
        "family_correctness_pairs_examined": families_tested,
        "pairs_satisfying_all_three_conditions": qualifying,
        "counterexamples_found": len(counterexamples),
        "counterexample": counterexamples[0] if counterexamples else None,
        "conditions_sufficient_over_this_space": not counterexamples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-controls", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = search(args.max_controls)
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"examined {report['family_correctness_pairs_examined']:,} ordered family/labelling pairs"
    )
    print(f"{report['pairs_satisfying_all_three_conditions']:,} satisfied all three conditions")
    if report["counterexamples_found"]:
        print("\nCOUNTEREXAMPLE FOUND. The condition in section 4.1 is still insufficient:")
        print(json.dumps(report["counterexample"], indent=2))
        return 1
    print("\nNo counterexample. Enabling every control was weakly optimal in all of them.")
    print("That is a fact about this space, not a proof for all spaces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
