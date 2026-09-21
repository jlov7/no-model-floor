# Everything you need. Requires Python 3.13+.
#
#   make test      the test suite
#   make probe     what each model does with no scaffold at all
#   make ablate    the decisive comparison: derived guards vs all guards on
#   make sweep     the headroom test, offline from ablation output
#   make headroom  the headline result in one file, no model needed
#   make exhaustive  all 328 configurations, three metrics, both surfaces
#
# MODEL and BASE_URL default to a small local model on Ollama. Override for anything else.

PYTHON   ?= python3.13
MODEL    ?= qwen3.5:2b
BASE_URL ?= http://127.0.0.1:11434/v1
SAMPLES  ?= 10

export PYTHONPATH := src:experiments

.PHONY: ci hooks test verify verify-external probe ablate sweep confirm confirm-sweep exhaustive dominance theorem model-ablation boundary external survey external-headroom rule-space reproduce-guard verify-theorem vocabulary vocabulary-report factorial uncertainty figures headroom all clean

ci:
	PYTHON=$(PYTHON) ./scripts/ci.sh

hooks:
	@ln -sf ../../scripts/pre-push .git/hooks/pre-push
	@echo "pre-push hook installed; scripts/ci.sh will run before each push"

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

probe:
	@mkdir -p out
	$(PYTHON) experiments/probe.py --base-url $(BASE_URL) --model $(MODEL) \
	  --response-format json_schema --reasoning-effort none --max-output-tokens 1200 \
	  --output out/probe-$(subst /,-,$(subst :,-,$(MODEL))).json

ablate:
	$(PYTHON) experiments/ablation.py --base-url $(BASE_URL) --model $(MODEL) \
	  --samples $(SAMPLES) --temperature 0.7 --outdir out/ablation/$(subst /,-,$(subst :,-,$(MODEL)))

# Recompute every published headroom figure from committed evidence. No model needed.
verify:
	@mkdir -p out
	$(PYTHON) headroom.py
	$(PYTHON) experiments/exhaustive.py --exploratory evidence/ablation \
	  --confirmatory evidence/confirmatory --output out/verify-exhaustive.json
	$(PYTHON) experiments/sweep.py --ablation-root evidence/ablation --output out/verify-sweep.json
	$(PYTHON) experiments/confirm_sweep.py --root evidence/confirmatory --output out/verify-confirm-sweep.json

# Requires third-party datasets that are not redistributed in this repository. Missing data fails.
verify-external:
	PYTHON=$(PYTHON) ./scripts/verify-external.sh

sweep:
	@mkdir -p out
	$(PYTHON) experiments/sweep.py --ablation-root out/ablation --output out/sweep.json

# The preregistered run, against a surface the finding was never developed on.
confirm:
	$(PYTHON) experiments/confirm.py --base-url $(BASE_URL) --model $(MODEL) \
	  --samples $(SAMPLES) --temperature 0.7 \
	  --outdir out/confirmatory/$(subst /,-,$(subst :,-,$(MODEL)))

confirm-sweep:
	@mkdir -p out
	$(PYTHON) experiments/confirm_sweep.py --root out/confirmatory --output out/confirm-sweep.json

headroom:
	$(PYTHON) headroom.py

factorial:
	@mkdir -p out/factorial
	@echo "Runs models. MODEL and BASE_URL required, e.g."
	@echo "  make factorial MODEL=qwen3.5:2b BASE_URL=http://127.0.0.1:11434/v1 OUT=F3-qwen3.5-2b"
	$(PYTHON) experiments/factorial.py --base-url $(BASE_URL) --model $(MODEL) \
	  --samples 8 --outdir out/factorial/$(OUT)

factorial-report:
	$(PYTHON) experiments/factorial_report.py --root evidence/factorial \
	  --output evidence/factorial-effects.json

reproduce-guard:
	$(PYTHON) experiments/reproduce_guard.py --base-url $(BASE_URL) \
	  --mind2web external/m2w/seeact/sample_labeled_all.json \
	  --eicu external/eicu/ehragent/eicu_ac.json \
	  --outdir evidence/guard-reproduction

vocabulary:
	@mkdir -p evidence/vocabulary
	$(PYTHON) experiments/vocabulary_sweep.py --base-url $(BASE_URL) --model $(MODEL) \
	  --samples 8 --outdir evidence/vocabulary/$(OUT)

vocabulary-report:
	$(PYTHON) experiments/vocabulary_report.py --root evidence/vocabulary \
	  --output evidence/vocabulary-effects.json

verify-theorem:
	$(PYTHON) experiments/verify_theorem.py --max-controls 3 --output evidence/theorem-search.json

external-headroom:
	$(PYTHON) experiments/external_headroom.py --root external \
	  --output evidence/external-headroom.json

rule-space:
	$(PYTHON) experiments/rule_space_headroom.py \
	  --mind2web external/m2w/seeact/sample_labeled_all.json \
	  --output evidence/rule-space-headroom.json

survey:
	$(PYTHON) experiments/floor_survey.py --root external/survey \
	  --output evidence/floor-survey.json

external:
	$(PYTHON) experiments/external_floor.py \
	  --mind2web external/m2w/seeact/sample_labeled_all.json \
	  --eicu external/eicu/ehragent/eicu_ac.json \
	  --output evidence/external-floor.json

boundary:
	@mkdir -p out
	$(PYTHON) experiments/boundary.py --exploratory evidence/ablation \
	  --confirmatory evidence/confirmatory --output out/boundary.json

model-ablation:
	@mkdir -p out
	$(PYTHON) experiments/model_ablation.py --exploratory evidence/ablation \
	  --confirmatory evidence/confirmatory --output out/model-ablation.json

theorem:
	@mkdir -p out
	$(PYTHON) experiments/theorem.py --output out/theorem.json

dominance:
	@mkdir -p out
	$(PYTHON) experiments/dominance.py --exploratory evidence/ablation \
	  --confirmatory evidence/confirmatory --output out/dominance.json

uncertainty:
	@mkdir -p out
	$(PYTHON) experiments/uncertainty.py --exhaustive evidence/exhaustive.json \
	  --exploratory evidence/ablation --confirmatory evidence/confirmatory \
	  --output out/uncertainty.json

figures:
	PYTHONPATH=docs $(PYTHON) -c "import make_figure; make_figure.main(); make_figure.headroom_figure(); make_figure.model_ablation_figure()"

exhaustive:
	@mkdir -p out
	$(PYTHON) experiments/exhaustive.py --exploratory evidence/ablation \
	  --confirmatory evidence/confirmatory --output out/exhaustive.json

all: test ablate sweep

clean:
	rm -rf out .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
