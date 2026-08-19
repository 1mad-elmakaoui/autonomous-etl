.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-spark data demo demo-fast demo-repair demo-optimize demo-pr test test-fast test-integration lint typecheck audit check clean docker-build docker-smoke k8s-validate k8s-apply observability-up observability-down temporal-up temporal-down worker submit

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev extras
	$(PY) -m pip install -e ".[dev]"

install-spark: ## Install dev *and* spark extras — what `make test` actually needs
	$(PY) -m pip install -e ".[dev,spark]"

data: ## Regenerate the example input data (seeded, reproducible)
	$(PY) examples/customer_pipeline/generate_data.py --rows 2000

demo: ## Full migration through to a measured validation verdict (needs a JVM)
	$(PY) -m etl_migrator.cli migrate examples/customer_pipeline/legacy_pipeline.py

demo-repair: ## Watch the repair loop fix a deliberately broken migration
	$(PY) -m etl_migrator.cli migrate examples/customer_pipeline/legacy_pipeline.py \
		--scenario customer_pipeline_broken --no-tests --no-optimize

demo-optimize: ## Benchmark a deliberately slow plan and keep the win only if it measures
	$(PY) -m etl_migrator.cli migrate examples/customer_pipeline/legacy_pipeline.py \
		--scenario customer_pipeline_slow --no-tests

demo-pr: ## Full migration through to a real pull request (needs a GitHub token)
	$(PY) -m etl_migrator.cli migrate examples/customer_pipeline/legacy_pipeline.py \
		--no-tests --no-optimize

demo-fast: ## Generation only, no execution (no JVM needed)
	$(PY) -m etl_migrator.cli migrate examples/customer_pipeline/legacy_pipeline.py --no-validate

test: ## Run the full suite, including Spark equivalence (needs a JVM)
	$(PY) -m pytest

test-integration: ## Run Temporal integration tests (needs `make temporal-up`)
	$(PY) -m pytest -m integration

test-fast: ## Run everything that needs no JVM and no running services
	$(PY) -m pytest -m "not spark and not integration"

lint: ## Lint
	$(PY) -m ruff check .

typecheck: ## Strict type check
	$(PY) -m mypy src

audit: ## Scan dependencies for known vulnerabilities (what CI's audit job runs)
	$(PY) -m pip_audit --strict --progress-spinner=off

docker-build: ## Build both images locally, exactly as CI does
	docker build --target cli -t etl-migrator:cli .
	docker build --target worker --build-arg INSTALL_SPARK=true -t etl-migrator:worker .

docker-smoke: docker-build ## Prove the built images actually start
	docker run --rm etl-migrator:cli --help
	docker run --rm --entrypoint java etl-migrator:worker -version
	@for image in etl-migrator:cli etl-migrator:worker; do \
		uid=$$(docker run --rm --entrypoint id $$image -u); \
		echo "$$image runs as uid $$uid"; \
		[ "$$uid" != "0" ] || { echo "FAIL: $$image runs as root"; exit 1; }; \
	done

observability-up: ## Prometheus + Grafana on localhost:9090 / :3000
	docker compose --profile observability up -d
	@echo "Grafana: http://localhost:3000 (dashboard provisioned, no login)"

observability-down: ## Stop the observability stack
	docker compose --profile observability down

k8s-validate: ## Schema-validate the Kubernetes manifests and check their policies
	$(PY) -m pytest tests/test_kubernetes.py -q

k8s-apply: ## Apply the manifests to the current kubectl context
	kubectl apply -k k8s/base

check: lint typecheck audit test ## Everything CI runs

temporal-up: ## Start Temporal + Postgres + UI (http://localhost:8080)
	docker compose up -d

temporal-down: ## Stop the local Temporal stack
	docker compose down

worker: ## Run a Temporal worker against the local stack
	$(PY) -m etl_migrator.cli worker

submit: ## Submit the example migration to Temporal
	$(PY) -m etl_migrator.cli submit examples/customer_pipeline/legacy_pipeline.py

clean: ## Remove caches and migration workspaces
	rm -rf .workspace .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
