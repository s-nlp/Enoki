PYTHON?=python
PYTHONPATH?=./

# Paths
app_dirs := VeriScore/veriscore_run.py \
	VeriScore/veriscore_utils.py \
	factowl/factowl_utils.py \
	factowl/factowl_run.py \
	SAFE/safe_utils.py \
	SAFE/safe_run.py \
	Claimify/claimify_utils.py \
	Claimify/run_claimify.py \
	Claimify/metrics_calc.py


.PHONY: setup vendor run fmt lint

vendor:
	$(PYTHON) -m pip install -r requirements.txt

fmt:
	isort $(app_dirs)
	black $(app_dirs)

lint:
	@(isok=true; \
	echo "===== black ====="; \
	black --check $(app_dirs) || isok=false; \
	echo "===== flake8 ====="; \
	flake8 $(app_dirs) || isok=false; \
	$$isok && echo "\nLINTERS OK" || echo "\nLINTERS FAILED"; \
	$$isok;)
