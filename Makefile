CURRENT_FILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(shell dirname $(CURRENT_FILE_PATH))

ENV := $(ROOT_DIR)/.venv
ENV_CPU := $(ROOT_DIR)/.venv
PY := $(ENV)/bin/python
PY_VERSION := 3.12
PYTHONPATH := $(ROOT_DIR)


$(ENV):
	bash scripts/python-env.sh create

.PHONY: check-cuda
check-cuda:
	bash scripts/info.sh

.PHONY: venv
env: $(ENV)
	bash scripts/python-env.sh update

.PHONY: venv-cpu
env-cpu:
	bash scripts/python-env.sh create cpu
	bash scripts/python-env.sh update cpu

.PHONY: clean
clean:
	rm -rf *~ __pycache__

.PHONY: distclean
distclean: clean
	mamba env remove -p $(ENV)

.PHONY: docker
docker-%:
	$(eval command = $(@:docker-%=%))
	bash scripts/docker.sh $(command)

.PHONY: jupyter
jupyter:
	PYTHONPATH=$(ROOT_DIR):$(ROOT_DIR)/apps/gnn_explainer \
		$(PY) -m jupyter lab \
		--no-browser \
		--ip=0.0.0.0 --port=8080 \
		--autoreload \
		--ServerApp.base_url=/egr \
		--IdentityProvider.token='' \
		--ServerApp.allow_origin='*' \
		--ServerApp.allow_remote_access=True \
		--ServerApp.disable_check_xsrf=True

.PHONY: marimo
marimo:
	PYTHONPATH=$(ROOT_DIR):$(ROOT_DIR)/apps/gnn_explainer \
		$(PY) -m marimo lab \
		--no-browser \
		--ip=0.0.0.0 --port=8080 \
		--autoreload \
		--ServerApp.base_url=/egr \
		--IdentityProvider.token='' \
		--ServerApp.allow_origin='*' \
		--ServerApp.allow_remote_access=True \
		--ServerApp.disable_check_xsrf=True

.PHONY: voila
voila:
	PYTHONPATH=$(ROOT_DIR) $(PY) -m voila --no-browser \
		--port=8866 \
		--no-browser --autoreload=true

.PHONY: jupyter-clean
jupyter-clean: nb-clean

.PHONY: nb-clean notebook-clean
nb-clean notebook-clean:
	@bash scripts/notebooks.sh clear-output

.PHONY: streamlit
streamlit:
	$(PY) -m streamlit run apps/web/streamlit/app.py

.PHONY: gaston
gaston:
	bash cr/install-3rd-party.sh
