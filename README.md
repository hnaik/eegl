# Explanation Enhanced Graph Learning (EEGL)

EEGL is an iterative framework that enhances Graph Neural Networks (GNNs) by mining frequent subgraphs from GNN explanations and feeding them back as additional node features.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- `gcc`, `make`, `wget`, and `patch` to build third-party tooling such as [Gaston](https://liacs.leidenuniv.nl/~nijssensgr/gaston/)
- A CUDA-capable GPU for GPU-backed experiments (optional, but recommended)
- [Podman](https://podman.io/) or Docker for container-based workflows

The repository also includes prebuilt Glasgow Subgraph Solver binaries under `external/` for supported platforms.

## Setup

### Local development

Create or refresh the default environment:

```sh
make env
```

This runs `uv sync` and creates `.venv/` if needed.

Install the Gaston frequent subgraph miner:

```sh
make gaston
```

If you only need runtime dependencies, install the non-development set with:

```sh
make deps
```

### Dev container (VS Code)

Open the repository in VS Code and select **Reopen in Container** when prompted. The dev container builds from `.devcontainer/Dockerfile` and runs `uv sync` automatically.

### Manual container workflow

```sh
make docker-build
make docker-run
make docker-login
```

The container helper uses Podman when available and falls back to Docker otherwise.

## Quick start

Run the bundled demo workflow:

```sh
make run-demo
```

This launches [workflows/run.py](workflows/run.py) with [run_configs/dev.yml](run_configs/dev.yml) and [run_configs/run_defaults.yml](run_configs/run_defaults.yml).

To preprocess fullerene data into pickled graphs:

```sh
make process-fullerenes
```

The default output location is `dataset/pickled/fullerenes`.

## Interactive tools

- `make jupyter` starts JupyterLab on port `8080`
- `make marimo` starts a Marimo lab server on port `8080`
- `make voila` starts Voilà on port `8866`
- `make nb-clean` clears output cells from changed notebooks

`make jupyter-clean` and `make notebook-clean` are aliases for `make nb-clean`.

## Environment variables

| Variable | Description |
|---|---|
| `EEGL_SOLVER_PATH` | Override the path to the Glasgow subgraph solver binary |
| `GASTON_BIN_PATH` | Override the path to the Gaston executable |
| `PYTHONPATH` | Should include the repository root for direct module execution |
| `LD_LIBRARY_PATH` | Used by the Glasgow solver to find libraries in `.deps/lib` |

## Makefile targets

| Target | Description |
|---|---|
| `check-cuda` | Print Torch and CUDA environment information |
| `env` | Create or refresh the default `uv` environment |
| `env-cpu` | Create or refresh the CPU-oriented environment variant |
| `clean` | Remove editor backups and local `__pycache__` directories |
| `distclean` | Run `clean` and remove `.venv/` |
| `docker-build` | Build the development container image |
| `docker-run` | Build and start the development container |
| `docker-login` | Open a shell in the running container |
| `jupyter` | Start JupyterLab on port `8080` |
| `marimo` | Start Marimo lab on port `8080` |
| `voila` | Start Voilà on port `8866` |
| `nb-clean` | Clear output from changed notebooks |
| `notebook-clean` | Alias for `nb-clean` |
| `jupyter-clean` | Alias for `nb-clean` |
| `streamlit` | Run the configured Streamlit app entrypoint (`apps/web/streamlit/app.py`) |
| `gaston` | Download, build, and install Gaston |
| `deps` | Install runtime dependencies with `uv sync --no-dev` |
| `process-fullerenes` | Build pickled fullerene graph datasets |
| `run-demo` | Run the demo workflow configuration |

## Project layout

- [egr](egr) contains the main EEGL library code
- [apps](apps) contains utility entrypoints for data conversion and feature generation
- [run_configs](run_configs) contains workflow configuration files
- [workflows](workflows) contains the orchestration code used for experiments
- [tests](tests) contains the automated test suite
- [nb](nb) contains notebook-related files and helpers

## Citation

For citations please use:

```latex
@misc{naik2024iterativegraphneuralnetwork,
      title={Iterative Graph Neural Network Enhancement via Frequent Subgraph Mining of Explanations}, 
      author={Harish G. Naik and Jan Polster and Raj Shekhar and Tamás Horváth and György Turán},
      year={2024},
      eprint={2403.07849},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.07849}, 
}
```