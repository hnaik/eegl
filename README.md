# Explanation Enhanced Graph Learning (EEGL)

EEGL is an iterative framework that enhances Graph Neural Networks (GNNs) by mining frequent subgraphs from GNN explanations and feeding them back as additional node features.

## Requirements

- Python 3.13+
- CUDA-capable GPU (recommended)
- [uv](https://docs.astral.sh/uv/) for Python environment management
- [Podman](https://podman.io/) or Docker for containerised development

## Setup

### Local development

Install dependencies:

```sh
uv sync
```

Install the [Gaston](https://liacs.leidenuniv.nl/~nijssensgr/gaston/) frequent subgraph miner (requires `gcc`, `make`, `wget`):

```sh
make gaston
```

### Dev container (VSCode)

Open the repository in VSCode and select **Reopen in Container** when prompted. The dev container builds from `.devcontainer/Dockerfile` and runs `uv sync` automatically on creation.

### Container (manual)

```sh
make docker-build   # build image (uses podman if available, else docker)
make docker-run     # build and start container
make docker-login   # open a shell in the running container
```

## Environment variables

| Variable | Description |
|---|---|
| `EEGL_SOLVER_PATH` | Path to the `glasgow_subgraph_solver` binary |
| `PYTHONPATH` | Should point to the repository root |

## Makefile targets

| Target | Description |
|---|---|
| `env` | Create / update the uv virtual environment |
| `distclean` | Remove the virtual environment |
| `gaston` | Build and install the Gaston subgraph miner |
| `docker-build` | Build the container image |
| `docker-run` | Build and start the container |
| `docker-login` | Open a shell in the running container |
| `jupyter` | Start JupyterLab on port 8080 |
| `check-cuda` | Print CUDA device information |

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