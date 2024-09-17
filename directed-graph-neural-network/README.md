# Implementation Details

This implementation is based on the paper "Edge Directionality Improves Learning on Heterophilic Graphs" by Rossi et al. (2023). Their README (README_base_implementation.md)

### Setup with conda

The same environment that is used for our reference implementation can be used. Check README_base_implementation.md for the setup with conda.

### Reproduce Experiments

To reproduce our grid search, execute 'run_experiments.py' inside './src/'. To reproduce the experiments with the best hyperparameters, run the following command:

```bash
python run.py --dataset=${dataset} --use_best_hyperparams --num_runs=10
```

inside './src/'. Replace ${dataset} with one of {chameleon,squirrel,arxiv-year,snap-patents,directed-roman-empire}.
