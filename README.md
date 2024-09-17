# Preventing Representation Rank Collapse in MPNNs by Splitting the Computational Graph

This repository contains the official implementation. Three different code bases are used for the experiments.

## Reference Implementation

Our implementations of MRS-GCN and MRS-SAGE can be found in the './figures/' directory. The implementation is based on PyTorch Geometric.

### Experiments on ZINC, Peptides-Func, and Peptides-Struct

These experiments are based on the Long Range Graph Benchmark (LRGB) and the updated implementation by Tönshoff et al., 2023. Check './LRGB/' for all details regarding these experiments.

### Experiments on Large-Scale Directed Graphs

Experiments for Chameleon, Squirrel, Arxiv-Year, Snap-Patents, Roman-Empire are based on the implementation by Rossi et al., 2023. Check './directed-graph-neural-network/' for all details regarding these experiments.

### Generating Figures

To reproduce Figures 3 and 4 of our paper, check './figures/' for all details. 
