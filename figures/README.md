# Generate Figures

This directory contains scripts to generate the figures of the paper.

## Setup with conda

The same environment as the one used for the LRGB experiments is used. To set up the environment, run the following commands:

```bash
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install pandas
pip install matplotlib
pip install numpy
pip install ogb

conda clean --all
```

## Rank-one distance (Figure 1)


To compare the Rank-one distance (ROD) of GCN and SAGE with their respective DA versions, run the following command:

```
python plot_rank_collapse.py
```

## Plot Training Losses (Figure 2)

```
python plot_losses.py
```