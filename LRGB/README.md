# Implementational Details

This implementation is based on the paper "Where Did the Gap Go? Reassessing the Long-Range Graph Benchmark" by Tönshoff et al., 2023.

Our models can be found in graphsgps/layer/. Our results for Tables 1, 2, and 3 were produced by the runs given in run_tab1.sh, run_tab2.sh, and run_tab3.sh. For optimal


## Installation

See README_base_implementation.md for installation instructions.

## Running Experiments

### Grid Search

Our grid searches for Tables 1, 2, 3 are based on the commands in run_tab1.sh, run_tab2.sh, and run_tab3.sh.

### Individual Experiments
A single experiment can be run with the following command:

```bash
conda activate graphgps
python main.py --cfg configs/${dataset}-${model}.yaml
```

Replace ${dataset} with one of {ZINC, peptides-func, peptides-struct} and ${model} with one of {gcn, sage, da-gcn, da-sage}.

Hyperparameters can be set as arguments to the main.py script. We used the following hyperparameters for our experiments:

* --repeat : Number of runs 
* gnn.dim_inner : Dimension of the hidden layers
* optim.base_lr : Base learning rate for cosine lr scheduler 
* seed : Random seed
* gnn.layers_mp : Number of message passing layers
* gnn.ordering_type : Type of node ordering used for DA models {random,features,pagerank,degree}
* gnn.max_params : Maximum allowed number of parameters
* gnn.residual : Whether to use residual connections
* gnn.jk : Which type of jump connection to use {None,cat,max}

