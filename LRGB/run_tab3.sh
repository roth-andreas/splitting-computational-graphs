#!/usr/bin/zsh

# Experiments on ZINC

# GCN
for seed in {0,1,2}; do
  for lr in {0.001,0.0003,0.0001}; do
    for layers in {1,2,4,8,16}; do
      for name in ZINC-GCN; do
            config=configs/${name}.yaml
            args="--cfg=$config --repeat 1 gnn.dim_inner 700 dataset.name subset gnn.dim_inner 64 optim.base_lr $lr seed $seed gnn.layers_mp $layers name_tag ${layers}_${lr}.%J"
            python main.py ${args}
        done
      done
    done
  done
done

# DA-GCN with orderings Random, Features, PPR, Degree
for seed in {0,1,2}; do
  for lr in {0.001,0.0003,0.0001}; do
    for layers in {1,2,4,8,16}; do
      for name in ZINC-DA-GCN; do
        for ordering_type in {random,features,pagerank,degree}; do
            config=configs/${name}.yaml
            args="--cfg=$config --repeat 1 gnn.dim_inner 700 dataset.name subset gnn.ordering_type ${ordering_type} gnn.dim_inner 64 optim.base_lr $lr seed $seed gnn.layers_mp $layers name_tag ${layers}_${lr}_${ordering_type}.%J"
            python main.py ${args}
        done
      done
    done
  done
done
