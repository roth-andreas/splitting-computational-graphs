#!/usr/bin/zsh

# Experiments on ZINC
for seed in {0,1,2,3,4}; do
  for lr in {0.005,0.001,0.0005}; do
    for layers in {1,2,4,8,16}; do
      for name in {ZINC-DA-GCN,ZINC-GCN}; do
        for ordering_type in degree; do
          config=configs/${name}.yaml
          # Base
          args="--cfg=$config --repeat 1 gnn.dim_inner 700 dataset.name subset optim.base_lr $lr seed $seed gnn.layers_mp $layers name_tag ${layers}_${lr}_${ordering_type}.%J"
          python main.py ${args}
          # Base + Res
          args="--cfg=$config --repeat 1 gnn.dim_inner 700 dataset.name subset gnn.residual True optim.base_lr $lr seed $seed gnn.layers_mp $layers name_tag ${layers}_${lr}_${ordering_type}_res.%J"
          python main.py ${args}
          # Base + JK
          args="--cfg=$config --repeat 1 gnn.dim_inner 700 dataset.name subset gnn.jk cat optim.base_lr $lr seed $seed gnn.layers_mp $layers name_tag ${layers}_${lr}_${ordering_type}_jk.%J"
          python main.py ${args}
          # Base + Res
          args="--cfg=$config --repeat 1 gnn.dim_inner 700 dataset.name subset dataset.node_encoder_name Atom+LapPE posenc_LapPE.enable True optim.base_lr $lr seed $seed gnn.layers_mp $layers name_tag ${layers}_${lr}_${ordering_type}_res.%J"
          python main.py ${args}
        done
      done
    done
  done
done
