#!/usr/bin/zsh

# Experiments on ZINC
for seed in {0,1,2}; do
  for lr in {0.001,0.0003,0.0001}; do
    for layers in {1,2,4,8,16}; do
      for name in {ZINC-DA-GCN,ZINC-GCN,ZINC-DA-SAGE,ZINC-SAGE}; do
        for ordering_type in degree; do
          config=configs/${name}.yaml
          args="--cfg=$config --repeat 1 gnn.dim_inner 64 optim.base_lr $lr seed $seed gnn.layers_mp $layers dataset.name subset name_tag ${layers}_${lr}_${ordering_type}.%J"
          python main.py ${args}
        done
      done
    done
  done
done

# Experiments on Peptides-Func
for seed in {0,1,2,3,4}; do
  for lr in {0.005,0.001,0.0005}; do
    for layers in {1,2,4,8,16}; do
      for name in {peptides-func-DA-GCN,peptides-func-GCN,peptides-func-DA-SAGE,peptides-func-SAGE}; do
        for ordering_type in degree; do
          config=configs/${name}.yaml
          args="--cfg=$config --repeat 1 optim.base_lr $lr seed $seed gnn.layers_mp $layers name_tag ${layers}_${lr}_${ordering_type}.%J"
          python main.py ${args}
        done
      done
    done
  done
done

# Experiments on Peptides-Struct
for seed in {0,1,2,3,4}; do
  for lr in {0.005,0.001,0.0005}; do
    for layers in {1,2,4,8,16}; do
      for name in {peptides-struct-DA-GCN,peptides-struct-GCN,peptides-struct-DA-SAGE,peptides-struct-SAGE}; do
        for ordering_type in degree; do
          config=configs/${name}.yaml
          args="--cfg=$config --repeat 1 optim.base_lr $lr seed $seed gnn.layers_mp $layers name_tag ${layers}_${lr}_${ordering_type}.%J"
          python main.py ${args}
        done
      done
    done
  done
done
