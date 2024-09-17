#!/usr/bin/zsh

# Baselines
python run.py --conv_type=dir-gcn --dataset=arxiv-year --hidden_dim=256 --dropout=0.0 --jk=cat --num_layers=6 --lr=0.005 --alpha=0.5 --patience=200 --num_runs=5

python run.py --conv_type=dir-gcn --dataset=snap-patents --hidden_dim=32 --dropout=0.0 --normalize --jk=max --num_layers=5 --lr=0.01 --alpha=0.5 --patience=200 --num_runs=5

python run.py --conv_type=dir-gcn --dataset=chameleon --hidden_dim=128 --dropout=0.0 --normalize --jk=max --num_layers=5 --lr=0.005 --alpha=1.0 --patience=400 --num_runs=10

python run.py --conv_type=dir-gcn --dataset=squirrel --hidden_dim=128 --dropout=0.0 --normalize --jk=max --num_layers=4 --lr=0.01 --alpha=1.0 --patience=400 --num_runs=10

python run.py --conv_type=dir-sage --dataset=directed-roman-empire --hidden_dim=256 --dropout=0.2 --jk=cat --num_layers=5 --lr=0.01 --alpha=0.5 --patience=200 --num_runs=10

conv_type=dag-gcn
dataset=arxiv-year
for lr in {0.01,0.005,0.001,0.0005}; do
  for dropout in {0.0,0.2,0.4,0.6}; do
    for num_layers in {4,5,6}; do
      python run.py --conv_type=${conv_type} --dataset=${dataset} --hidden_dim=256 --dropout=${dropout} --jk=cat --num_layers=${num_layers} --lr=${lr} --alpha=0.5 --patience=200 --num_runs=5
    done
  done
done

dataset=snap-patents
for lr in {0.01,0.005,0.001,0.0005}; do
  for dropout in {0.0,0.2,0.4,0.6}; do
    for num_layers in {4,5,6}; do
      python run.py --conv_type=${conv_type} --dataset=${dataset} --hidden_dim=32 --dropout=${dropout} --normalize --jk=max --num_layers=${num_layers} --lr=${lr} --alpha=0.5 --patience=200 --num_runs=5
    done
  done
done

dataset=chameleon
for lr in {0.01,0.005,0.001,0.0005}; do
  for dropout in {0.0,0.2,0.4,0.6}; do
    for num_layers in {4,5,6}; do
      python run.py --conv_type=${conv_type} --dataset=${dataset} --hidden_dim=128 --dropout=${dropout} --normalize --jk=max --num_layers=${num_layers} --lr=${lr} --alpha=1.0 --patience=400 --num_runs=10
    done
  done
done

dataset=squirrel
for lr in {0.01,0.005,0.001,0.0005}; do
  for dropout in {0.0,0.2,0.4,0.6}; do
    for num_layers in {4,5,6}; do
      python run.py --conv_type=${conv_type} --dataset=${dataset} --hidden_dim=128 --dropout=${dropout} --normalize --jk=max --num_layers=${num_layers} --lr=${lr} --alpha=1.0 --patience=400 --num_runs=10
    done
  done
done

conv_type=da-sage
dataset=directed-roman-empire
for lr in {0.01,0.005,0.001,0.0005}; do
  for dropout in {0.0,0.2,0.4,0.6}; do
    for num_layers in {4,5,6}; do
      python run.py --conv_type=${conv_type} --dataset=${dataset} --hidden_dim=256 --dropout=${dropout} --jk=cat --num_layers=${num_layers} --lr=${lr} --alpha=0.5 --patience=200 --num_runs=10
    done
  done
done
