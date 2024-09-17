import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_losses = []

mean_losses = np.array([
    pd.read_csv('results/DA-SAGE_losses.csv', delimiter=';')['DA-SAGE_mean'].values,
    pd.read_csv('results/DA-GCN_losses.csv', delimiter=';')['DA-GCN_mean'].values,
    pd.read_csv('results/GCN_losses.csv', delimiter=';')['GCN_mean'].values,
    pd.read_csv('results/SAGE_losses.csv', delimiter=';')['SAGE_mean'].values
])

std_losses = np.array([
    pd.read_csv('results/DA-SAGE_losses.csv', delimiter=';')['DA-SAGE_std'].values,
    pd.read_csv('results/DA-GCN_losses.csv', delimiter=';')['DA-GCN_std'].values,
    pd.read_csv('results/GCN_losses.csv', delimiter=';')['GCN_std'].values,
    pd.read_csv('results/SAGE_losses.csv', delimiter=';')['SAGE_std'].values
])

y = np.arange(500)
plt.figure(figsize=(5, 2.5))
fontsize = 12
plt.rcParams.update({'font.size': fontsize})
plt.plot(y, mean_losses[3], '--', color='#55C667', label="GCN")
plt.plot(y, mean_losses[2], '--', color='#404788', label="SAGE")
plt.plot(y, mean_losses[1], color='#55C667', label="MRS-GCN")
plt.plot(y, mean_losses[0], color='#404788', label="MRS-SAGE")

plt.fill_between(y, mean_losses[0] - std_losses[0], mean_losses[0] + std_losses[0], alpha=0.2, color='#404788')
plt.fill_between(y, mean_losses[1] - std_losses[1], mean_losses[1] + std_losses[1], alpha=0.2, color='#55C667')
plt.fill_between(y, mean_losses[2] - std_losses[2], mean_losses[2] + std_losses[2], alpha=0.2, color='#404788')
plt.fill_between(y, mean_losses[3] - std_losses[3], mean_losses[3] + std_losses[3], alpha=0.2, color='#55C667')

plt.ylabel(f'Training loss')
plt.xlabel('Number of steps')
plt.legend(ncol=2)
plt.yscale('log')

plt.savefig(f'./figures/loss_peptides_func.svg', bbox_inches='tight')
plt.show()
