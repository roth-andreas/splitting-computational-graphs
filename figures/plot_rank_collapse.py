import torch
import torch_geometric as pyg
import numpy as np
import matplotlib.pyplot as plt

from models import SimpleModel


def get_data():
    dataset = pyg.datasets.Planetoid(root='./datasets/', name='Cora')
    return pyg.transforms.LargestConnectedComponents(1, 'strong')(dataset[0])


def run(conv, num_layers):
    data = get_data()
    data.edge_index = pyg.utils.remove_self_loops(data.edge_index)[0]

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel(data.num_node_features, 16, 16, num_layers, conv).to(device)

    data.to(device)
    with torch.no_grad():
        model.eval()
        _, stats = model(data)

    return torch.FloatTensor(stats).cpu()


def plot_energies(stat_list, log_y=True, xlabel='Number of layers', ylabel='Rank-one Distance'):
    plt.figure(figsize=(5, 2.5))
    fontsize = 12
    plt.rcParams.update({'font.size': fontsize})
    plt.set_cmap(plt.get_cmap('viridis'))
    colors = ['#55C667', '#404788', '#55C667', '#404788']
    for i, (conv, stat_list) in enumerate(stat_list.items()):
        mean = stat_list['mean']
        y = np.arange(0, len(mean))
        linestyle = '-' if conv.startswith('MRS') else '--'
        plt.plot(y, mean, linestyle, label=f"{conv}", c=colors[i])
        plus = stat_list['min']
        minus = stat_list['max']
        #plt.fill_between(y, minus, plus, alpha=0.4, color=colors[i])
    if log_y:
        plt.yscale('log')
    plt.legend(ncol=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'./figures/rank_one_diff.svg', bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    num_layers = 128
    stat_list = {}
    for conv in ['GCN', 'SAGE', 'MRS-GCN', 'MRS-SAGE']:
        conv_stats = []
        for seed in range(50):
            pyg.seed_everything(seed)
            conv_stats.append(run(conv, num_layers))
        stats = torch.stack(conv_stats)
        stat_list[conv] = {'mean': stats.nanmean(0),
                           'min': torch.nan_to_num(stats, nan=10).min(0)[0],
                           'max': torch.nan_to_num(stats, nan=1e-20).max(0)[0]}
    plot_energies(stat_list)
