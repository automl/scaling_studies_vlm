import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_matrix(corr_mat, keys, filename):
    sns.clustermap(
        corr_mat,
        xticklabels=keys,
        yticklabels=keys,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
    )

    plt.savefig(filename)
    plt.close()
