import matplotlib.pyplot as plt
import pandas as pd

def plot_outcomes_histogram(fp, e, bn):
    df = pd.read_csv(fp)
    plt.hist(df['PTrue'], bins=20, edgecolor='tab:blue')  # Adjust 'bins' as needed
    plt.title(f'Histogram of Probability that Hypothesis Steals(X,Y) is True in {bn}')
    plt.xlabel(f'Probability P(Hyp=True|{e})')
    plt.xlim(0, 1)
    plt.ylabel('Frequency of occurrence of this probability')
    plt.show()
    plt.savefig(f'figures/{bn}probabilities{e}.png')


def plot_outcomes_histogram_all():
    for bn_type in ["", "t", "f", "hb", "h"]:
        for fp in [f"data/results/{bn_type}noEvidence.csv",
                   f"data/results/{bn_type}SEEN.csv",
                   f"data/results/{bn_type}NOTSEEN.csv",
                   f"data/results/{bn_type}SEENANDRELIABLE.csv"
                   ]:

            df = pd.read_csv(fp)
            plt.hist(df['PTrue'], bins=20, alpha=0.4, label=f'{fp} Probabilities')  # Adjust 'bins' as needed

        plt.title(f'Histogram of Probability that Hypothesis Steals(X,Y) is True in {bn_type}')
        plt.xlabel(f'Probability P(Hyp=True|E) in {bn_type}')
        plt.xlim(0, 1)
        plt.ylabel('Frequency of occurrence of this probability')
        plt.legend()
        plt.savefig(f'figures/{bn_type}probabilities.png')
        plt.clf()
