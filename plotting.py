import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    for bn_type in ["", "T", "F", "HB", "H"]:
        for fp in [f"data/results/{bn_type.lower()}noEvidence.csv",
                   f"data/results/{bn_type.lower()}SEEN.csv",
                   f"data/results/{bn_type.lower()}NOTSEEN.csv",
                   f"data/results/{bn_type.lower()}SEENANDRELIABLE.csv"
                   ]:

            s = fp.split(f"results/{bn_type.lower()}")[1]
            s = s.split(".")[0]
            df = pd.read_csv(fp)

            if df['PTrue'].nunique() == 1:
                val = df['PTrue'].iloc[0]
                plt.axvline(val, linestyle='-', label=f'{s} = {val}', color='red')
            else:

                sns.histplot(data=df, x='PTrue', binwidth=0.01, label=f'{s} Probabilities in {bn_type}')
            #plt.hist(df['PTrue'], bins=20, alpha=0.4, label=f'{s} Probabilities', density=True)  # Adjust 'bins' as needed

        plt.title(f'Histogram of Probability that Hypothesis Steals(X,Y) is True in {bn_type}')
        plt.xlabel(f'Probability P(Hyp=True|E) in {bn_type}')
        plt.xlim(0, 1.05)
        plt.ylabel('Frequency of occurrence of this probability')
        plt.legend()
        plt.savefig(f'figures/{bn_type}probabilities.png')
        plt.clf()

plot_outcomes_histogram_all()