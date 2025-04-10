import ast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

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
    for bn_type in ["gt", "T", "F", "HB", "H"]:
        folder_path = f"data/results/{bn_type.lower()}"
        for filename in os.listdir(folder_path):
            print(folder_path)
            print(filename)
            fp = os.path.join(folder_path, filename)
            s = fp.split(f"results/{bn_type.lower()}/")[1]
            s = s.split(".")[0]
            if s != "noEvidence":
                s = ast.literal_eval(s)
                n_s = ""
                for k in s.keys():
                    val = s[k][0]
                    s_key = k
                    if bn_type != "HB":
                        s_key = k.split("R")
                        s_key = s_key[0]
                        if "_" in s_key:    #breaking "vision_observation" into "vision"
                            s_key = s_key.split("_")[0]
                    n_s = f"{n_s}{s_key}:{val} "
            else:
                n_s = "{}"

            df = pd.read_csv(fp)
            if df['PTrue'].nunique() == 1:
                val = df['PTrue'].iloc[0]
                plt.axvline(val, linestyle='-', label=f'{n_s} = {val}', color='red')
            else:
                #sns.histplot(data=df, x='PTrue', binwidth=0.01, label=f'{s} Probabilities in {bn_type}')
                plt.hist(df['PTrue'], bins=20, alpha=0.4, label=f'{n_s}')  # Adjust 'bins' as needed

        plt.title(f'Histogram of Probability that Hypothesis Steals(X,Y) is True in {bn_type}')
        plt.xlabel(f'Probability P(Hyp=True|E) in {bn_type}')
        plt.xlim(0, 1.05)
        plt.ylabel('Frequency of occurrence of this probability')
        plt.legend()
        plt.savefig(f'figures/{bn_type}probabilities.png')
        plt.clf()

#plot_outcomes_histogram_all()