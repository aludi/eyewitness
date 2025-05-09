import ast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import textwrap

def plot_outcomes_histogram(fp, e, bn):
    df = pd.read_csv(fp)
    plt.hist(df['PTrue'], bins=20, edgecolor='tab:blue')  # Adjust 'bins' as needed
    plt.title(f'Histogram of Probability that Hypothesis Steals(X,Y) is True in {bn}')
    plt.xlabel(f'Probability P(Hyp=True|{e})')
    plt.xlim(0, 1)
    plt.ylabel('Frequency of occurrence of this probability')
    plt.show()
    plt.savefig(f'figures/{bn}probabilities{e}.png')


def plot():
    #plot_outcomes_histogram_all()
    #plot_differences()
    #plot_dif_per_BN()
    plot_dif_per_BN_collective()


def plot_dif_per_BN():
    folder_path = f"data/results/difference"
    all_outcomes = os.listdir(folder_path)

    for outcome in all_outcomes:
        if outcome != "allresults.csv":
            fp = f"{folder_path}/{outcome}"
            df = pd.read_csv(fp)
            a, bn, ev = outcome.split("-")
            plt.hist(df['DPTrue'], bins=20, alpha=0.4)  # Adjust 'bins' as needed

            plt.title(f'Histogram of difference in BN {bn}, ev {ev}')
            plt.xlabel(f'Difference between GT and BN posterior prediction in {bn}')
            plt.xlim(0, 1.05)
            plt.ylabel('Frequency of occurrence of this difference')
            plt.legend()
            plt.savefig(f'figures/deltasPlotted/{bn}{ev}difference.png')
            plt.close()
            #plt.show()

    df = pd.read_csv(f"data/results/difference/allresults.csv")
    df.boxplot(["DPTrue"], by=["evidence", "bn"])
    plt.xticks(rotation=90, fontsize=5)
    plt.savefig(f'figures/deltasPlotted/alldifsboxplot.png')
    plt.show()


def plot_dif_per_BN_collective():
    df = pd.read_csv(f"data/results/collective/difference.csv")
    ordered_categories = ['f', 't', 'h', 'hb']
    df['BN'] = pd.Categorical(df['BN'], categories=ordered_categories, ordered=True)

    #df.boxplot(["DT"], by=["BN", "Evidence"])
    g = sns.FacetGrid(df, col="Evidence", col_wrap=4, height=3)  # Adjust col_wrap as needed
    g.map(sns.stripplot, "BN", "DT", palette="tab10", size=6)

    # Add titles and adjust spacing
    g.set_axis_labels("BN", "DT")
    g.set_titles("{col_name}")

    # Wrap titles to be more readable
    for ax in g.axes.flat:
        title = ax.get_title()  # Get the title
        wrapped_title = textwrap.fill(title, width=60)  # Wrap at width 20
        ax.set_title(wrapped_title, fontsize=7)

    for ax in g.axes.flat:
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    g.fig.suptitle("Difference between Ground Truth and BN posterior prediction per BN for Each Evidence Category", fontsize=16)
    g.tight_layout()
    plt.savefig(f'figures/deltasPlotted/collectiveDif.png')
    plt.clf()
    #plt.show()





def plot_differences():
    # get all evidence sets that we have results for by peeking into a folder
    folder_path = f"data/results/gt"
    ev_set_names = os.listdir(folder_path)

    for ev_set in ev_set_names:
        for bn_type in ["gt", "T", "F", "H", "HB"]:
            folder_path = f"data/results/{bn_type.lower()}"
            if ev_set in os.listdir(folder_path):

                fp = os.path.join(folder_path, ev_set)

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
                    plt.axvline(val, linestyle='-', label=f'{bn_type}', color='red')
                else:
                    #sns.histplot(data=df, x='PTrue', binwidth=0.01, label=f'{s} Probabilities in {bn_type}')
                    plt.hist(df['PTrue'], bins=20, alpha=0.4, label=f'{bn_type}')  # Adjust 'bins' as needed
            else:
                # somehow fix HB resolution
                pass

        plt.title(f'Histogram of Probability that Hypothesis Steals(X,Y) is True per Evidence Set {n_s}')
        plt.xlabel(f'Probability P(Hyp=True|{n_s})')
        plt.xlim(0, 1.05)
        plt.ylabel('Frequency of occurrence of this probability')
        plt.legend()
        plt.savefig(f'figures/difference/{n_s}probabilities.png')
        plt.clf()



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