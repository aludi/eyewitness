import ast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import textwrap
import json


def plot():
    #plot_outcomes_histogram_all()
    #plot_dif_per_BN_collective()
    plot_KL_dif()


def plot_KL_dif():
    df = pd.read_csv(f"data/results/joints/jointDFs.csv")
    x = df.groupby("BN")[['KL-val', "difference"]].sum().reset_index()
    x.plot(kind="bar", x="BN", y=["KL-val"])
    plt.savefig(f'figures/jointcollective.png')
    #plt.show()
    plt.clf()

    folder_path = f"data/results/joints/individual"
    all_outcomes = os.listdir(folder_path)
    co = []
    for outcome in all_outcomes:
        fp = os.path.join(folder_path, outcome)
        df = pd.read_csv(fp)
        x = df.groupby("BN")[['KL-val', "difference"]].sum().reset_index()
        x["hyp"] = outcome
        print(x)
        co.append(x)
    df = pd.concat(co)
    print(df)
    #df = df.groupby("BN")[['KL-val']]
    colors = ['#FF6347', '#3CB371', '#1E90FF']
    df.boxplot(column='KL-val', by='BN', grid=False, patch_artist=True,
               boxprops=dict(facecolor=colors[0], color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               flierprops=dict(marker='o', color='red', markersize=5)
               )
    #ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.title('KL-values per BN', fontsize=16)
    plt.ylabel('KL-value')
    plt.savefig(f'figures/boxplotKL.png')
    plt.show()
    plt.clf()



'''' OLD FUNCTIONS'''

def plot_outcomes_histogram(fp, e, bn):
    df = pd.read_csv(fp)
    plt.hist(df['PTrue'], bins=20, edgecolor='tab:blue')  # Adjust 'bins' as needed
    plt.title(f'Histogram of Probability that Hypothesis Steals(X,Y) is True in {bn}')
    plt.xlabel(f'Probability P(Hyp=True|{e})')
    plt.xlim(0, 1)
    plt.ylabel('Frequency of occurrence of this probability')
    plt.show()
    plt.savefig(f'figures/{bn}probabilities{e}.png')


def plot_dif_per_BN():
    folder_path = f"data/results/difference"
    all_outcomes = os.listdir(folder_path)

    '''for outcome in all_outcomes:
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
            #plt.show()

            plt.savefig(f'figures/deltasPlotted/{bn}{ev}difference.png')
            plt.close()
    '''
    df = pd.read_csv(f"data/results/difference/allresults.csv")
    df.boxplot(["DPTrue"], by=["Evidence", "BN"])
    plt.xticks(rotation=90, fontsize=5)
    plt.savefig(f'figures/deltasPlotted/alldifsboxplot.png')
    #plt.show()



def plot_dif_per_BN_collective():

    relevant_evidence_vals = [
        {},
        {"Testimony": "True"},
        {"Testimony": "False"},
        {"Testimony": "True",
         "objectivityReliability":"True",
          "veracityReliability": "True",
          "vision_observationReliability": "True"},
        {"Testimony": "True",
         "objectivityReliability": "True",
         "veracityReliability": "True",
         "vision_observationReliability": "False"},
        {"Testimony": "True",
         "objectivityReliability": "False",
         "veracityReliability": "True",
         "vision_observationReliability": "True"},
         {"Testimony": "True",
         "objectivityReliability": "True",
         "veracityReliability": "False",
         "vision_observationReliability": "True"},
        {"Testimony": "False",
         "objectivityReliability": "True",
         "veracityReliability": "True",
         "vision_observationReliability": "True"},
        {"Testimony": "True",
         "objectivityReliability": "True",
         "veracityReliability": "False",
         "vision_observationReliability": "False"},
        {"Testimony": "True",
         "objectivityReliability": "False",
         "veracityReliability": "True",
         "vision_observationReliability": "False"},
        {"Testimony": "True",
         "objectivityReliability": "False",
         "veracityReliability": "False",
         "vision_observationReliability": "True"},
        {"Testimony": "False",
         "objectivityReliability": "True",
         "veracityReliability": "True",
         "vision_observationReliability": "True"},
        {"Testimony": "False",
         "objectivityReliability": "False",
         "veracityReliability": "False",
         "vision_observationReliability": "False"}
    ]

    df = pd.read_csv(f"data/results/collective/difference.csv")
    ordered_categories = ['f', 't', 'h', 'hb']
    df['BN'] = pd.Categorical(df['BN'], categories=ordered_categories, ordered=True)

    # Flatten the list of dicts into one set of key-value pairs to match
    target_items = set()
    for d in relevant_evidence_vals:
        #print(d)
        #print(d.items())
        target_items.update(d.items())

    #print(target_items)

    # Subset the DataFrame
    #print(relevant_evidence_vals)

    #print(df['Evidence'].apply(lambda x: any(item in ast.literal_eval(x).items() for item in target_items)))
    df['Evidence'] = df['Evidence'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # List of dicts to match exactly
    #dict_list = [{'one': 'True'}, {'three': 'True'}]

    # Filter
    filtered_df = df[df['Evidence'].isin(relevant_evidence_vals)]
    pd.set_option('display.max_columns', None)

    print(filtered_df)
    df = filtered_df
    df["Evidence"] = df["Evidence"].astype(str)
    print(df)

    #df = df[df['Evidence'].apply(lambda x: any(item in ast.literal_eval(x).items() for item in target_items))]

    print(df["Evidence"].unique())
    df.sort_values(by=['Evidence'])

    #df.boxplot(["DT"], by=["BN", "Evidence"])
    g = sns.FacetGrid(df, col="Evidence", col_wrap=4, height=3)  # Adjust col_wrap as needed
    g.map(sns.stripplot, "BN", "DT", palette="tab10", size=8)

    # Add titles and adjust spacing
    g.set_axis_labels("BN", "DT")
    g.set_titles("{col_name}")

    # Wrap titles to be more readable
    for ax in g.axes.flat:
        title = ax.get_title()  # Get the title
        wrapped_title = textwrap.fill(title, width=35)  # Wrap at width 20
        ax.set_title(wrapped_title, fontsize=11)

    for ax in g.axes.flat:
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    g.fig.suptitle("Difference between Ground Truth and BN posterior prediction per BN for Each Evidence Category", fontsize=16)
    g.tight_layout()
    plt.savefig(f'figures/deltasPlotted/collectiveDif.png')
    plt.show()
    plt.clf()

    df_melted = df.melt(id_vars=['Evidence', 'BN'], value_vars=['PTrue', 'FTrue'],
                        var_name='Type', value_name='Probability')

    g = sns.catplot(
        data=df_melted,
        kind="bar",
        x="BN", y="Probability", hue="Type",
        col="Evidence",
        height=3, aspect=1,
        col_wrap=4
    )

    g.set_titles("Evidence: {col_name}")
    g.set_axis_labels("BN", "Probability")

    for ax in g.axes.flat:
        title = ax.get_title()  # Get the title
        wrapped_title = textwrap.fill(title, width=40)  # Wrap at width 20
        ax.set_title(wrapped_title, fontsize=9)

    for ax in g.axes.flat:
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    g.fig.suptitle("Ground Truth and BN posterior prediction per BN for Each Evidence Category", fontsize=16)

    plt.tight_layout()
    plt.savefig(f'figures/collectiveProbs.png')

    plt.show()
    plt.clf()



def plot_differences():
    # get all evidence sets that we have results for by peeking into a folder
    folder_path = f"data/results/gt"
    ev_set_names = os.listdir(folder_path)

    #for ev_set in ev_set_names:
    for bn_type in ["gt", "T", "F", "H", "HB"]:
        folder_path = f"data/results/{bn_type.lower()}"
        hlist = os.listdir(folder_path)
        for h in hlist:
            print(h)

            fp = os.path.join(folder_path, h)

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


        plt.title(f'Histogram of Probability that Hypothesis Steals(X,Y) is True per Evidence Set {n_s}')
        plt.xlabel(f'Probability P(Hyp=True|{n_s})')
        plt.xlim(0, 1.05)
        plt.ylabel('Frequency of occurrence of this probability')
        plt.legend()
        plt.savefig(f'figures/difference/{n_s}probabilities.png')
        plt.clf()



def plot_outcomes_histogram_all():
    folder_path = f"data/results/gt"
    hyps = []
    evidence = set()
    for filename in os.listdir(folder_path):
        hyp = filename.split("-gt.csv")[0]
        hyps.append(hyp)
        fp = os.path.join(folder_path, filename)
        df = pd.read_csv(fp)
        d_ev = set(df["Evidence"].to_list())
        evidence.update(d_ev)

    '''print(hyps)
    print(evidence)'''


    for ev in evidence:
        dict_d = {"gt":[], "f":[], "h":[]}
        for bn in ["-gt", "f", "h"]:


            for hyp in hyps:
                if bn == "-gt":
                    folder_path = f"data/results/gt"
                    filename = f"{hyp}{bn}"

                else:
                    folder_path = f"data/results/{bn}"
                    filename = f"{bn}{hyp}"

                fn = f"{filename}.csv"
                fp = os.path.join(folder_path, fn)
                df = pd.read_csv(fp)
                '''print(bn)
                print(ev)
                print(df)
                print(type(ev))'''

                d =  normalize_dict_string(ev)
                df['nE'] = df['Evidence'].apply(normalize_dict_string)

                # Step 2: Subset df_a where NormalizedEvidence exists in df_b
                df = df[df['nE'] == d]

                #df = df[df["Evidence"]== ev]
                #print(df)
                if bn == "-gt":
                    #print(df["FTrue"])
                    dict_d["gt"].append(df["FTrue"].item())

                else:
                    if df.size > 0:
                        dict_d[bn].append(df["PTrue"].item())
                    else:
                        dict_d[bn].append(0)

        #print(dict_d)

        '''plt.hist(dict_d['gt'], bins=20, alpha=0.3, label='gt', density=True)
        plt.hist(dict_d['f'], bins=20, alpha=0.3, label='f', density=True)
        plt.hist(dict_d['h'], bins=20, alpha=0.3, label='h', density=True)'''

        sns.kdeplot(dict_d['gt'], label='gt', fill=True, alpha=0.3)
        sns.kdeplot(dict_d['f'], label='f', fill=True, alpha=0.3)
        sns.kdeplot(dict_d['h'], label='h', fill=True, alpha=0.3)


        plt.legend()
        plt.title(f'{ev[1:-1]}', fontsize=10)
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.xlim(0, 1)
        plt.savefig(f'figures/histograms/{ev}.png')
        plt.clf()
        #plt.show()



def normalize_dict_string(s):
    d = ast.literal_eval(s)  # safely convert string to dict
    #print(json.dumps(d, sort_keys=True))
    for k in d.keys():
        d[k] = str(d[k])
    return json.dumps(d, sort_keys=True)


def plot_outcomes_histogram_all1():

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
