from copy import deepcopy
import pandas as pd
import numpy as np
import pyAgrum as gum
import pyAgrum.lib.image as gumimage
import copy
import re
import os, shutil
import time
from itertools import product


df = pd.read_csv("data/stealing_testimony_1.csv")

df1 = df[df["time_testimony"] == 40]
df["ReliabilityPerturbed"] = df[["vision_perturb", "memory_perturb", "veracity_perturb"]].max(axis=1)
df["Reliability"] = ~df["ReliabilityPerturbed"]


def create_tuple_list(sub_df):
    #for run in list(df1["run"].unique()):
    #df2 = df1[df1["run"] == run]
    df3 = sub_df[sub_df["action1"] == "stealing from"]
    tuple_list = list(zip(df3["subj_agent_prof"], df3["action1"], df3['object_agent']))
    # Assign this list to a new column in every row
    sub_df['stealingagents'] = [tuple_list] * len(sub_df)

    return sub_df

def create_innocent_tuple_list(sub_df):
    #for run in list(df1["run"].unique()):
    #df2 = df1[df1["run"] == run]
    prof = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (1, 1, 1), (1, 1, 2), (2, 2, 0), (2, 2, 1),
            (1, 2, 0), (2, 2, 2)]
    if len(sub_df["stealingagents"].to_list()) == 0:
        stealers = []
    else:
        stealers = sub_df["stealingagents"].to_list()[0]
    n_stealer = []
    for (a, b, c) in stealers:
        n_stealer.append((a, c))
    innocents = []
    #print(n_stealer)

    for i in range(0, 3):
        for j in range(0, 3):
            if i != j and (str(prof[i]), j) not in n_stealer:
                #print(prof[i], j)

                innocents.append((str(prof[i]),'not stealing from', j))

    #exit()

    # Assign this list to a new column in every row
    sub_df['innocentagents'] = [innocents] * len(sub_df)

    return sub_df

df = df.groupby('run', group_keys=False).apply(create_tuple_list)
df = df.groupby('run', group_keys=False).apply(create_innocent_tuple_list)
df['frequency_theft'] = df['stealingagents'].apply(len)/(df['stealingagents'].apply(len)+df["innocentagents"].apply(len))
df_grouped = df.groupby('run').agg(

    value_mean=('frequency_theft', 'mean'),
    value_sd=('frequency_theft', 'std'),
    guilty_agents=('stealingagents', lambda x: list(x)[0]),
    innocent_agents=('innocentagents', lambda x: list(x)[0]),
    len_guilty=("stealingagents", lambda x:len(list(x)[0])),
    len_inno=("innocentagents", lambda x:len(list(x)[0]))).reset_index()

df_grouped.to_csv("data/stealing.csv")
#print("mean probability of hypothesis...")
#print(df_grouped["value_mean"].mean())

df1 = df[(df["time_testimony"] == 40)]
df1 = df1[(df1["action1"] == "observed")]
mask = pd.notna(df1['action2'])
#filtered_data = df1[mask]
filtered_data = df1
#observed_agent,memory_perturb,memorized_agent
filtered_data[["run", "time_observation", "subj_agent", "action1", "object_agent", "obj_agent_prof", "action2", "indirect_agent", "testified_agent"]].sort_values(by=['run','time_observation', 'subj_agent','object_agent']).to_csv("data/testetes.csv")
filtered_data = filtered_data[["run", "time_observation", "subj_agent", "action1", "object_agent",
                               "obj_agent_prof", "action2", "indirect_agent",
                               "observed_agent", "memorized_agent","testified_agent",
                               "vision_perturb", "memory_perturb", "veracity_perturb",
                               "Reliability", "stealingagents","innocentagents"]].sort_values(by=['run','time_observation', 'subj_agent','object_agent'])

#print("FILTERED DATA")
#print(filtered_data)

# Get unique values for each category
runs = filtered_data["run"].unique()
times = filtered_data["time_observation"].unique()
subj_agents = filtered_data["subj_agent"].unique()
object_agents = filtered_data["object_agent"].unique()

# Generate all possible combinations
all_combinations = pd.DataFrame(
    list(product(runs, times, subj_agents, object_agents)),
    columns=["run", "time_observation", "subj_agent", "object_agent"]
)

#print(all_combinations)
merged = all_combinations.merge(filtered_data, on=["run", "time_observation", "subj_agent", "object_agent"], how="left", indicator=True)

# Select missing rows
missing_rows = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

# Fill other columns with default values (modify as needed)
missing_rows["action1"] = "not observed"
missing_rows["object_agent_prof"] = None
missing_rows["action2"] = None
missing_rows["indirect_agent"] = None
missing_rows["testified_agent"] = None
missing_rows["vision_perturb"] = False
missing_rows["memory_perturb"] = False
missing_rows["veracity_perturb"] = False
missing_rows["Reliability"] = True
missing_rows["observed_agent"] = None
missing_rows["memorized_agent"] = None



stealing_agents_map = filtered_data.groupby("run")["stealingagents"].first().to_dict()
inn_agents_map = filtered_data.groupby("run")["innocentagents"].first().to_dict()

# Assign the correct stealingAgents value to missing rows
missing_rows["stealingagents"] = missing_rows["run"].map(stealing_agents_map)
missing_rows["innocentagents"] = missing_rows["run"].map(inn_agents_map)


# Append missing rows to the original dataframe
df_complete = pd.concat([filtered_data, missing_rows], ignore_index=True)

#print(df_complete)

df_complete.sort_values(by=['run','time_observation', 'subj_agent','object_agent']).to_csv("data/testetes1.csv")

df_complete['hypothesis'] = df_complete.apply(
    lambda row: (row['subj_agent'], row['action1'], row["object_agent"], row["action2"], row["indirect_agent"]) if row['action1'] == 'observed' and row["action2"] == "stealing from" else False,
    axis=1
)
# hoevaak ziet agent 1 agent 2 stelen, uit ALLE mogelijke observaties (dus ook: 2 zien die niet steelt)



for hypothesis in df_complete['hypothesis'].unique():
    #print("\n\n", hypothesis)
    df_complete['H'] = np.where(df_complete['hypothesis'] == hypothesis, True, False)
    #print(df_complete["H"].value_counts()/df_complete["H"].value_counts().sum())


df1 = df_complete
df1['hypothesis'] = df1.apply(
    lambda row: (row['obj_agent_prof'], row['action2'], row["indirect_agent"]) if row['action1'] == 'observed' and row["action2"] == "stealing from" else False,
    axis=1
)


df1['statementAgent'] = df1.apply(
    lambda row: (row["subj_agent"], row["action1"], row['testified_agent'], row['action2'], row["indirect_agent"]) if row['action1'] == 'observed' and row["action2"] == "stealing from" else False,
    axis=1
)

df1['statementAgentFalse'] = df1.apply(
    lambda row: (row["subj_agent"], row["action1"], row['testified_agent'], "not stealing from", row["indirect_agent"]) if row['action1'] == 'observed' and row["action2"] == "stealing from"
                 else False,
    axis=1
)
df1['correctreport'] = df1.apply(lambda row: row['statementAgent'] in row['stealingagents'], axis=1)
df1['incorrectreport'] = df1.apply(lambda row: row['statementAgentFalse'] in row['innocentagents'], axis=1)


#print(repr(df1['hypothesis'].iloc[0]))  # Show raw string representation
#print(repr("('(1, 1, 1)', 'stealing from', 1.0)"))
crosstab_results = {}
tt_l = []
ft_l = []

df1.sort_values(by=['run','time_observation', 'subj_agent','object_agent']).to_csv("data/tetete.csv")
# HB'''
print("Hartmann & Bovens structure data")
dict_of_bn_HB = {}

for hypothesis in df1['hypothesis'].unique():
    dict_of_bn_HB[hypothesis] = {}

    for i in range(0, 3): # looping over the agents
        print("\n\n",i, "observed", hypothesis)
        #print(df1["action1"].value_counts())



        if hypothesis != False:
            (a, b, c) = hypothesis
            test_hyp = (i, "observed", a, b, c)
        else:
            test_hyp = False
        df1['H'] = np.where(df1['hypothesis'] == hypothesis, True, False)
        df1['T'] = np.where(df1['statementAgent'] == test_hyp, True, False)

        dict_of_bn_HB[hypothesis]["hypothesis"] = hypothesis
        dict_of_bn_HB[hypothesis]["H"] = df1["H"].value_counts() / df1["H"].value_counts().sum()
        dict_of_bn_HB[hypothesis]["testimony"] = f"{i} says that {hypothesis}"

        d2 = df1[["run", "time_observation", "subj_agent", "action1", "object_agent", "action2", "hypothesis", "statementAgent",  "correctreport", "incorrectreport", "Reliability", "H", "T", "stealingagents", "innocentagents"]]
        #print("aaa")
        #print(d2["action1"].value_counts())
        d2.sort_values(by=['run','time_observation', 'subj_agent','object_agent']).to_csv("data/tt.csv")
        df1[["run", "Reliability", "H", "T"]].to_csv("data/tt1.csv")

        #print(df1)
        #print(df1["H"].value_counts()/df1["H"].value_counts().sum())

        sub_df = df1[(df1["subj_agent"] == i) & (df1["action1"] == "observed")]

        #print(sub_df["Reliability"].value_counts()/sub_df["Reliability"].value_counts().sum())
        dict_of_bn_HB[hypothesis]["Reliability"] = sub_df["Reliability"].value_counts()/sub_df["Reliability"].value_counts().sum()

        #print(pd.crosstab([df1['Reliability'], df1['H']],df1["T"]))
        crosstab = pd.crosstab([df1['Reliability'], df1['H']],df1["T"], normalize="index")
        #print(crosstab)
        dict_of_bn_HB[hypothesis]["Testimony"] = crosstab

        #normalized_by_column = crosstab.div(crosstab.sum(axis=0), axis=1)
        #print(crosstab)

        #print(normalized_by_column)

        #crosstab_results[hypothesis] = normalized_by_column

dict_of_bn_F = {}
# Fenton
for hypothesis in df1['hypothesis'].unique():
    dict_of_bn_F[hypothesis] = {}
    for i in range(0, 3):
        print("\n\n",i, "observed", hypothesis)
        #df[["vision_perturb", "memory_perturb", "veracity_perturb"]
        df1["reliableVision"] = ~df1["vision_perturb"]
        df1["reliableMemory"] = ~df1["memory_perturb"]
        df1["reliableVeracity"] = ~df1["veracity_perturb"]
        df1["FentonReliability"] = df1[["reliableVision", "reliableMemory", "reliableVeracity"]].min(axis=1)

        '''print(df1["reliableVision"].value_counts())
        print(df1["reliableMemory"].value_counts())
        print(df1["reliableVeracity"].value_counts())

        print(df1["reliableVision"].value_counts()/df1["reliableVision"].value_counts().sum())
        print(df1["reliableMemory"].value_counts()/df1["reliableMemory"].value_counts().sum())
        print(df1["reliableVeracity"].value_counts()/df1["reliableVeracity"].value_counts().sum())'''

        sub_df = df1[(df1["subj_agent"] == i) & (df1["action1"] == "observed")]

        dict_of_bn_F[hypothesis]["reliableVision"] = sub_df["reliableVision"].value_counts() / sub_df[
            "reliableVision"].value_counts().sum()
        dict_of_bn_F[hypothesis]["reliableMemory"] = sub_df["reliableMemory"].value_counts() / sub_df[
            "reliableMemory"].value_counts().sum()
        dict_of_bn_F[hypothesis]["reliableVeracity"] = sub_df["reliableVeracity"].value_counts() / sub_df[
            "reliableVeracity"].value_counts().sum()

        crosstab = pd.crosstab([df1['reliableVision'], df1['reliableMemory'], df1["reliableVeracity"]], df1["FentonReliability"], normalize="index")
        #print(crosstab)
        dict_of_bn_F[hypothesis]["FentonReliability"] = crosstab


        df1['H'] = (np.where(df1['hypothesis'] == hypothesis, True, False))
        if hypothesis != False:
            (a, b, c) = hypothesis
            test_hyp = (i, "observed", a, b, c)
        else:
            test_hyp = False
        df1['H'] = np.where(df1['hypothesis'] == hypothesis, True, False)
        df1['T'] = np.where(df1['statementAgent'] == test_hyp, True, False)

        dict_of_bn_F[hypothesis]["hypothesis"] = hypothesis
        dict_of_bn_F[hypothesis]["H"] = df1["H"].value_counts() / df1["H"].value_counts().sum()
        dict_of_bn_F[hypothesis]["testimony"] = f"{i} says that {hypothesis}"

        df1[["run", "subj_agent", "action1", "hypothesis", "statementAgent",
             "correctreport", "incorrectreport", "reliableVision", "reliableMemory",
             "reliableVeracity", "FentonReliability", "H", "T", "stealingagents", "innocentagents"]].to_csv("data/ttF.csv")

        df1[["run", "reliableVision", "reliableMemory",
             "reliableVeracity", "FentonReliability", "H", "T"]].to_csv("data/ttF1.csv")

        #print(df1)
        #print(df1["H"].value_counts()/df1["H"].value_counts().sum())
        crosstab = pd.crosstab([df1['FentonReliability'], df1['H']],df1["T"], normalize="index")
        dict_of_bn_F[hypothesis]["Testimony"] = crosstab

        #print(crosstab)


#print((df1["Reliability"] == df1["FentonReliability"]).value_counts())

df1['ObservedAgent'] = df1.apply(
    lambda row: (row["subj_agent"], row["action1"], row['observed_agent'], row['action2'], row["indirect_agent"]) if row['action1'] == 'observed' and row["action2"] == "stealing from" else False,
    axis=1
)


df1['MemorizedAgent'] = df1.apply(
    lambda row: (row["subj_agent"], row["action1"], row['memorized_agent'], row['action2'], row["indirect_agent"]) if row['action1'] == 'observed' and row["action2"] == "stealing from" else False,
    axis=1
)



df1['VeracityAgent'] = df1.apply(
    lambda row: (row["subj_agent"], row["action1"], row['testified_agent'], row['action2'], row["indirect_agent"]) if row['action1'] == 'observed' and row["action2"] == "stealing from" else False,
    axis=1
)

dict_of_bn_timmer = {}

for hypothesis in df1['hypothesis'].unique():
    if hypothesis is not False:
        dict_of_bn_timmer[hypothesis] = {}
        for i in range(0, 3):
            print("\n\n",i, "observed", hypothesis)

            df1['H'] = (np.where(df1['hypothesis'] == hypothesis, True, False))
            dict_of_bn_timmer[hypothesis]["hypothesis"] = hypothesis
            dict_of_bn_timmer[hypothesis]["H"] = df1["H"].value_counts()/df1["H"].value_counts().sum()
            dict_of_bn_timmer[hypothesis]["testimony"] = f"{i} says that {hypothesis}"




            if hypothesis != False:
                (a, b, c) = hypothesis
                test_hyp = (i, "observed", a, b, c)
            else:
                test_hyp = False

            df1["reliableVision"] = ~df1["vision_perturb"]
            df1["reliableMemory"] = ~df1["memory_perturb"]
            df1["reliableVeracity"] = ~df1["veracity_perturb"]


            df1['H'] = np.where(df1['hypothesis'] == hypothesis, True, False)
            df1['Ob'] = np.where(df1['ObservedAgent'] == test_hyp, True, False)
            #print(hypothesis)
            #print(test_hyp)
            #if df1["H"].any():
            if df1["Ob"].any():
                crosstab = pd.crosstab([df1['H'], df1["reliableVision"]], df1["Ob"])
                #print("de kans op het observeren gegeven vision van de hypothese")
                #print(crosstab)
                dict_of_bn_timmer[hypothesis]["Observed H"] = crosstab

            df1['Mb'] = np.where(df1['MemorizedAgent'] == test_hyp, True, False)
            if df1["Mb"].any():
                crosstab = pd.crosstab([df1['Ob'], df1["reliableMemory"]], df1["Mb"])
                #print("de kans op het memorizen gegeven vision")
                #print(crosstab)
                dict_of_bn_timmer[hypothesis]["Objective H"] = crosstab


            df1['Vb'] = np.where(df1['VeracityAgent'] == test_hyp, True, False)
            if df1["Vb"].any():
                crosstab = pd.crosstab([df1['Mb'], df1["reliableVeracity"]], df1["Vb"])
                #print("de kans op het correcte zeggen gegeven belief/memory")
                #print(crosstab)
                dict_of_bn_timmer[hypothesis]["Testimony"] = crosstab



            '''print(df1["reliableVision"].value_counts())
            print(df1["reliableMemory"].value_counts())
            print(df1["reliableVeracity"].value_counts())

            print(df1["reliableVision"].value_counts()/df1["reliableVision"].value_counts().sum())
            print(df1["reliableMemory"].value_counts()/df1["reliableMemory"].value_counts().sum())
            print(df1["reliableVeracity"].value_counts()/df1["reliableVeracity"].value_counts().sum())'''

            sub_df = df1[(df1["subj_agent"] == i) & (df1["action1"] == "observed")]

            dict_of_bn_timmer[hypothesis]["reliableVision"] = sub_df["reliableVision"].value_counts()/sub_df["reliableVision"].value_counts().sum()
            dict_of_bn_timmer[hypothesis]["reliableMemory"] = sub_df["reliableMemory"].value_counts()/sub_df["reliableMemory"].value_counts().sum()
            dict_of_bn_timmer[hypothesis]["reliableVeracity"] = sub_df["reliableVeracity"].value_counts()/sub_df["reliableVeracity"].value_counts().sum()





dict_of_bn_hepler= {}

for hypothesis in df1['hypothesis'].unique():
    if hypothesis is not False:
        dict_of_bn_hepler[hypothesis] = {}
        for i in range(0, 3):
            print("\n\n",i, "observed", hypothesis)

            df1['H'] = (np.where(df1['hypothesis'] == hypothesis, True, False))
            dict_of_bn_hepler[hypothesis]["hypothesis"] = hypothesis
            dict_of_bn_hepler[hypothesis]["H"] = df1["H"].value_counts()/df1["H"].value_counts().sum()
            dict_of_bn_hepler[hypothesis]["testimony"] = f"{i} says that {hypothesis}"

            sub_df = df1[(df1["action1"] == "not observed") | (df1["action1"] == "observed")]

            dict_of_bn_hepler[hypothesis]["observed"] = sub_df["action1"].value_counts()/sub_df["action1"].value_counts().sum()

            if hypothesis != False:
                (a, b, c) = hypothesis
                test_hyp = (i, "observed", a, b, c)
            else:
                test_hyp = False

            df1["reliableVision"] = ~df1["vision_perturb"]
            df1["reliableMemory"] = ~df1["memory_perturb"]
            df1["reliableVeracity"] = ~df1["veracity_perturb"]


            df1['H'] = np.where(df1['hypothesis'] == hypothesis, True, False)
            df1['Ob'] = np.where(df1['ObservedAgent'] == test_hyp, True, False)
            #print(hypothesis)
            #print(test_hyp)
            #if df1["H"].any():
            if df1["Ob"].any():
                crosstab = pd.crosstab([df1['H'], df1["action1"], df1["reliableVision"]], df1["Ob"])
                #print("de kans op het observeren gegeven vision van de hypothese")
                #print(crosstab)
                dict_of_bn_hepler[hypothesis]["Observed H"] = crosstab

            df1['Mb'] = np.where(df1['MemorizedAgent'] == test_hyp, True, False)
            if df1["Mb"].any():
                crosstab = pd.crosstab([df1['Ob'], df1["reliableMemory"]], df1["Mb"])
                #print("de kans op het memorizen gegeven vision")
                #print(crosstab)
                dict_of_bn_hepler[hypothesis]["Objective H"] = crosstab


            df1['Vb'] = np.where(df1['VeracityAgent'] == test_hyp, True, False)
            if df1["Vb"].any():
                crosstab = pd.crosstab([df1['Mb'], df1["reliableVeracity"]], df1["Vb"])
                #print("de kans op het correcte zeggen gegeven belief/memory")
                #print(crosstab)
                dict_of_bn_hepler[hypothesis]["Testimony"] = crosstab




            sub_df = df1[(df1["subj_agent"] == i) & (df1["action1"] == "observed")]

            dict_of_bn_hepler[hypothesis]["reliableVision"] = sub_df["reliableVision"].value_counts()/sub_df["reliableVision"].value_counts().sum()
            dict_of_bn_hepler[hypothesis]["reliableMemory"] = sub_df["reliableMemory"].value_counts()/sub_df["reliableMemory"].value_counts().sum()
            dict_of_bn_hepler[hypothesis]["reliableVeracity"] = sub_df["reliableVeracity"].value_counts()/sub_df["reliableVeracity"].value_counts().sum()

dict_list = [dict_of_bn_HB, dict_of_bn_F, dict_of_bn_timmer, dict_of_bn_hepler]



for key in dict_of_bn_HB.keys():
    print(dict_of_bn_HB[key]["hypothesis"])
    print(dict_of_bn_HB[key]["testimony"])


    bn = gum.BayesNet(f"{dict_of_bn_HB[key]["testimony"]}")
    for key_key in dict_of_bn_HB[key].keys():
        if key_key in ["H", "Reliability", "Testimony"]:
            id_c = bn.add(gum.LabelizedVariable(key_key, key_key, 2))
    bn.addArc("H", "Testimony")
    bn.addArc("Reliability", "Testimony")
    for key_key in ["H", "Reliability", "Testimony"]:
        print(key_key)
        print(dict_of_bn_HB[key][key_key])
        if isinstance(dict_of_bn_HB[key][key_key], pd.Series):
            #print("test")
            print(dict_of_bn_HB[key][key_key])
            print(dict_of_bn_HB[key][key_key][True])
            print(dict_of_bn_HB[key][key_key][False])

            print(dict_of_bn_HB[key][key_key].tolist())

            bn.cpt(key_key).fillWith([dict_of_bn_HB[key][key_key][False], dict_of_bn_HB[key][key_key][True]]) # false true

        else:
            if type(dict_of_bn_HB[key][key_key]) is not bool:
                print(dict_of_bn_HB[key][key_key])
                for index_values, row in dict_of_bn_HB[key][key_key].iterrows():
                    print(len(dict_of_bn_HB[key][key_key]))
                    if len(dict_of_bn_HB[key][key_key]) < 4:
                        continue
                    # Convert index values into a dictionary dynamically
                    index_dict = dict(zip(dict_of_bn_HB[key][key_key].index.names, index_values))

                    # Convert row values to a dictionary
                    row_dict = row.to_dict()
                    print(index_dict, list(row_dict.values()))
                    if len(list(row_dict.values())) == 2:
                        bn.cpt(key_key)[index_dict] = list(row_dict.values())
                    else:
                        continue

                #print(f"Keys: {index_dict}, Values: {row_dict}")
    gum.saveBN(bn,f"bns/HB{dict_of_bn_HB[key]["testimony"]}.net")
    print(bn)

exit()

for key in dict_of_bn_F.keys():
    bn = gum.BayesNet(f"{dict_of_bn_F[key]["hypothesis"]}")
    for key_key in dict_of_bn_F[key].keys():
        if key_key in ["H", "FentonReliability", "reliableVision", "reliableMemory",
                       "reliableVeracity", "Testimony"]:
            id_c = bn.add(gum.LabelizedVariable(key_key, key_key, 2))
    bn.addArc("H", "Testimony")
    bn.addArc("FentonReliability", "Testimony")
    bn.addArc("reliableVision", "FentonReliability")
    bn.addArc("reliableMemory", "FentonReliability")
    bn.addArc("reliableVeracity", "FentonReliability")

    for key_key in ["H", "FentonReliability", "Testimony", "reliableVision", "reliableMemory",
                       "reliableVeracity"]:
        print(key_key)
        if isinstance(dict_of_bn_F[key][key_key], pd.Series):
            #print("test")
            print(dict_of_bn_F[key][key_key])
            print(dict_of_bn_F[key][key_key][True])
            print(dict_of_bn_F[key][key_key][False])

            print(dict_of_bn_F[key][key_key].tolist())

            bn.cpt(key_key).fillWith([dict_of_bn_F[key][key_key][False], dict_of_bn_F[key][key_key][True]]) # false true

        else:
            if type(dict_of_bn_F[key][key_key]) is not bool:
                for index_values, row in dict_of_bn_F[key][key_key].iterrows():
                    # Convert index values into a dictionary dynamically
                    index_dict = dict(zip(dict_of_bn_F[key][key_key].index.names, index_values))

                    # Convert row values to a dictionary
                    row_dict = row.to_dict()
                    print(index_dict, list(row_dict.values()))
                    bn.cpt(key_key)[index_dict] = list(row_dict.values())

                #print(f"Keys: {index_dict}, Values: {row_dict}")
    gum.saveBN(bn,f"bns/F{dict_of_bn_F[key]["hypothesis"]}.net")
    print(bn)

for key in dict_of_bn_timmer.keys():
    bn = gum.BayesNet(f"{dict_of_bn_timmer[key]["hypothesis"]}")
    for key_key in dict_of_bn_timmer[key].keys():
        if key_key in ["H", "Reliability", "reliableVision", "reliableMemory",
                       "reliableVeracity", "Testimony"]:
            id_c = bn.add(gum.LabelizedVariable(key_key, key_key, 2))
    bn.addArc("H", "Testimony")
    bn.addArc("FentonReliability", "Testimony")
    bn.addArc("reliableVision", "FentonReliability")
    bn.addArc("reliableMemory", "FentonReliability")
    bn.addArc("reliableVeracity", "FentonReliability")

    for key_key in ["H", "FentonReliability", "Testimony", "reliableVision", "reliableMemory",
                       "reliableVeracity"]:
        print(key_key)
        if isinstance(dict_of_bn_timmer[key][key_key], pd.Series):
            #print("test")
            print(dict_of_bn_timmer[key][key_key])
            print(dict_of_bn_timmer[key][key_key][True])
            print(dict_of_bn_timmer[key][key_key][False])

            print(dict_of_bn_timmer[key][key_key].tolist())

            bn.cpt(key_key).fillWith([dict_of_bn_timmer[key][key_key][False], dict_of_bn_timmer[key][key_key][True]]) # false true

        else:
            if type(dict_of_bn_timmer[key][key_key]) is not bool:
                for index_values, row in dict_of_bn_timmer[key][key_key].iterrows():
                    # Convert index values into a dictionary dynamically
                    index_dict = dict(zip(dict_of_bn_timmer[key][key_key].index.names, index_values))

                    # Convert row values to a dictionary
                    row_dict = row.to_dict()
                    print(index_dict, list(row_dict.values()))
                    bn.cpt(key_key)[index_dict] = list(row_dict.values())

                #print(f"Keys: {index_dict}, Values: {row_dict}")
    gum.saveBN(bn,f"bns/T{dict_of_bn_timmer[key]["hypothesis"]}.net")
    print(bn)


exit()








df["everstolen"] = None
df2 = df[(df["action1"] == "stealing from") | (df["action1"] == "not stealing from")]

steal_groups = df2.groupby(['run', 'subj_agent', 'object_agent'])['action1'].transform(lambda x: 'stealing from' in x.values)
print(steal_groups)
df.loc[df2.index,'everstolen'] = steal_groups

print(df["everstolen"].unique())
print(df[["run", "subj_agent", "object_agent", "indirect_agent", "action1", "action2", "everstolen"]])
exit()

df.to_csv("data/stealing_testimony_12test.csv")
dff = df


print(dff["run"].value_counts())
print(dff["action1"].value_counts(normalize=True))
print(dff["everstolen"].value_counts(normalize=True))

print(dff.groupby("run")["action1"].value_counts(normalize=True))

print(dff.groupby("run")["everstolen"].value_counts(normalize=True))

print(dff.groupby("run")["action1"].value_counts(normalize=True).groupby(level=1).mean())


df1 = dff[dff["time_testimony"] == 40]
print(df1)
exit()
df1["ReliabilityPerturbed"] = df1[["vision_perturb", "memory_perturb", "veracity_perturb"]].max(axis=1)
df1["Reliability"] = ~df1["ReliabilityPerturbed"]
print(df1)
df1 = df1[(df1["everstolen"] == True) & ((df1["action1"] == "observed") | (df1["action1"]=="not observed")) & (df1["action2"] == "stealing from")]
print(df1[["subj_agent", "action1", "obj_agent_prof", "action2", "testified_agent", "Reliability"]])
x = df1["obj_agent_prof"] == df1["testified_agent"]
print(x)


print(dff.groupby("everstolen")["everstolen"].value_counts(normalize=True))


exit()


# Select rows that do not match the condition (contrast rows)

# de kans op stelen is, voor de gegeven parameters,
# als je kijkt naar alle mogelijke daders is het 0.0524.
# maar als je kijkt naar "werd er gestolen van x?"

# de kans dat 1 gegeven dader van 1 gegeven slachtoffer steelt is 0.000524
# als in: 0.0524*0.1*0.1, namelijk kans op stelen in simulatie, dan kies een random dader, en kies een random slachtoffer
# individuen kunnen daarvan afwijken. Maar dus de kans H moet 0.00052 zijn


# maar nu de situatie: gegeven dat de hypothese niet waar is. Wil je dan de kans op: we weten dat er is gestolen?
# of ook al die andere situaties meerekenen (namelijk 0 steelt niet van 1, wat is de kans dat 2 zegt dat 0 van 1 steelt?
# okay maar eigenlijk <1,1,1> steelt niet van 1, wat is de kans dat 2 zegt dat <1,1,1> van 1 steelt?
# er is een kans dat 2 dit zegt, maar alleen in situaties waar al gestolen word (namelijk situaties waar ze het verkeerd heeft gezien/gelogen/etc
# impliciet lijkt dit samen te vallen.. met dat er gestolen word. Maar dat is in de echte wereld dus niet noodzakelijkerwijs zo

print()
exit()


dff1 = dff[(dff["action1"] == "stealing from")]

print(dff1["subj_agent"].value_counts()/len(dff1))
print((dff1["subj_agent"].value_counts()/len(dff1)).sum())
print(dff1["object_agent"].value_counts()/len(dff1))
print((dff1["object_agent"].value_counts()/len(dff1)).sum())


dff1 = dff[(dff["action1"] == "not stealing from")]

print(dff1["subj_agent"].value_counts()/len(dff1))
print((dff1["subj_agent"].value_counts()/len(dff1)).sum())
print(dff1["object_agent"].value_counts()/len(dff1))
print((dff1["object_agent"].value_counts()/len(dff1)).sum())

exit()


# Concatenate the filtered rows from df_new to df_original



df12.to_csv("data/stealing_testimony_12.csv")

exit()
df1 = df[(df["action1"] == "not stealing from") | (df["action1"] == "stealing from")]
print(df1["action1"].value_counts())

df11 = df[df["action1"] == "stealing from"]
df11.to_csv("data/stealing_testimony_2.csv")

print(df11)

df2 = df[((df["action1"] == "not observed") | (df["action1"] == "observed"))]
#print(df2)

print(df2["action1"].value_counts())
print(df2["action2"].value_counts())
exit()

df["ReliabilityPerturbed"] = df[["vision_perturb", "memory_perturb", "veracity_perturb"]].max(axis=1)
df["Reliability"] = ~df["ReliabilityPerturbed"]
df["Hypothesis"] = df[["testified_thief","victim"]].astype(str).agg(' stole from '.join, axis=1)
df["Report"] = df[["testified_thief","victim"]].astype(str).agg(' stole from '.join, axis=1)
df["ValHypothesis"] = df["real_thief"] == df["testified_thief"]

hb_df = df[['Hypothesis', "ValHypothesis", "Report", "Reliability"]].copy()

print(hb_df["ValHypothesis"].value_counts()/len(hb_df))
print(hb_df["Reliability"].value_counts()/len(hb_df))
print(hb_df.groupby(["ValHypothesis", "Reliability"]).size()/len(hb_df))




'''
hyp_rep = df[["Hypothesis", "Report"]].drop_duplicates()
for _, row in hyp_rep.iterrows():
    # Filter DataFrame for the current combination
    filtered_df = hb_df[(hb_df['Hypothesis'] == row['Hypothesis']) & (hb_df['Report'] == row['Report'])]

    # Display the result for the current combination
    print(f"Filtered rows for Hypothesis={row['Hypothesis']} and report={row['Report']}:\n")
    print(filtered_df, "\n")
'''

hb_df.to_csv("data/hb_data.csv", index=False)
