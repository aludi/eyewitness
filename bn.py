from copy import deepcopy
import pandas as pd
import pyAgrum as gum
import pyAgrum.lib.image as gumimage
import itertools as iter
import copy
import re
import os, shutil
import time


df = pd.read_csv("data/stealing_testimony.csv")

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
