from fontTools.misc.bezierTools import epsilon

from world.server import server
from world.model import ForestFire
import pandas as pd
import itertools
import ast
import numpy as np
import csv
import re
import os
import json
from bn import make_bn, bn_inference, bn_inference_collective
from plotting import plot
import math
import logging

class Experiment():

    def __init__(self):
        self.num_runs = 5000
        self.num_agents = 8
        self.simulation_time = 40
        self.simulation_data = None
        self.final_model = None
        self.bn_types = ["HB", "F", "T", "H"]

    ''' 
    DATA COLLECTION AND RUNNING MODEL
    COLLECTING THE DATA BY RUNNING THE MODEL. CONTAINS FUNCTIONS COLLECT_DATA AND RUN_MODEL.
    WRITES SIMULATION DATA TO DATA/DATACOLLECTOR
    '''

    def collect_data(self, runs):
        open('experiment.log', 'w').close()# DELEte old funs
        logging.basicConfig(
            filename='experiment.log',  # Log file path
            level=logging.INFO,  # Log level
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.log = logging.getLogger(__name__)
        self.log.info("Logger is working")

        df_list = []
        for i in range(0, runs):
            print("runs", i)
            self.log.info(f"Run number : {i}")

            model = self.run_model()
            self.final_model = model
            df = self.create_dataframe(model)
            df["run"] = i
            df_list.append(df)
        df = pd.concat(df_list)
        self.simulation_data = df
        df.to_csv("data/datacollector.csv", index=True)
        print(df.shape)

    def run_model(self):
        #print(self.num_agents)
        model = ForestFire(30, 30, 0.00001, num_agents=self.num_agents, log=self.log)
        for i in range(0, self.simulation_time+1):
            model.step()
        return model

    def create_dataframe(self, model):
        dict = model.reporters
        l_store = []
        for key in dict.keys():
            arr = dict[key]
            key_mini_df = ["Testimony", key]
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    for k in range(arr.shape[2]):
                        l = [dict["Testimony"][i, j, k], dict[key][i, j, k]]
                        key_mini_df.append(l)
            columns = key_mini_df[0:2]
            rows = key_mini_df[2:]
            df = pd.DataFrame(rows, columns=columns)
            if key not in ["Hyp", "Testimony"]:
                df[columns[1]] = df[columns[1]].astype(bool)
            l_store.append(df)
        merged_df = l_store[0].set_index("Testimony")
        for df in l_store[1:]:
            merged_df = merged_df.join(df.set_index("Testimony"))
        merged_df = merged_df.reset_index()
        return merged_df


    ''' CALCULATING GROUND TRUTH FREQUENCIES
    HERE WE CALCULATE THE GROUND TRUTH FREQUENIES.
    - GET_GROUND_TRUTH : BOOKKEEPING
    - GET FREQUENCY OUTCOMES : GET A PRIORI PROBABILITY OF STEAL HYPOTHESIS FOR ALL AGENTS
    - GET FREQUENCY OUTCOMES GIVEN OBSERVATION : GET A POSTERIORI PROBABILITY GIVEN TESTIMONY=TRUE
    - GET FREQUENCY OUTCOMES GIVEN NO OBSERVATION : GET A POSTERIOR PROBABILITY GIVEN TESTIMONY=FALSE
    '''

    def get_ground_truth(self):
        # get posteriors with evidence setting in ground truth
        # prepare data for bayesian networks
        # in both individual and collective setting

        df = pd.read_csv("data/datacollector.csv")

        self.calculate_joint(df)
        self.posterior_frequencies(df, "collective")
        self.prepare_bn_data(df, "collective")

        hyp_tests = df["Testimony"].unique().tolist()
        for hyp in hyp_tests:
            subset = df[df["Testimony"] == hyp]  # length should be num of runs.
            self.posterior_frequencies(subset, hyp)
            self.prepare_bn_data(subset, hyp)
            self.calculate_joint(subset)

    def calculate_joint(self, df):
        s = []
        cols = ["vHyp", "vTestimony", "vseen_stealing", "vreliability_vision", "vobjective_interpretation",
                "vreliability_objective", "vveracity", "vreliability_veracity"]
        for col_name in cols:
            s.append(df[col_name])

        joint_counts = df.groupby(cols).size()
        joint_probabilities = joint_counts / joint_counts.sum()
        joint_probabilities_df = joint_probabilities.reset_index(name='probability')
        if df["Testimony"].nunique() !=1:
            joint_probabilities_df.to_csv("data/jointprobs/gt.csv")
        else:
            t = df["Testimony"].unique()[0]
            joint_probabilities_df.to_csv(f"data/jointprobs/individual/gt{t}.csv")

        # for individual hypotheses.





    def posterior_frequencies(self, df, collective):
        if collective == "collective":
            hyp = ""
        else:
            hyp = collective
            collective = "gt"

        # we're not going over all cols, only the ones we could "observe" somehow??
        cols = ["vTestimony", "vreliability_vision","vreliability_objective", "vreliability_veracity"]
        values = [True, False, 'unspecified']
        # Generate all combinations of True, False, and 'unspecified' for each column
        combinations = list(itertools.product(values, repeat=len(cols)))
        combinations_dict = []
        for combo in combinations:
            # Create a dictionary for each combination
            d = {cols[i]: combo[i] for i in range(len(cols)) if combo[i] != 'unspecified'}
            combinations_dict.append(d)

        posteriors_list = [["Evidence", "FTrue", "FFalse"]]
        for d in combinations_dict:
            subset = df
            for k in d.keys():
                subset = subset[subset[k] == d[k]]
            val_c = subset["vHyp"].value_counts(normalize=True)
            posteriors_list.append([d, val_c.get(True, 0), val_c.get(False, 0)])

        with open(f"data/results/{collective}/{hyp}-gt.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerows(posteriors_list)

        # convert evidence names post-hoc
        df = pd.read_csv(f"data/results/{collective}/{hyp}-gt.csv")

        df["Evidence"] = df["Evidence"].apply(lambda x: self.rename_keys(ast.literal_eval(x)) if isinstance(x, str) else x)
        print(df["Evidence"])
        df.to_csv(f"data/results/{collective}/{hyp}-gt.csv")


    def rename_keys(self, d):
        key_map = {
            'vTestimony': "Testimony",
            "vreliability_vision": "vision_observationReliability",
            "vreliability_objective": "objectivityReliability",
            "vreliability_veracity": "veracityReliability"
        }

        if isinstance(d, dict):
            return {key_map.get(k, k): v for k, v in d.items()}
        return d



    def prepare_bn_data(self, df, collective):
        df1 = df.copy()

        hyp_testim = collective

        for bn in self.bn_types:

            if collective != "collective" and collective != "combinedDFs":
                hyp = hyp_testim
                collective = f"bndata/{bn.lower()}"
            else:
                hyp = ""
                collective = "combinedDFs"


            df1["Hypothesis"] = df["vHyp"]
            df1["Testimony"] = df["vTestimony"]
            df1["vision_observationReliability"] = df["vreliability_vision"]
            df1["objectivityReliability"] = df["vreliability_objective"]
            df1["veracityReliability"] = df["vreliability_veracity"]

            df1["testimony_seen"] = df["vseen_stealing"]
            df1["testimony_objectivity"] = df["vobjective_interpretation"]
            df1["testimony_veracity"] = df["vveracity"]


            if bn == "HB" or bn == "F":
                df1["Reliable"] = df[["vreliability_vision","vreliability_objective", "vreliability_veracity"]].min(axis=1)
                if bn == "HB":
                    vars = ["Hypothesis", "Testimony", "Reliable"]
                elif bn == "F":
                    vars = ["Hypothesis", "Testimony", "Reliable","vision_observationReliability",
                            "objectivityReliability", "veracityReliability"]
            elif bn == "T":
                vars = ["Hypothesis", "Testimony",
                        "testimony_seen", "testimony_objectivity",
                        "vision_observationReliability", "objectivityReliability", "veracityReliability"]

            elif bn == "H":
                vars = ["Hypothesis", "Testimony",
                "testimony_seen", "testimony_objectivity", "testimony_veracity",
                "vision_observationReliability", "objectivityReliability", "veracityReliability"]

            print(collective, bn, hyp)
            print(f"data/{collective}/{bn.lower()}{hyp}.csv")
            df1[vars].to_csv(f"data/{collective}/{bn.lower()}{hyp}.csv", index=False)


    '''
    PREPROCESSING AND SENSE-MAKING: TURNING THE DATA INTO RELEVANT INFORMATION
    GENERATE THE POSSIBLE HYPOTHESES
    GENERATE THE RELEVANT VARIABLES
    DEFINING THE VALUES FOR ALL VARIABLES PER BN
    '''



    def generate_hypotheses(self):
        # generate all possible combinations: <1,1,1> STOLE FROM 0, <111> STOLE FROM 2, <111> STOLE FROM 3, etc.
        if self.final_model is not None:
            profiles = self.final_model.prof
        else:
            prof_opportunities = list(itertools.product(range(2), repeat=3))
            for i in range(2, 10):
                if len(list(itertools.product(range(i), repeat=3))) >= self.num_agents:
                    prof_opportunities = list(itertools.product(range(i), repeat=3))
                    break
            profiles = prof_opportunities
        hypotheses = []
        for i in range(0, self.num_agents):
            for j in range(0, self.num_agents):
                if i != j:  # an agent cannot steal from itself
                    hypotheses.append((profiles[i], "STEAL", j))
        return hypotheses

    def generate_bn_variables(self,witness_agent, h, bn):
        variables = {}
        if bn == "HB":
            variables["stealhyp"] = h
            variables["testifies"] = (witness_agent, "TESTIFIES", h)
            variables["reliability"] = (witness_agent, "RELIABILITY", h)

        elif bn == "F":
            variables["stealhyp"] = h
            variables["testifies"] = (witness_agent, "TESTIFIES", h)
            variables["reliability"] = (witness_agent, "RELIABILITY", h)
            variables["visiongood"] = (witness_agent, "VISION", h)
            variables["objectivegood"] = (witness_agent, "OBJECTIVITY", h)
            variables['veracitygood'] = (witness_agent, "VERACITY", h)

        elif bn == "T":
            variables["stealhyp"] = h
            variables["testifies"] = (witness_agent, "TESTIFIES", h)
            variables["testimony_seen"] = (witness_agent, "SEES", h)
            variables["testimony_objectivity"] = (witness_agent, "OBJECTIVE", h)
            variables['veracitygood'] = (witness_agent, "VERACITY", h)

        elif bn == "H":
            variables["stealhyp"] = h
            variables["testimony_seen"] = (witness_agent, "SEES", h)
            variables["observed"] = (witness_agent, "TESTIFIES", h)
            variables["testifies"] = (witness_agent, "TESTIFIES", h)
            variables["testimony_objectivity"] = (witness_agent, "OBJECTIVE", h)
            variables["testimony_veracity"] = (witness_agent, "VERACITY", h)
            variables["visiongood"] = (witness_agent, "VISION", h)
            variables["objectivegood"] = (witness_agent, "OBJECTIVITY", h)
            variables['veracitygood'] = (witness_agent, "VERACITY", h)
        else:
            print("not implemented")

        return variables

    def get_events_from_df(self, witness_agent, df, variables, bn):
        final_step = df["Step"].max()
        df1 = df[(df["Step"] == final_step) & (df["id"] == witness_agent)] # final step of of the dataframe
        df1 = df1.copy()
        # evaluating the hypothesis "((1, 0, 0), STEAL, 8)" as stored in variables["stealhyp"]
        df1["stealEvents"] = df1['stealEvents'].apply(ast.literal_eval) # evaluate not as string but as list of tuples
        df1[variables["stealhyp"]] = df1['stealEvents'].apply(
            lambda x: any(len(t) >= 4 and variables["stealhyp"] == (t[1], t[2], t[3]) for t in x))

        df1["Hypothesis"] = df1[variables["stealhyp"]]

        # evaluating the testimony "(1 "Observed" (1, 0, 0), STEAL, 8))" as stored in variables["testimony"]
        # this depends on the implementation per bn
        if bn == "HB" or bn == "F":
            df1 = self.get_testimony(variables, df1)
            df1 = self.determine_permutation_observations(variables, df1)
            df1['Reliable'] = df1[['vision_observationReliability','objectivityReliability', 'veracityReliability']].min(axis=1)
            df1["Testimony"] = df1[variables["testifies"]]

            if bn == "HB":
                df1[["Hypothesis", "Testimony", "Reliable"]].to_csv(f"data/bndata/hb/{variables["testifies"]}.csv", index=False)
            elif bn == "F":
                df1[["Hypothesis", "Testimony", "Reliable",
                     "vision_observationReliability", "objectivityReliability", "veracityReliability"]].to_csv(f"data/bndata/f/{variables["testifies"]}.csv", index=False)
            else:
                print("not building implemented in line 238")
                exit()

        elif bn == "T":
            #print(witness_agent, variables["stealhyp"])

            df1 = self.is_hyp_agent_in_obs_list(df1, variables, "vision_observation", "testimony_seen")
            df1 = self.is_hyp_agent_in_obs_list(df1, variables, "objectivity", "testimony_objectivity")
            df1 = self.is_hyp_agent_in_obs_list(df1, variables, "veracity", "testifies")

            df1 = self.determine_permutation_observations(variables, df1)
            df1["Testimony"] = df1["testifies"]

            df1[["Hypothesis", "Testimony",
                 "testimony_seen","testimony_objectivity",
                 "vision_observationReliability", "objectivityReliability", "veracityReliability"]].to_csv(
                f"data/bndata/t/CHECK{variables["testifies"]}.csv",index=False)

        elif bn == "H":
            df1 = self.is_hyp_agent_in_obs_list(df1, variables, "vision_observation", "testimony_seen")
            df1 = self.is_hyp_agent_in_obs_list(df1, variables, "objectivity", "testimony_objectivity")

            df1["testimony_veracity"] = df1['veracity'].apply(ast.literal_eval)  # evaluate not as string but as list of tuples
            v = variables["testimony_veracity"]
            (a, b, (c, d, e)) = v   #0 VERACITY (0, 0, 0) STEAL 1

            df1["testifies"] = df1['testimony_veracity'].apply(     # there is a testimony about the stealing event that is targeting the agent e
                lambda x: any(len(t) >= 4 and (str(d), e) == (t[2], t[3]) for t in x)
            )
            #df1["testifies"] = df1['testimony_veracity'].apply(lambda x: x != [])   # TODO: now I just check if the agent does not testify, but specifically it should not testify about the specific stealing

            df1["originalVeracity"] = df1["testimony_veracity"]
            df1["testimony_veracity"] = df1['testimony_veracity'].apply(    # is the testimony about the stealing event where c targeting agent e? -> can this take the value of none?
                lambda x: any(len(t) >= 4 and (c, str(d), e) == (t[1], t[2], t[3]) for t in x)
            )
            df1 = self.determine_permutation_observations(variables, df1)
            '''
            now it will also be false if is is None (no testimony) as well as if it's about someone else?
            '''
            df1['testimony_veracity'] = df1['testimony_veracity']
            df1["Testimony"] = df1["testifies"]
            df1[["Hypothesis", "Testimony",
                 "testimony_seen", "testimony_objectivity",
                 "vision_observationReliability", "objectivityReliability", "veracityReliability", "testimony_veracity"]].to_csv(
                f"data/bndata/h/{variables["testifies"]}.csv",index=False)
        else:
            print("not implemented")



    def is_hyp_agent_in_obs_list(self, df1, variables, var_list_col, var_node_name):
        '''
        This is the main pattern of the code. For each column that contains a list
        of observations (so: seen, objective, veracity, etc), we need to check
        if the hypothesised agent under consideration (with the correct victim) occurs in this list.
        If it does, we write true, if it does not, we write false.
        '''
        df1[var_node_name] = df1[var_list_col].apply(
            ast.literal_eval)  # evaluate not as string but as list of tuples
        v = variables[var_node_name]
        #print(df1[["run", "step", var_node_name]])
        d1 = df1[df1["run"] == 434]
        pd.set_option('display.max_colwidth', None)
        #print(d1[[var_node_name, var_list_col]])
        #exit()
        (a, b, (c, d, e)) = v
        df1[f"original{var_list_col}"] = df1[var_list_col]
        df1[var_node_name] = df1[var_node_name].apply(
            lambda x: any(len(t) >= 4 and (c, str(d), e) == (t[1], t[2], t[3]) for t in x)
        )
        print(df1[var_node_name])
        print(df1[var_node_name].value_counts())
        return df1

    def get_testimony(self, variables, df1):
        '''
        Function for determining what the agent testifies. Currently,
        we extract the data from the "veracity" column, which is what
        the agent testifies after all permutations have taken place.
        The meaning:
        Testimony = True: the witness agent saw the agent in the hypothesis steal from the victim
        Testimony False: the witness agent did not see the hypothesis agent steal, but another agent, or
        the agent did not see anyone steal from the victim.
        '''

        df1["testifies"] = df1['veracity'].apply(
            ast.literal_eval)  # evaluate not as string but as list of tuples
        v = variables["testifies"]
        (a, b, (c, d, e)) = v
        df1["originalTestifies"] = df1["testifies"]

        NO = "False"  # "None" #False # False or NT

        df1[variables["testifies"]] = df1['testifies'].apply(
            lambda x: NO if x == [] else NO if all(len(t) >= 4 and e != (t[3]) for t in x) else any(
                len(t) >= 4 and (c, str(d), e) == (t[1], t[2], t[3]) for t in x)
        )
        return df1

    def permutation_function(self, df1, col_var_name, variables):
        '''
        col_var_name can take the values of {vision_objectivity, objectivity, veracity}
        change the variables to check whether the agent was seen
        '''
        (p, s, vic) = variables["stealhyp"]
        NO = "True"
        df1[f"{col_var_name}_permuted_tl"] = df1[f"{col_var_name}_permuted"].apply(
            ast.literal_eval)  # evaluate not as string but as list of tuples
        df1[f"temp{col_var_name}"] = df1[f"{col_var_name}_permuted_tl"].apply(
            lambda x: [(t[0][3], t[1]) for t in x])  # select relevant info
        df1[f"{col_var_name}Perturb"] = df1[f"temp{col_var_name}"].apply(
            lambda x: next((t[1] for t in x if t[0] == vic), NO))  # match on victim id
        df1[f'{col_var_name}Reliability'] = df1[f"{col_var_name}Perturb"].apply(
            lambda x: not x if x != "None" else "None")  # not perturbed = reliable
        return df1

    def determine_permutation_observations(self, variables, df1):
        '''
        We have three ways that observations can be changed: the agent can have bad vision,
        the agent can have low objectivity, and the agent can choose to lie about what they saw.
        In this function we check all the columns whether in each of these columns there
        was a change/permutation/disturbance in the observation with respect to the
        relevant event (the hypothesis). So we check if, if the hypothesis is Stole(000, 1),
        with witness agent 2, if witness agent 2 was reliable in observing the hypothesis.
        '''
        for rel_col in ["vision_observation", "objectivity", "veracity"]:
            df1 = self.permutation_function(df1, rel_col, variables)
        return df1


    '''
    BAYESIAN NETWORK CONSTRUCTION AND INFERENCE
    '''
    def make_bns(self):
        self.individual_bn_creation()
        self.collective_bn()

    def enumerate_hypotheses(self):
        list_l = []
        if self.final_model is not None:
            profiles = self.final_model.prof
        else:
            prof_opportunities = list(itertools.product(range(2), repeat=3))
            for i in range(2, 10):
                if len(list(itertools.product(range(i), repeat=3))) >= self.num_agents:
                    prof_opportunities = list(itertools.product(range(i), repeat=3))
                    break
            profiles = prof_opportunities

        for j in range(0, self.num_agents):
            for k in range(0, self.num_agents):
                for prof in profiles:
                    s = f"{k}T{prof}S{j}"
                    list_l.append(s)
        return list_l


    def individual_bn_creation(self):
        #hypotheses = self.generate_hypotheses()
        hypotheses = self.enumerate_hypotheses()
        for hypothesis in hypotheses:
            for bn in self.bn_types:
                variables = hypothesis
                data = f"{bn.lower()}{hypothesis}"
                df = pd.read_csv(f"data/bndata/{bn.lower()}/{data}.csv")
                make_bn(df, bn, variables)

    def collective_bn(self):
        for bn in self.bn_types:
            variables = "collective"
            df = pd.read_csv(f"data/combinedDFs/{bn.lower()}.csv")
            make_bn(df, bn, variables)

    def bns_inference(self):
        bn_inference(self.bn_types)
        #bn_inference_collective(self.bn_types)

    def compare_joints(self):
        self.compare_joint("", "")
        for h in self.enumerate_hypotheses():
            self.compare_joint(f"individual/", h)



    def compare_joint(self,var1, var2):
        # individual and collective
        # get individual gt... subset on relevant hypothesis

        gt_joint = pd.read_csv(f"data/jointprobs/{var1}gt{var2}.csv")
        list_df = []
        # rename the columns of the GT dataframe
        rename_heads = {"vHyp":"Hypothesis",
         "vTestimony":"Testimony",
         "vseen_stealing":"testimony_seen",
         "vreliability_vision":"vision_observationReliability",
        "vobjective_interpretation":"testimony_objectivity",
         "vreliability_objective":"objectivityReliability",
         "vveracity":"testimony_veracity",
         "vreliability_veracity":"veracityReliability"}
        gt_joint = gt_joint.rename(columns=rename_heads)
        gt_joint["Reliable"] = (gt_joint["veracityReliability"] & gt_joint["objectivityReliability"] & gt_joint["vision_observationReliability"])

        gt_joint = gt_joint.loc[:, ~gt_joint.columns.str.contains("^Unnamed")]

        print(gt_joint.info())

        for bn in self.bn_types:
            bn_joint = pd.read_csv(f"data/jointprobs/{var1}{bn.lower()}{var2}.csv")
            bn_joint = bn_joint.loc[:, ~bn_joint.columns.str.contains("^Unnamed")]

            rel_cols_bn = bn_joint.columns
            cols_gt = gt_joint.columns
            cols_to_sum_out = list(set(cols_gt) - set(rel_cols_bn))

            # Get all other variable columns except the one to sum out and the probability column
            group_cols = [col for col in gt_joint.columns if col not in cols_to_sum_out +["probability"]]

            # Group by remaining variables and sum probabilities
            df_marginalized = gt_joint.groupby(group_cols, as_index=False)["probability"].sum()

            print(bn_joint.info())




            cols = list(bn_joint.columns)
            cols.remove("probability")

            for key in cols:
                assert key in bn_joint.columns, f"{key} not in df1"
                assert key in df_marginalized.columns, f"{key} not in df2"

            collapse_cols = cols

            # Create a new column with the label-value pairs
            bn_joint["Evidence"] = bn_joint[collapse_cols].apply(
                lambda row: ", ".join(f"{col}{val}" for col, val in row.items()), axis=1
            )

            df_marginalized["Evidence"] = df_marginalized[collapse_cols].apply(
                lambda row: ", ".join(f"{col}{val}" for col, val in row.items()), axis=1
            )

            bn_joint = bn_joint[["Evidence", "probability"]]
            df_marginalized = df_marginalized[["Evidence", "probability"]]

            merged = pd.merge(bn_joint, df_marginalized, on="Evidence", how="outer")
            rename_heads = {"probability_x":"probability", "probability_y":"GTF"}

            merged = merged.rename(columns=rename_heads)
            # Fill missing values with 0
            merged_filled = merged.fillna(0)
            #print(merged_filled)
            merged_filled["BN"] = bn

            # calcualte KL divergence over the joint

            epsilon = 1e-10

            print(var2)

            merged_filled["KL-val"] = merged_filled.apply(
                lambda row: row["GTF"] * math.log2((row["GTF"] + epsilon) / (row["probability"] + epsilon))
                if row["probability"] > 0 else 0, axis=1
            )
            print(f"KL-divergence {bn} : ", merged_filled["KL-val"].sum())

            merged_filled["difference"] = merged_filled.apply(
                lambda row: (row["probability"]-row["GTF"])**2, axis=1
            )
            print(f"difference {bn} : ", merged_filled["difference"].sum()/merged_filled.shape[0])

            merged_filled["Hypothesis"] = var2
            list_df.append(merged_filled)



            # it seems like the only conclusion we can draw from this
            # is that the distributions are different from the ground truth, and
            # we cannot say how bad that is...

        df = pd.concat(list_df)
        df.to_csv(f"data/results/joints/{var1}jointDFs{var2}.csv")


    def calculate_differences(self):
        self.calculate_differences_individual()
        self.calculate_differences_collective()
        pass

    def calculate_differences_collective(self):
        gt_df = f"data/results/collective/-gt.csv"
        bn_df = f"data/results/collective/outcomes.csv"
        gt_df = pd.read_csv(gt_df)
        bn_df = pd.read_csv(bn_df)
        gt_df["Evidence"] = gt_df["Evidence"]
        gt_df["FTrue"] = gt_df["FTrue"]
        gt_df["FFalse"] = gt_df["FFalse"]
        g_df = gt_df[["Evidence", "FTrue", "FFalse"]]

        g_df.loc[:,"Evidence"] = g_df["Evidence"].apply(self.normalize_dict_string)
        bn_df.loc[:,"Evidence"] = bn_df["Evidence"].apply(self.normalize_dict_string)

        #print(g_df)
        #exit()
        pd.set_option('display.max_columns', None)
        merged = pd.merge(bn_df, g_df, on='Evidence')
        merged["DT"] = abs(merged["FTrue"] -merged["PTrue"])
        merged["DF"] = abs(merged["FFalse"] -merged["PFalse"])
        merged.to_csv(f"data/results/collective/difference.csv")


    def get_ground_truth_collective(self):
        folder_path = f"data/results/gt"
        all_outcomes = sorted(os.listdir(folder_path))
        collected_dfs = []
        for file in all_outcomes:
            fp = f"{folder_path}/{file}"
            df = pd.read_csv(fp)
            df["FreqStealTrue"] = df["PTrue"]*df["total"]
            df["FreqStealFalse"] = df["PFalse"]*df["total"]
            ev = file.split(".csv")[0]
            df1 = df[["FreqStealTrue","FreqStealFalse"]]
            #print(ev, "\n", df1.sum())
            df1 = pd.DataFrame([df1.sum()])
            df1["ev"] = ev
            df1["sumEvents"] = df1["FreqStealTrue"] + df1["FreqStealFalse"]
            df1["FStealT"] = df1["FreqStealTrue"]/df1["sumEvents"]
            df1["FStealF"] = df1["FreqStealFalse"]/df1["sumEvents"]
            #print(df1)
            df2 = df1[["ev", "FStealT", "FStealF"]]
            print(df2)
            #print("\n\n\n\n")
            collected_dfs.append(df2)

        df = pd.concat(collected_dfs)
        print(df)
        df.to_csv(f"data/results/collective/gt.csv")

    def normalize_dict_string(self, s):
        d = ast.literal_eval(s)  # safely convert string to dict
        #print(json.dumps(d, sort_keys=True))
        for k in d.keys():
            d[k] = str(d[k])
        return json.dumps(d, sort_keys=True)  # convert dict to JSON string with sorted keys


    def calculate_differences_individual(self):
        df_col = []
        for bn_type in self.bn_types:
            folder_path = f"data/results/{bn_type.lower()}"
            # Loop through all files and directories in the folder
            for filename in os.listdir(folder_path):
                hypothesis = filename.removeprefix(bn_type.lower())
                gt_s = f"{hypothesis.split(".csv")[0]}-gt.csv"
                file_path = os.path.join(folder_path, filename)
                df_GT = pd.read_csv(f"data/results/gt/{gt_s}")

                try:
                    df_BN = pd.read_csv(file_path)
                except Exception as e:
                    df_BN = df_GT.copy()
                    df_BN["PTrue"] = 0
                    df_BN["PFalse"] = 0

                df_GT[f"PTrueGT"] = df_BN["PTrue"]
                df_GT[f"PFalseGT"] = df_BN["PFalse"]
                d_GT = df_GT[["Evidence", "PTrueGT", "PFalseGT"]]

                #print(d_GT["Evidence"])
                d_GT.loc[:,"Evidence"] = d_GT["Evidence"].apply(self.normalize_dict_string)
                df_BN.loc[:,"Evidence"] = df_BN["Evidence"].apply(self.normalize_dict_string)

                if not set(df_BN["Evidence"]).issubset(set(d_GT["Evidence"])):
                    print("confuse")
                    print(set(df_BN["Evidence"]) - set(d_GT["Evidence"]))
                    print(bn_type, hypothesis)
                results = pd.merge(d_GT, df_BN,on="Evidence")
                results["Hypothesis"] = hypothesis
                results["BN"] = bn_type
                results["DPTrue"] = abs(results["PTrueGT"]-results["PTrue"])
                results["DPTFalse"] = abs(results["PFalseGT"]-results["PFalse"])
                df_col.append(results)
        all_results = pd.concat(df_col)
        all_results.to_csv("data/results/difference/allresults.csv")


def run_visual():
    # this calls "server.py" and launches it
    server.launch(open_browser=True)

def run_experiment():

    e = Experiment()
    print("running simulation")
    #e.collect_data(e.num_runs)
    print("preprocessing data")
    #e.get_ground_truth()
    print("creating bns")
    #e.make_bns()
    print("compare the joint probs of the BNs with the GT")
    #e.compare_joints()
    print("calculating posteriors")
    #e.bns_inference()
    print("calculating differences")
    #e.calculate_differences()
    print("plotting outcomes")
    plot()


run_experiment()
#run_visual()


