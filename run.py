from world.server import server
from world.model import ForestFire
import pandas as pd
import itertools
import ast
import numpy as np
import csv
import re
import os
from bn import make_bn, bn_inference, bn_inference_collective
from plotting import plot

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
        df_list = []
        for i in range(0, runs):
            print("runs", i)
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
        model = ForestFire(30, 30, 0.00001, num_agents=self.num_agents)
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


    def calculate_joint(self, df):
        s = []
        cols = ["vHyp", "vTestimony", "vseen_stealing", "vreliability_vision", "vobjective_interpretation",
                "vreliability_objective", "vveracity", "vreliability_veracity"]
        for col_name in cols:
            s.append(df[col_name])
        joint_counts = df.groupby(cols).size()
        joint_probabilities = joint_counts / joint_counts.sum()
        joint_probabilities_df = joint_probabilities.reset_index(name='probability')
        joint_probabilities_df.to_csv("data/jointprobs/gt.csv")

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
        hypotheses = self.enumerate_hypotheses()
        for hypothesis in hypotheses:
            bn_inference(self.bn_types, hypothesis)
        bn_inference_collective(self.bn_types)


    def calculate_differences(self):
        #self.calculate_differences_individual()
        self.calculate_differences_collective()
        pass

    def calculate_differences_collective(self):
        gt_df = f"data/results/collective/gt.csv"
        bn_df = f"data/results/collective/outcomes.csv"
        gt_df = pd.read_csv(gt_df)
        bn_df = pd.read_csv(bn_df)
        gt_df["Evidence"] = gt_df["ev"]
        gt_df["FTrue"] = gt_df["FStealT"]
        gt_df["FFalse"] = gt_df["FStealF"]
        g_df = gt_df[["Evidence", "FTrue", "FFalse"]]
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



    def calculate_differences_individual(self):
        df_col = []
        for bn_type in self.bn_types:
            #folder_path = f"bns/{bn_type.lower()}"
            # Loop through all files and directories in the folder
            if bn_type == "HB":
                e_all_true = {"Testimony": "True", "Reliable": "True"}
                e_unreliable1 = {"Testimony": "True", "Reliable": "False"}
                e_unreliable2 = {"Testimony": "True", "Reliable": "False"}
                e_unreliable3 = {"Testimony": "True", "Reliable": "False"}
                e_unreliable4 = {"Testimony": "True", "Reliable": "False"}

            else:
                e_all_true = {"Testimony": "True", "vision_observationReliability": "True",
                              "objectivityReliability": "True", "veracityReliability": "True"}
                e_unreliable1 = {"Testimony": "True", "vision_observationReliability": "False",
                                 "objectivityReliability": "False", "veracityReliability": "False"}
                e_unreliable2 = {"Testimony": "True", "vision_observationReliability": "False",
                                 "objectivityReliability": "True", "veracityReliability": "True"}
                e_unreliable3 = {"Testimony": "True", "vision_observationReliability": "True",
                                 "objectivityReliability": "False", "veracityReliability": "True"}
                e_unreliable4 = {"Testimony": "True", "vision_observationReliability": "True",
                                 "objectivityReliability": "True", "veracityReliability": "False"}
            for evidence in [{}, {"Testimony": "True"}, {"Testimony": "False"}, e_all_true, e_unreliable1,
                             e_unreliable2, e_unreliable3, e_unreliable4]:
                #print(evidence)
                folder_path = f"data/results/{bn_type.lower()}/{evidence}.csv"
                #print(folder_path)

                df_BN = pd.read_csv(folder_path)
                if bn_type == "HB":
                    if evidence == e_all_true:
                        e = {"Testimony": "True", "vision_observationReliability": "True",
                                      "objectivityReliability": "True", "veracityReliability": "True"}
                    else:
                        e = {"Testimony": "True", "vision_observationReliability": "True",
                             "objectivityReliability": "True", "veracityReliability": "False"}
                    df_GT = pd.read_csv(f"data/results/gt/{e}.csv")
                else:
                    df_GT = pd.read_csv(f"data/results/gt/{evidence}.csv")

                df_BN['hyp'] = df_BN['Hypothesis'].str.findall(r'(\d+)')
                df_GT['hyp'] = df_GT['Hypothesis'].str.findall(r'(\d+)')


                df_GT[f"PTrueGT"] = df_BN["PTrue"]
                df_GT[f"PFalseGT"] = df_BN["PFalse"]

                BN_d = df_BN[["hyp", f"PTrue", f"PFalse"]]
                BN_d = BN_d[BN_d['hyp'].apply(lambda x: x != [])]
                BN_d["hyp"] = BN_d["hyp"].apply(lambda x: str(x))


                GT_d = df_GT[["hyp", f"PTrueGT", f"PFalseGT"]]
                GT_d["hyp"] = GT_d["hyp"].apply(lambda x: str(x))

                results = pd.merge(BN_d, GT_d,on="hyp")

                results["DPTrue"] = abs(results["PTrueGT"] - results[f"PTrue"])
                results["DPFalse"] = abs(results["PFalseGT"] - results[f"PFalse"])
                results["evidence"] = str(evidence)
                results["bn"] = bn_type

                print(results["evidence"])

                df_col.append(results)

                #print(bn_type, evidence)
                #print(results)
                results.to_csv(f"data/results/difference/D-{bn_type}-{evidence}.csv")

                #print(df_BN)
                #print(df_GT)
                # merge on column hyp. rename the other columns
            all_results = pd.concat(df_col)
            all_results.to_csv("data/results/difference/allresults.csv")



def run_visual():
    # this calls "server.py" and launches it
    server.launch(open_browser=True)

def run_experiment():

    e = Experiment()
    print("running simulation")
    e.collect_data(e.num_runs)
    print("preprocessing data")
    e.get_ground_truth()
    print("creating bns")
    e.make_bns()
    print("calculating posteriors")
    e.bns_inference()
    print("calculating differences")
    '''e.calculate_differences()
    print("plotting outcomes")
    plot()'''


run_experiment()
#run_visual()


