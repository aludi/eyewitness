from world.server import server
from world.model import ForestFire
import pandas as pd
import itertools
import ast
import numpy as np
import cairo
import csv
import matplotlib.pyplot as plt
import os
from bn import make_bn, bn_inference
from plotting import plot_outcomes_histogram_all

class Experiment():

    def __init__(self):
        self.num_runs = 1200
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
            #print(model.relevant_data)
            df = model.datacollector.get_agent_vars_dataframe()
            df["run"] = i
            df_list.append(df)
        df = pd.concat(df_list)
        self.simulation_data = df
        df.to_csv("data/datacollector.csv", index=True)
        print(df.shape)

    def run_model(self):
        print(self.num_agents)
        model = ForestFire(30, 30, 0.00001, num_agents=self.num_agents)
        for i in range(0, self.simulation_time+1):
            model.step()
        return model


    ''' CALCULATING GROUND TRUTH FREQUENCIES
    HERE WE CALCULATE THE GROUND TRUTH FREQUENIES.
    - GET_GROUND_TRUTH : BOOKKEEPING
    - GET FREQUENCY OUTCOMES : GET A PRIORI PROBABILITY OF STEAL HYPOTHESIS FOR ALL AGENTS
    - GET FREQUENCY OUTCOMES GIVEN OBSERVATION : GET A POSTERIORI PROBABILITY GIVEN TESTIMONY=TRUE
    - GET FREQUENCY OUTCOMES GIVEN NO OBSERVATION : GET A POSTERIOR PROBABILITY GIVEN TESTIMONY=FALSE
    '''

    def get_ground_truth(self):
        df = pd.read_csv("data/datacollector.csv")
        final_step = df["Step"].max()
        df1 = df[(df["Step"] == final_step)]
        df1 = df1.copy()

        df1["stealEvents"] = df1['stealEvents'].apply(ast.literal_eval)  # evaluate not as string but as list of tuples
        df1['ObsStealEvents'] = df1['stealEvents'].apply(lambda lst: [t[1:] for t in lst])
        df1['ObsStealEvents'] = df1['ObsStealEvents'].apply(lambda lst: [(t[0], str(t[1]), t[2]) for t in lst])

        '''self.get_frequency_outcomes(df1, evidence={})   # writes to csv
        self.get_frequency_outcomes_given_observation(df1, "SEEN")
        self.get_frequency_outcomes_given_observation(df1, "NOTSEEN")'''
        self.get_frequency_outcomes_given_observation(df1, "SEENANDRELIABLE")


    def get_frequency_outcomes(self, df1, evidence):
        hyp_dict_count = {}
        hypotheses = self.generate_hypotheses()
        max_run= df1["run"].max()
        for hypothesis in hypotheses:
            hyp_dict_count[hypothesis] = (0, 0, 0)
            for i in range(0, max_run):  # iterate over all runs
                if hypothesis in df1[df1["run"] == i]["ObsStealEvents"].iloc[0]:
                    #print("hyp true: ", hypothesis)
                    (true_count, false_count, tot) = hyp_dict_count[hypothesis]
                    hyp_dict_count[hypothesis] = (true_count + 1, false_count, tot + 1)
                else:
                    #print("hyp false: ", hypothesis)
                    (true_count, false_count, tot) = hyp_dict_count[hypothesis]
                    hyp_dict_count[hypothesis] = (true_count, false_count + 1, tot + 1)
        self.count_freq_outcomes(hyp_dict_count, evidence)

    def count_posterior(self, hypothesis, df1, i, j, hyp_dict_count):
        if hypothesis in df1[(df1["run"] == i) & (df1["id"] == j)]["ObsStealEvents"].iloc[0]:
            # print("hyp true: ", j, hypothesis)
            (true_count, false_count, tot) = hyp_dict_count[f"{j}{hypothesis}"]
            hyp_dict_count[f"{j}{hypothesis}"] = (true_count + 1, false_count, tot + 1)
        else:
            # print("hyp false: ", j, hypothesis)
            (true_count, false_count, tot) = hyp_dict_count[f"{j}{hypothesis}"]
            hyp_dict_count[f"{j}{hypothesis}"] = (true_count, false_count + 1, tot + 1)
        return hyp_dict_count

    def count_freq_outcomes(self, hyp_dict_count, evidence):
        outcomes = [["Hypothesis", "PTrue", "PFalse", "total"]]
        for h in hyp_dict_count.keys():
            (t, f, tot) = hyp_dict_count[h]
            #print(h, t, f, tot)
            if tot > 0:
                #print(h, t / tot, f / tot, tot)
                outcomes.append([h, t / tot, f / tot, tot])

        with open(f"data/results/{evidence}.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerows(outcomes)

    def match_tuple(self, row, hypothesis, colname):
        observations = row[colname].apply(ast.literal_eval).iloc[0]
        #print(observations)
        #print(hypothesis)
        for ((a, b, c, d), e) in observations:
            if (b, c, d) == hypothesis:
                if e == True:
                    #print(observations)
                    #print(hypothesis)
                    return e
        return False

    def get_frequency_outcomes_given_observation(self, df1, evidence):
        hyp_dict_count = {}
        hypotheses = self.generate_hypotheses()
        max_run= df1["run"].max()
        df1['ObsVeracity'] = df1['veracity'].apply(ast.literal_eval)
        df1['ObsVeracity'] = df1['ObsVeracity'].apply(lambda lst: [t[1:] for t in lst])
        df1['ObsVeracity'] = df1['ObsVeracity'].apply(lambda lst: [(t[0], str(t[1]), t[2]) for t in lst])
        for hypothesis in hypotheses:
            for j in range(0, self.num_agents):
                hyp_dict_count[f"{j}{hypothesis}"] = (0, 0, 0)
                for i in range(0, max_run):  # iterate over all runs
                    if evidence == "SEEN":
                        if hypothesis in df1[(df1["run"] == i)&(df1["id"]==j)]["ObsVeracity"].iloc[0]:
                            hyp_dict_count = self.count_posterior(hypothesis, df1, i, j, hyp_dict_count)
                    elif evidence == "NOTSEEN":
                        if hypothesis not in df1[(df1["run"] == i)&(df1["id"]==j)]["ObsVeracity"].iloc[0]:
                            hyp_dict_count = self.count_posterior(hypothesis, df1, i, j, hyp_dict_count)
                    elif evidence == "SEENANDRELIABLE":
                        df2 = df1[(df1["run"] == i) & (df1["id"] == j)]
                        if not self.match_tuple(df2, hypothesis, "vision_observation_permuted") and \
                                not self.match_tuple(df2, hypothesis, "objectivity_permuted") and \
                                not self.match_tuple(df2, hypothesis, "veracity_permuted"):
                            hyp_dict_count = self.count_posterior(hypothesis, df1, i, j, hyp_dict_count)


                        #df1['ObsVeracity'] = df1['ObsVeracity'].apply(lambda lst: [t[0][1:][1:] for t in lst])
                        #if hypothesis in df1[(df1["run"] == i) & (df1["id"] == j)]["ObsVeracity"].iloc[0] and
                        #    hypothesisexit()
        self.count_freq_outcomes(hyp_dict_count, evidence)


    '''
    PREPROCESSING AND SENSE-MAKING: TURNING THE DATA INTO RELEVANT INFORMATION
    GENERATE THE POSSIBLE HYPOTHESES
    GENERATE THE RELEVANT VARIABLES
    DEFINING THE VALUES FOR ALL VARIABLES PER BN
    '''

    def preprocess_data(self):
        #df = self.simulation_data
        df = pd.read_csv("data/datacollector.csv")
        hypotheses = self.generate_hypotheses()
        for hypothesis in hypotheses:
            for witness_agent in range(0, self.num_agents):
                for bn in self.bn_types:
                    variables = self.generate_bn_variables(witness_agent, hypothesis, bn)
                    self.get_events_from_df(witness_agent, df, variables, bn)


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
            df1 = self.is_hyp_agent_in_obs_list(df1, variables, "vision_observation", "testimony_seen")
            df1 = self.is_hyp_agent_in_obs_list(df1, variables, "objectivity", "testimony_objectivity")
            df1 = self.is_hyp_agent_in_obs_list(df1, variables, "veracity", "testifies")
            df1 = self.determine_permutation_observations(variables, df1)
            df1["Testimony"] = df1["testifies"]

            df1[["Hypothesis", "Testimony",
                 "testimony_seen","testimony_objectivity",
                 "vision_observationReliability", "objectivityReliability", "veracityReliability"]].to_csv(
                f"data/bndata/t/{variables["testifies"]}.csv",index=False)

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
        (a, b, (c, d, e)) = v
        df1[f"original{var_list_col}"] = df1[var_list_col]
        df1[var_node_name] = df1[var_node_name].apply(
            lambda x: any(len(t) >= 4 and (c, str(d), e) == (t[1], t[2], t[3]) for t in x)
        )
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
        hypotheses = self.generate_hypotheses()
        for hypothesis in hypotheses:
            for witness_agent in range(0, self.num_agents):
                for bn in self.bn_types:
                    variables = self.generate_bn_variables(witness_agent, hypothesis, bn)
                    df = pd.read_csv(f"data/bndata/{bn.lower()}/{variables["testifies"]}.csv")
                    make_bn(df, bn, variables)

    def bns_inference(self):
        bn_inference(self.bn_types)


def run_visual():
    # this calls "server.py" and launches it
    server.launch(open_browser=True)

def run_experiment():
    e = Experiment()
    print("running simulation")
    #e.collect_data(e.num_runs)
    print("preprocessing data")
    #e.preprocess_data()
    print("creating bns")
    #e.make_bns()
    print("calculating ground truth")
    e.get_ground_truth()
    print("calculating posteriors")
    e.bns_inference()
    print("plotting outcomes")
    plot_outcomes_histogram_all()


#run_experiment()
run_visual()


