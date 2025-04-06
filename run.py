from world.server import server
from world.model import ForestFire
import pandas as pd
import itertools
import ast
import pyAgrum as gum
import numpy as np
import pyAgrum.lib.image as gumimage
import cairo

class Experiment():

    def __init__(self):
        self.num_runs = 500
        self.num_agents = 8
        self.simulation_time = 40
        self.simulation_data = None
        self.final_model = None
        self.bn_types = ["HB", "F", "T"] #H


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
            variables["observed"] = (witness_agent, "TESTIFIES", h)
            variables["testifies"] = (witness_agent, "TESTIFIES", h)
            variables["visiongood"] = (witness_agent, "VISION", h)
            variables["objectivegood"] = (witness_agent, "OBJECTIVITY", h)
            variables['veracitygood'] = (witness_agent, "VERACITY", h)
        else:
            print("not implemented")

        return variables


    def preprocess_data(self):
        #df = self.simulation_data
        df = pd.read_csv("data/datacollector.csv")
        hypotheses = self.generate_hypotheses()
        for hypothesis in hypotheses:
            for witness_agent in range(0, self.num_agents):
                for bn in self.bn_types:
                    variables = self.generate_bn_variables(witness_agent, hypothesis, bn)
                    #print(variables)
                    self.get_events_from_df(witness_agent, df, variables, bn)

    def create_BN_from_df(self, df, struct, hypotheses):
        arcs = {}
        learner = gum.BNLearner(df)
        events = []

        if struct == "HB":
            events = ["Hypothesis", "Testimony", "Reliable"]
            arcs["mandatory"] = [("Hypothesis", "Testimony"), ("Reliable", "Testimony")]

        elif struct == "F":
            events = ["Hypothesis", "Testimony", "Reliable", "visionReliability",
                      "objectivityReliability", "veracityReliability"]

            arcs["mandatory"] = [("Hypothesis", "Testimony"), ("Reliable", "Testimony"),
                                 ("visionReliability", "Reliable"), ("objectivityReliability", "Reliable"),
                                 ("veracityReliability", "Reliable")]

        elif struct == "T":
            events = ["Hypothesis", "Testimony", "testimony_seen", "testimony_objectivity",
                      "visionReliability", "objectivityReliability",
                      "veracityReliability"]

            arcs["mandatory"] = [("Hypothesis", "testimony_seen"), ("visionReliability", "testimony_seen"),
                                 ("testimony_seen", "testimony_objectivity"),
                                 ("objectivityReliability", "testimony_objectivity"),
                                 ("testimony_objectivity", "Testimony"), ("veracityReliability", "Testimony")]
        elif struct == "H":
            print("Hepler structure not implemented yet")
            exit()
        else:
            print("unknown structure passed to BN creation")
            exit()

        l = []
        for e1 in events:
            for e in events:
                if (e1, e) not in arcs["mandatory"]:
                    l.append((e1, e))

        arcs["forbidden"] = l

        for arc in arcs["mandatory"]:
            (h, t) = arc
            learner.addMandatoryArc(h, t)

        for arc in arcs["forbidden"]:
            (h, t,) = arc
            learner.addForbiddenArc(h, t)

        learner.useNoPrior()
        bn = learner.learnBN()
        gum.saveBN(bn, f"bns/{struct.lower()}/{struct.lower()}{hypotheses["testifies"]}.net")
        #gumimage.exportInference(bn, f"bns/{struct.lower()}/{struct.lower()}{hypotheses["testifies"]}.png")

        self.convert_networks_to_hugin(f"bns/{struct.lower()}/{struct.lower()}{hypotheses["testifies"]}")

    def permute_observations(self, variables, df1):

        (p, s, vic) = variables["stealhyp"]

        NO = "None"#True #False #"None" # the value if the thing is not observed, False, or None

        df1["vision_observation_permuted_tl"] = df1['vision_observation_permuted'].apply(
            ast.literal_eval)  # evaluate not as string but as list of tuples
        df1["tempvis"] = df1["vision_observation_permuted_tl"].apply(
            lambda x: [(t[0][3], t[1]) for t in x])  # select relevant info
        df1["visionPerturb"] = df1["tempvis"].apply(
            lambda x: next((t[1] for t in x if t[0] == vic), NO))  # match on victim id
        df1['visionReliability'] = df1["visionPerturb"].apply(lambda x: not x if x != "None" else "None")  # not perturbed = reliable

        df1["objectivity_permuted_tl"] = df1['objectivity_permuted'].apply(
            ast.literal_eval)  # evaluate not as string but as list of tuples
        df1["tempob"] = df1["objectivity_permuted_tl"].apply(
            lambda x: [(t[0][3], t[1]) for t in x])  # select relevant info
        df1["objectivityPermuted"] = df1["tempob"].apply(
            lambda x: next((t[1] for t in x if t[0] == vic), NO))  # match on victim id
        df1['objectivityReliability'] = df1["objectivityPermuted"].apply(lambda x: not x if x != "None" else "None")  # not perturbed = reliable

        df1["veracity_permuted_tl"] = df1['veracity_permuted'].apply(
            ast.literal_eval)  # evaluate not as string but as list of tuples
        df1["tempvera"] = df1["veracity_permuted_tl"].apply(
            lambda x: [(t[0][3], t[1]) for t in x])  # select relevant info
        df1["veracityPermuted"] = df1["tempvera"].apply(
            lambda x: next((t[1] for t in x if t[0] == vic), NO))  # match on victim id
        df1['veracityReliability'] = df1["veracityPermuted"].apply(
            lambda x: not x if x != "None" else "None")  # not perturbed = reliable
        #print(df1[['tempvis','visionPerturb', 'visionReliability']])
        #print(df1[['tempob','objectivityPermuted', 'objectivityReliability']])
        #print(df1[['tempvera','veracityPermuted', 'veracityReliability']])
        return df1

    def get_testimony(self, variables, df1):
        df1["testifies"] = df1['veracity'].apply(
            ast.literal_eval)  # evaluate not as string but as list of tuples
        v = variables["testifies"]
        (a, b, (c, d, e)) = v
        df1["originalTestifies"] = df1["testifies"]

        NO = "None" #False # False or NT

        df1[variables["testifies"]] = df1['testifies'].apply(
            lambda x: NO if x == [] else NO if all(len(t) >= 4 and e != (t[3]) for t in x) else any(
                len(t) >= 4 and (c, str(d), e) == (t[1], t[2], t[3]) for t in x)
        )
        return df1


    def get_events_from_df(self, witness_agent, df, variables, bn):
        final_step = df["Step"].max()
        df1 = df[(df["Step"] == final_step) & (df["id"] == witness_agent)] # final step of of the dataframe
        df1 = df1.copy()
        # evaluating the hypothesis "((1, 0, 0), STEAL, 8)" as stored in variables["stealhyp"]
        df1["stealEvents"] = df1['stealEvents'].apply(ast.literal_eval) # evaluate not as string but as list of tuples
        df1[variables["stealhyp"]] = df1['stealEvents'].apply(
            lambda x: any(len(t) >= 4 and variables["stealhyp"] == (t[1], t[2], t[3]) for t in x))

        # evaluating the testimony "(1 "Observed" (1, 0, 0), STEAL, 8))" as stored in variables["testimony"]
        # this depends on the implementation per bn
        if bn == "HB" or bn == "F":
            df1 = self.get_testimony(variables, df1)
            df1[["stealEvents", variables["stealhyp"], "originalTestifies", variables["testifies"]]].to_csv(f"data/test{variables["testifies"]}.csv", index=True)
            df1 = self.permute_observations(variables, df1)
            df1['Reliable'] = df1[['visionReliability','objectivityReliability', 'veracityReliability']].min(axis=1)
            #print(df1[['vision_observation_permuted','objectivity_permuted','veracity_permuted', 'visionReliability','objectivityReliability', 'veracityReliability','Reliable']])

            #df1[["run",variables["stealhyp"], variables["testifies"], "Reliable", "visionReliability", "objectivityReliability", "veracityReliability","stealEvents", "vision_observation", "veracity"]].to_csv(f"data/out/datacollector{variables["testifies"]}.csv", index=True)

            df1["Hypothesis"] = df1[variables["stealhyp"]]
            df1["Testimony"] = df1[variables["testifies"]]
            if bn == "HB":
                df1[["Hypothesis", "Testimony", "Reliable"]].to_csv(f"data/bndata/hb/{variables["testifies"]}.csv", index=True)
                df2 =df1[["Hypothesis", "Testimony", "Reliable"]]
                self.create_BN_from_df(df2, "HB", variables)

            elif bn == "F":
                df1[["Hypothesis", "Testimony", "Reliable",
                     "visionReliability", "objectivityReliability", "veracityReliability"]].to_csv(f"data/bndata/f/{variables["testifies"]}.csv",
                                                                    index=True)
                df2 = df1[["Hypothesis", "Testimony", "Reliable", "visionReliability", "objectivityReliability", "veracityReliability"]]
                self.create_BN_from_df(df2, "F", variables)
            else:
                print("not building implemented in line 238")
                exit()
        elif bn == "T":

            df1["testimony_seen"] = df1['vision_observation'].apply( ast.literal_eval)  # evaluate not as string but as list of tuples
            v = variables["testimony_seen"]
            (a, b, (c, d, e)) = v

            df1["originalvision"] = df1["vision_observation"]
            df1["testimony_seen"] = df1['testimony_seen'].apply(
                lambda x:  any( len(t) >= 4 and (c, str(d), e) == (t[1], t[2], t[3]) for t in x)
            )
            df1["testimony_objectivity"] = df1['objectivity'].apply(
                ast.literal_eval)  # evaluate not as string but as list of tuples
            v = variables["testimony_objectivity"]
            (a, b, (c, d, e)) = v

            df1["originalobjectivity"] = df1["objectivity"]
            df1["testimony_objectivity"] = df1['testimony_objectivity'].apply(
                lambda x: any(len(t) >= 4 and (c, str(d), e) == (t[1], t[2], t[3]) for t in x)
            )

            df1["testifies"] = df1['veracity'].apply(
                ast.literal_eval)  # evaluate not as string but as list of tuples
            v = variables["testifies"]
            (a, b, (c, d, e)) = v
            df1["originalTestifies"] = df1["testifies"]
            df1["testifies"] = df1['testifies'].apply(
                lambda x: any(len(t) >= 4 and (c, str(d), e) == (t[1], t[2], t[3]) for t in x)
            )
            df1 = self.permute_observations(variables, df1)

            df1[["run", variables["stealhyp"], "testifies", "testimony_seen", "visionReliability",
                 "testimony_objectivity","objectivityReliability", "veracityReliability", "stealEvents", "vision_observation",
                 "veracity"]].to_csv(f"data/out/datacollector{variables["testifies"]}.csv", index=True)
            df1["Hypothesis"] = df1[variables["stealhyp"]]
            df1["Testimony"] = df1["testifies"]

            df1[["Hypothesis", "Testimony",
                 "testimony_seen","testimony_objectivity",
                 "visionReliability", "objectivityReliability", "veracityReliability"]].to_csv(
                f"data/bndata/t/{variables["testifies"]}.csv",
                index=True)
            df2 = df1[["Hypothesis", "Testimony", "testimony_seen","testimony_objectivity",
                       "visionReliability", "objectivityReliability",
                       "veracityReliability"]]

            self.create_BN_from_df(df2, "T", variables)

        elif bn == "H":
            pass
        else:
            print("not implemented")


    def add_quotes(self, input_string):
        # Remove the parentheses from the string
        elements = input_string.split()

        # Add quotes around each element
        quoted_elements = [f'\"{elem}\"' for elem in elements]

        # Join the quoted elements back into a string with spaces
        output_string = f"({' '.join(quoted_elements)} );\n"

        return output_string

    def convert_networks_to_hugin(self, name):
        # print(name)
        with open(f"{name}.net", 'r') as file:
            lines = file.readlines()

        # Remove the first three lines
        lines[2] = f"name = \"bn\";\n"

        for i, line in enumerate(lines):
            if "states" in line:
                s = line.split("=")[1]
                s_i = s.replace("(", "")
                s_i = s_i.replace(");", "")
                new_s = self.add_quotes(s_i)
                lines[i] = line.replace(s, new_s)

        with open(f"{name}hugin.net", 'w') as file:
            file.writelines(lines)





    def run_model(self):
        print(self.num_agents)
        model = ForestFire(30, 30, 0.00001, num_agents=self.num_agents)
        for i in range(0, self.simulation_time+1):
            model.step()
        #print(model.datacollector.get_agent_vars_dataframe())
        return model

def run_visual():
    # this calls "server.py" and lanches it
    server.launch(open_browser=True)


def run_experiment():
    e = Experiment()
    #e.collect_data(e.num_runs)
    e.preprocess_data()

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Don't truncate content within columns

run_experiment()

#run_visual()


