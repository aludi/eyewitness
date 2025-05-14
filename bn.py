import pandas as pd
import pyAgrum as gum
import pyAgrum.lib.image as gumimage
import os
import itertools
import re
import csv

def make_bn(df, struct, hypotheses):
     create_BN_from_df(df, struct, hypotheses)


def create_BN_from_df(df, struct, hypotheses):
    arcs = {}
    learner = gum.BNLearner(df)
    events = []

    if struct == "HB":
        events = ["Hypothesis", "Testimony", "Reliable"]
        arcs["mandatory"] = [("Hypothesis", "Testimony"), ("Reliable", "Testimony")]

    elif struct == "F":
        events = ["Hypothesis", "Testimony", "Reliable", "vision_observationReliability",
                  "objectivityReliability", "veracityReliability"]

        arcs["mandatory"] = [("Hypothesis", "Testimony"), ("Reliable", "Testimony"),
                             ("vision_observationReliability", "Reliable"), ("objectivityReliability", "Reliable"),
                             ("veracityReliability", "Reliable")]

    elif struct == "T":
        events = ["Hypothesis", "Testimony", "testimony_seen", "testimony_objectivity",
                  "vision_observationReliability", "objectivityReliability",
                  "veracityReliability"]

        arcs["mandatory"] = [("Hypothesis", "testimony_seen"), ("vision_observationReliability", "testimony_seen"),
                             ("testimony_seen", "testimony_objectivity"),
                             ("objectivityReliability", "testimony_objectivity"),
                             ("testimony_objectivity", "Testimony"), ("veracityReliability", "Testimony")]
    elif struct == "H":
        events = ["Hypothesis", "Testimony", "testimony_seen", "testimony_objectivity",
                  "vision_observationReliability", "objectivityReliability",
                  "veracityReliability", "testimony_veracity"]

        arcs["mandatory"] = [("Hypothesis", "testimony_seen"), ("vision_observationReliability", "testimony_seen"),
                             ("testimony_seen", "testimony_objectivity"),
                             ("objectivityReliability", "testimony_objectivity"),
                             ("testimony_objectivity", "testimony_veracity"),
                             ("veracityReliability", "testimony_veracity"),
                             ("testimony_veracity", "Testimony")]
        # print("Hepler structure not implemented yet")
        # exit()
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
    print(bn)
    if hypotheses != "collective":
        gum.saveBN(bn, f"bns/{struct.lower()}/{struct.lower()}{hypotheses}.net")
        # gumimage.exportInference(bn, f"bns/{struct.lower()}/{struct.lower()}{hypotheses["testifies"]}.png")
        convert_networks_to_hugin(f"bns/{struct.lower()}/{struct.lower()}{hypotheses}")
    else: # collective bn
        gum.saveBN(bn, f"bns/collective/{struct.lower()}.net")
        convert_networks_to_hugin(f"bns/collective/{struct.lower()}")

    calculate_joint(bn,hypotheses, struct)


def calculate_joint(bn, hyp, bn_type):
    variables = bn.names()
    ie = gum.LazyPropagation(bn)
    ie.addJointTarget(set(variables))
    p = ie.jointPosterior(set(variables))
    I = gum.Instantiation(p)
    list_jp = []
    for i in I.loopIn():
        d = i.todict(True)
        d["probability"] = p.get(i)
        list_jp.append(d)
        #print(p.get(i))
        #print(i, p)

    df = pd.DataFrame(list_jp)
    if hyp != "collective":
        df.to_csv(f"data/jointprobs/individual/{bn_type.lower()}{hyp}.csv")
    else:
        df.to_csv(f"data/jointprobs/{bn_type.lower()}.csv")





def get_evidence(bn_fp):
    #file_path = f"bns/{col}/{bn_type.lower()}.net"
    # Loop through all files and directories in the folder
    #ev_list = get_evidence(bn_type)
    try:
        bn = gum.loadBN(bn_fp)

    except Exception as e:
        return []

    #print(bn.names())
    domains = {}
    no_inf_nodes =  ["Hypothesis", "testimony_seen", "testimony_objectivity", "testimony_veracity"]
    if "hb" not in bn_fp:
        no_inf_nodes.append("Reliable")
    for name in bn.names():
        #print(name, bn.variableFromName(name).labels())
        if name not in no_inf_nodes:
            domains[name] = list(bn.variableFromName(name).labels())
            domains[name].append("unspecified")

    cols = list(domains.keys())

    # Generate all combinations using each column's specific domain
    combinations = itertools.product(*(domains[col] for col in cols))

    # Build the list of dictionaries, skipping any 'unspecified' values
    combinations_dict = []
    #print(combinations)
    for combo in combinations:
        d = {cols[i]: combo[i] for i in range(len(cols)) if combo[i] != 'unspecified'}
        combinations_dict.append(d)

    #for d in combinations_dict:
        #print(d)
    #exit()
    return combinations_dict


    '''
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

    return [{}, {"Testimony": "True"}, {"Testimony": "False"}, e_all_true, e_unreliable1,
                     e_unreliable2, e_unreliable3, e_unreliable4]'''



def bn_inference_collective(bn_types):
    outcomes = [["Evidence", "BN", "PTrue", "PFalse"]]
    for bn_type in bn_types:
        file_path = f"bns/collective/{bn_type.lower()}.net"
        # Loop through all files and directories in the folder
        ev_list = get_evidence(file_path)

        for evidence in ev_list:

            inf_problem = [["Hypothesis"]]

            try:
                bn = gum.loadBN(file_path)
                ie = gum.LazyPropagation(bn)

                ie.setEvidence(evidence)

                pt = ie.posterior("Hypothesis")[{"Hypothesis":"True"}]
                pf = ie.posterior("Hypothesis")[{"Hypothesis":"False"}]
                outcomes.append([evidence, bn_type.lower(), pt, pf])

            except Exception as e:
                print(e)
                print(f"{file_path} {evidence} missing data, cannot calculate posterior")
                inf_problem.append([file_path])     # things that are not observed in the dataset have a probability of 0
                outcomes.append([file_path, -1, -1])
    print(outcomes)

    with open("data/results/collective/outcomes.csv", 'w') as f:
        wr = csv.writer(f)
        wr.writerows(outcomes)


def bn_inference(bn_types):
    for bn_type in bn_types:
        folder_path = f"bns/{bn_type.lower()}"
        # Loop through all files and directories in the folder
        for filename in os.listdir(folder_path):
            if "hugin" not in filename:
                outcomes = [["Evidence", "PTrue", "PFalse"]]
                file_path = os.path.join(folder_path, filename)
                ev_list = get_evidence(file_path)

                for evidence in ev_list:

                    #print(evidence)
                    #inf_problem = [["Hypothesis"]]
                    #filename = f"{bn_type.lower()}{hypothesis}"
                    try:
                        bn = gum.loadBN(file_path)
                        ie = gum.LazyPropagation(bn)

                        ie.setEvidence(evidence)

                        pt = ie.posterior("Hypothesis")[{"Hypothesis": "True"}]
                        pf = ie.posterior("Hypothesis")[{"Hypothesis": "False"}]
                        outcomes.append([evidence, pt, pf])

                    except Exception as e:
                        print(e)
                        print(f"{filename} {evidence} missing data, cannot calculate posterior")
                        #inf_problem.append([file_path])  # things that are not observed in the dataset have a probability of 0
                        outcomes.append([evidence, -1, -1])

                with open(f"data/results/{bn_type.lower()}/{filename.split(".net")[0]}.csv", 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(outcomes)



def add_quotes(input_string):
    # Remove the parentheses from the string
    elements = input_string.split()

    # Add quotes around each element
    quoted_elements = [f'\"{elem}\"' for elem in elements]

    # Join the quoted elements back into a string with spaces
    output_string = f"({' '.join(quoted_elements)} );\n"

    return output_string

def convert_networks_to_hugin(name):
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
            new_s = add_quotes(s_i)
            lines[i] = line.replace(s, new_s)

    with open(f"{name}hugin.net", 'w') as file:
        file.writelines(lines)


