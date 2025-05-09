import pandas as pd
import pyAgrum as gum
import pyAgrum.lib.image as gumimage
import os
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

def get_evidence(bn_type):
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
                     e_unreliable2, e_unreliable3, e_unreliable4]

def bn_inference_collective(bn_types):
    outcomes = [["Evidence", "BN", "PTrue", "PFalse"]]
    for bn_type in bn_types:
        file_path = f"bns/collective/{bn_type.lower()}.net"
        # Loop through all files and directories in the folder
        ev_list = get_evidence(bn_type)

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
                outcomes.append([file_path, 0, 1])
    print(outcomes)

    with open("data/results/collective/outcomes.csv", 'w') as f:
        wr = csv.writer(f)
        wr.writerows(outcomes)


def bn_inference(bn_types, hypothesis):
    for bn_type in bn_types:
        folder_path = f"bns/{bn_type.lower()}"
        # Loop through all files and directories in the folder
        ev_list = get_evidence(bn_type)
        for evidence in ev_list:
            #print(evidence)
            outcomes = [["Hypothesis", "PTrue", "PFalse", "total"]]
            inf_problem = [["Hypothesis"]]
            filename = f"{bn_type.lower()}{hypothesis}"
            file_path = os.path.join(folder_path, filename)
            try:
                bn = gum.loadBN(file_path)
                ie = gum.LazyPropagation(bn)

                ie.setEvidence(evidence)

                pt = ie.posterior("Hypothesis")[{"Hypothesis": "True"}]
                pf = ie.posterior("Hypothesis")[{"Hypothesis": "False"}]
                outcomes.append([file_path, pt, pf])

            except Exception as e:
                print(e)
                print(f"{filename} {evidence} missing data, cannot calculate posterior")
                inf_problem.append([file_path])  # things that are not observed in the dataset have a probability of 0
                outcomes.append([file_path, 0, 1])

            ev = evidence

            with open(f"data/results/{bn_type.lower()}/{ev}.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(outcomes)

            with open(f"data/out/problem/{bn_type.lower()}PROBLEM{ev}.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(inf_problem)

            df = pd.read_csv(f"data/results/{bn_type.lower()}/{ev}.csv")
            # Apply the extraction
            '''
            df["sort_key"] = df["Hypothesis"].apply(extract_sort_key)

            # Sort by the composite key
            df= df.sort_values("sort_key").drop(columns="sort_key")
            print(df["Hypothesis"])
            #print(df["sort_key"])
            df.to_csv(f"data/results/{bn_type.lower()}/{ev}.csv")'''

def extract_sort_key(hypo):
    match = re.search(r"\((\d+), 'TESTIFIES', \(\((\d), (\d), (\d)\), 'STEAL', (\d+)\)\)", hypo)
    if match:
        a, b1, b2, b3, c = map(int, match.groups())
        return (a, b1, b2, b3, c)
    else:
        return (999, 999, 999, 999, 999)  # fallback for malformed strings





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


