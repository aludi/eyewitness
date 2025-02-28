from forest_fire.server import server
from forest_fire.model import ForestFire
import pandas as pd

def collect_data(runs):
    df_list = []
    for i in range(0, runs):
        model = run_model()
        df = pd.DataFrame(model.relevant_data)
        df["run"] = i
        df_list.append(df)
    df = pd.concat(df_list)
    df.to_csv("data/stealing_testimony.csv", index=False)

def run_model():
    model = ForestFire(50, 50, 0.00001)
    for i in range(0, 83):
        model.step()
    return model

def run_visual():
    server.launch(open_browser=True)

collect_data(50)

#run_visual()


