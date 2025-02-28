import mesa

from .model import ForestFire
from .agent import TreeCell, Ranger, Place

COLORS = {"Fine": "#00AA00", "On Fire": "#880000", "Burned Out": "#000000",
          "human":"#FFFF00", "thief":"#FF0000",
          "store": "#FFC0CB",
          "off":"#90EE90", "bad_on":"#89CFF0","good_on":"#0096FF"}


def forest_fire_portrayal(agent):
    portayal = {}
    if agent is None:
        return

    shape = "rect"
    layer=1
    s=""
    if type(agent) == Ranger:
        layer = 2
        s = str(agent.people_id)
        if agent.intent:
            color = COLORS["thief"]
        else:
            color = COLORS[agent.condition]

    if type(agent) == Place:
        layer=0
        if agent.store_location == "store":
            color = COLORS[agent.store_location]
        else:
            color = COLORS[agent.condition]

    portrayal = {"Shape": shape, "w": 1, "h": 1, "Filled": "true", "Color":color,
                 "Layer": layer, "alpha":0.8, "text":s,
                 "text_color": "black"}
    (x, y) = agent.pos
    portrayal["x"] = x
    portrayal["y"] = y

    return portrayal




canvas_element = mesa.visualization.CanvasGrid(
    forest_fire_portrayal, 50, 50, 500, 500
)
tree_chart = mesa.visualization.ChartModule(
    [{"Label": label, "Color": color} for (label, color) in COLORS.items()]
)
pie_chart = mesa.visualization.PieChartModule(
    [{"Label": label, "Color": color} for (label, color) in COLORS.items()]
)

model_params = {
    "height": 50,
    "width": 50,
    "density": mesa.visualization.Slider("Tree density", 0.00001, 0.01, 1.0, 0.01),
}
server = mesa.visualization.ModularServer(
    ForestFire, [canvas_element, tree_chart, pie_chart], "Forest Fire", model_params
)
#server.launch()