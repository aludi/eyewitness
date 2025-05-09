import mesa

from world.model import ForestFire
from world.agent import TreeCell, Ranger, Place
from mesa.visualization.modules import TextElement

COLORS = {"Fine": "#00AA00", "On Fire": "#880000", "Burned Out": "#000000",
          "human": "white",#"#FFFF00",
          "thief":"red",#"#bc0011",
          "intent":"white",#"#f9ca8a",
          "attempt":"white", #"#fc9501",
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
        s = str(agent.people_id) +str(agent.profile)
        shape = "circle"
        r = 1
        if agent.stealing or agent.attempted_stealing:
            color=COLORS["thief"]

        elif agent.attempted_stealing:
            color=COLORS["attempt"]
        elif agent.intent:
            color = COLORS["intent"]
        else:
            color = COLORS[agent.condition]

    if type(agent) == Place:
        layer=0
        if agent.store_location == "store":
            color = COLORS[agent.store_location]
        else:
            color = COLORS[agent.condition]

    portrayal = {"Shape": shape, "r":1, "w": 1, "h": 1, "Filled": "true", "Color":color,
                 "Layer": layer, "alpha":0.8, "text":s,
                 "text_color": "black"}
    (x, y) = agent.pos
    portrayal["x"] = x
    portrayal["y"] = y

    return portrayal


class Text(TextElement):
    def render(self, model):
        str_ = ""
        for agent in model.people_agents:
            # the \t or \n in the string below do not show up in the webpage viz.
            str_ = str_ + str(agent.profile) + " : " + str(agent.state) + str(agent.stolen_goods) +",\t"
        return str_


# create the simulation grid
canvas_element = mesa.visualization.CanvasGrid(
    forest_fire_portrayal, 50, 50, 500, 500
)

# create the text that describes what the agents are doing
text = Text()

# model params, such as the height, width of the simulation, and the number of agents (hardcoded for the visual run).
model_params = {
    "height": 50,
    "width": 50,
    "density": mesa.visualization.Slider("Tree density", 0.00001, 0.01, 1.0, 0.01),
    "num_agents":0
}

# this instantiates the server, with the agent grid (canvas_element) and the text that describes what the agents are doing
server = mesa.visualization.ModularServer(
    ForestFire, [canvas_element, text], "Forest Fire", model_params
)
#server.launch()