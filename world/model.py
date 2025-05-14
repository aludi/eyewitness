import mesa
from world.agent import TreeCell, Ranger, Place
import random, itertools
import numpy as np
import pandas as pd
import logging




'''agent_reporters={"step": lambda a: a.model.schedule.time,
                             "id": "people_id",
                             "pos": "pos", "profile": "profile",
                             "state": "state",
                             "target": (lambda x:False if x.target is None else x.target.people_id),
                             "intent": "intent",
                             "attemptedStealing": "attempted_stealing",
                             "stealing": "stealing",
                             "stolenfrom":"stolen_from",
                             "agents_perceived_good": "agents_observed_good",
                             "agents_perceived_bad": "agents_observed_bad",
                             "stealEvents":"agents_steal_events",
                             "couldobserve":"couldobserve",
                             "observed_stealing_good":"observed_stealing_good",
                             "observed_stealing_bad":"observed_stealing_bad",
                             "vision_observation":"vision_observation",
                            "vision_observation_permuted":"vision_observation_permuted",
                            "objectivity":"objectivity",
                            "objectivity_permuted":"objectivity_permuted",
                            "veracity":"veracity",
                             "veracity_permuted":"veracity_permuted"},
                             
todo:
goal locations
agent jitter in walk
predator behaviour for stealing

'''

class ForestFire(mesa.Model):
    """
    Simple Forest Fire model.
    """

    def __init__(self, width=50, height=50, density=0.0, num_agents=10, log=None):
        """
        Create a new forest fire model.

        Args:
            width, height: The size of the grid to model
            density: What fraction of grid cells have a tree in them.
        """
        # Set up model objects
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.MultiGrid(width, height, torus=False)
        if log is None:
            logging.basicConfig(
                filename='experiment.log',  # Log file path
                level=logging.INFO,  # Log level
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            log = logging.getLogger(__name__)
            log.info("Logger is working")
        self.model_log = log

        self.model_log.info("Logging has started.")

        self.place_agents = []
        self.people_agents = []
        self.data_collection_time = []
        self.collected_data_model_dict = {}

        prof_opportunities = list(itertools.product(range(2), repeat=3))
        #print(len(list(itertools.product(range(2), repeat=3))), num_agents)
        #print(prof_opportunities)
        #exit()
        for i in range(2, 10):
            if len(list(itertools.product(range(i), repeat=3))) >= num_agents:
                prof_opportunities=list(itertools.product(range(i), repeat=3))
                break


        self.prof = prof_opportunities
        #print(self.prof)


        self.relevant_data = []



        unique_id = 0
        self.ranger_num = num_agents
        self.stealevents = []


        # set goal spaces
        self.goal_spaces = []

        for i in range(0, 100):
            self.goal_spaces.append(
                (random.randrange(0, width), random.randrange(0, height)))


        # Place a tree in each cell with Prob = density
        for contents, (x, y) in self.grid.coord_iter():
            if (x,y) in self.goal_spaces:
                store="store"
            else:
                store = "no store"
            new_place = Place(unique_id, store, model=self)
            unique_id+=1
            self.place_agents.append(new_place)
            self.grid.place_agent(new_place, (x,y))

        ppl_id = 0
        profles = []
        for ranger_ind in range(0, self.ranger_num):
            pos_x = self.random.randint(0, width-1)
            pos_y = self.random.randint(0,height-1)
            new_ranger = Ranger(unique_id = unique_id, people_id=ppl_id, model=self)
            #print(ppl_id, new_ranger.profile)
            new_ranger.set_goal(random.choice(self.goal_spaces))
            profles.append(new_ranger.profile)
            ppl_id +=1
            unique_id += 1
            self.grid.place_agent(new_ranger, (pos_x, pos_y))
            goal_x = self.random.randint(0, width - 1)
            goal_y = self.random.randint(0, height - 1)
            new_ranger.set_goal((goal_x, goal_y))
            self.schedule.add(new_ranger)
            self.people_agents.append(new_ranger)

        #print(set(profles))
        #self.initialize_data_model_dict()

        # Define dimensions
        dim = self.ranger_num  # You can set to 9 if using 0 to 8

        # Create empty arrays
        hyp_array = np.full((dim, dim, dim),"", dtype=object)  # storing steal strings
        testimony_array = np.full((dim, dim, dim),"", dtype=object) # storing testimony strings

        vhyp_array = np.full((dim, dim, dim), False, dtype=bool)  # for storing booleans
        vtestimony_array = np.full((dim, dim, dim), False, dtype=bool)
        vseen_stealing= np.full((dim, dim, dim), False, dtype=bool)
        vreliability_vision= np.full((dim, dim, dim), True, dtype=bool)
        vobjective_interpretation= np.full((dim, dim, dim), False, dtype=bool)
        vreliability_objective= np.full((dim, dim, dim), True, dtype=bool)
        vveracity= np.full((dim, dim, dim), False, dtype=bool)
        vreliability_veracity= np.full((dim, dim, dim), True, dtype=bool)
        # Fill the arrays

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    hyp_array[i, j, k] = f"{self.prof[i]}S{j}"
                    testimony_array[i, j, k] = f"{k}T{self.prof[i]}S{j}"

        # Store in dictionary
        self.reporters = {"Hyp": hyp_array,
                "vHyp": vhyp_array,
                "Testimony": testimony_array,
                "vTestimony": vtestimony_array,
                "vseen_stealing" :vseen_stealing,
                "vreliability_vision" : vreliability_vision,
                "vobjective_interpretation" :vobjective_interpretation,
                "vreliability_objective":vreliability_objective,
                "vveracity" :vveracity,
                "vreliability_veracity" : vreliability_veracity}

        #print(self.reporters["Hyp"][0][0][0])  # "0S0"
        #print(self.reporters["vHyp"][0][0][0])  # True

        self.running = True


    def step(self):
        """
        Advance the model by one step.
        """
        self.model_log.info(f"\t timestep: {self.schedule.time}")
        condition = "off"
        for agent in self.place_agents:
            agent.set_condition(condition)
            agent.set_owner = []

        if self.schedule.time == 40: #%10 == 0:
            self.running = False

        self.schedule.step()



    @staticmethod
    def count_type(model, tree_condition):
        """
        Helper method to count trees in a given condition in a given model.
        """
        count = 0
        for tree in model.schedule.agents:
            if tree.condition == tree_condition:
                count += 1
        return count