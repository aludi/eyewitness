import mesa
import numpy as np
import math
import random

class Ranger(mesa.Agent):
    def __init__(self, unique_id, people_id, model):
        """
        Create a new ranger.

        """
        super().__init__(unique_id, model)
        prof = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (1,1,1), (1,1,2), (2,2,0), (2,2,1), (1,2,0), (2,2,2)]
        self.goal = (0,0)
        self.profile = prof[people_id]
        self.people_id = people_id
        self.condition = "human"
        self.good_visible_cells=[]
        self.bad_visible_cells=[]
        self.was_observed_stealing_by = []
        self.valuable = 0.8
        self.stolen_goods = []
        self.target = None
        self.intent = False
        self.stealing = False
        self.attempted_stealing = False

    def set_goal(self, pos):
        self.goal = pos

    def calc_heading(self, old_pos, new_pos):
        x_old, y_old = old_pos
        x_new, y_new = new_pos

        dx = x_new - x_old
        dy = y_new - y_old

        angle = math.degrees(math.atan2(dy, dx))  # Get angle in degrees
        return angle % 360  # Ensure it stays within [0, 360]

    def get_cone_vision(self, agent_pos, heading, radius, angle):
        visible_cells = set()
        cx, cy = agent_pos  # Agent's position (center of vision)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy  # Candidate cell

                # Compute distance
                distance = math.sqrt(dx ** 2 + dy ** 2)
                if distance > radius or distance == 0:
                    continue  # Skip out-of-range cells and the agent's own cell

                # Compute angle relative to agent's direction
                cell_angle = math.degrees(math.atan2(dy, dx)) % 360
                min_angle = (heading - angle / 2) % 360
                max_angle = (heading + angle / 2) % 360

                # Handle angle wrapping around 0°
                if min_angle < max_angle:
                    in_cone = min_angle <= cell_angle <= max_angle
                else:
                    in_cone = cell_angle >= min_angle or cell_angle <= max_angle

                if in_cone:
                    visible_cells.add((x, y))
        return visible_cells

    def people_seen(self):
        bad_range = []
        good_range = []

        for cell in self.bad_visible_cells:
            px, py = cell
            if 0 <= px < self.model.grid.width and 0 <= py < self.model.grid.height:
                cell_cont = self.model.grid.get_cell_list_contents([cell])
                filtered_agents = [agent for agent in cell_cont if isinstance(agent, Ranger)]
                for fa in filtered_agents:
                    bad_range.append(fa.people_id) # maybe to extend with a location?

        for cell in self.good_visible_cells:
            px, py = cell
            if 0 <= px < self.model.grid.width and 0 <= py < self.model.grid.height:
                cell_cont = self.model.grid.get_cell_list_contents([cell])
                filtered_agents = [agent for agent in cell_cont if isinstance(agent, Ranger)]
                for fa in filtered_agents:
                    good_range.append(fa.people_id) # maybe to extend with a location?
        only_bad = []
        for agent_id in bad_range:
            if agent_id not in good_range:
                only_bad.append(agent_id)

        return bad_range, good_range

    def move_step_to_goal(self):
        hx, hy = self.goal
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True, include_center=True)
        bestx, besty = 1000, 1000
        best = bestx*besty
        for step in possible_steps:
            stepx, stepy = step
            if ((hx - stepx) ** 2 + (hy - stepy) ** 2 < best):
                best = (hx - stepx) ** 2 + (hy - stepy) ** 2
                bestx, besty = stepx, stepy

        if (bestx, besty) == (1000, 1000):
            possible_steps = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=False)
            new_position = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)
        else:
            self.model.grid.move_agent(self, (bestx, besty))


    def move(self):
        old_pos = self.pos
        self.move_step_to_goal()
        new_pos = self.pos
        heading_angle = self.calc_heading(old_pos, new_pos)
        self.heading_angle = heading_angle



    def get_vision(self):
        self.bad_visible_cells = self.get_cone_vision(self.pos, self.heading_angle, radius=15, angle=70)
        self.good_visible_cells = self.get_cone_vision(self.pos, self.heading_angle, radius=10, angle=70)

        for cell in self.bad_visible_cells:
            for place in self.model.place_agents:
                if place.pos == cell:
                    place.set_condition("bad_on")
                    place.set_owner_1(self)  # this agent can see this place

        for cell in self.good_visible_cells:
            for place in self.model.place_agents:
                if place.pos == cell:
                    place.set_condition("good_on")
                    place.set_owner_1(self)  # this agent can see this place


    def steal_attempt(self):
        self.attempted_stealing = True
        if self.pos == self.target.pos and self.target.valuable > 0:
            # steal threshold:
            if random.random() > 0.1:
                self.stealing = True
                self.stolen_goods.append(self.target.valuable)
                self.target.valuable = 0
        #self.target = None



    def determine_goal(self):
        #print("\n\n\n")
        #print(self.people_id)
        if self.attempted_stealing == True:
            self.intent = False
            self.attempted_stealing = False
            self.stealing = False
            self.target = None

        #print(self.target, self.intent, self.attempted_stealing, self.stealing)

        if self.pos == self.goal:
            if self.target is None:
                # set new goal
                #print("go to new store")
                self.set_goal(random.choice(self.model.goal_spaces))
            else:
                #print(self.target, self.intent)
                #print("target reached")
                self.steal_attempt()

        if self.target is None:
            for cell in self.bad_visible_cells:
                px, py = cell
                if 0 <= px < self.model.grid.width and 0 <= py < self.model.grid.height:
                    cell_cont = self.model.grid.get_cell_list_contents([cell])
                    filtered_agents = [agent for agent in cell_cont if isinstance(agent, Ranger)]
                    # risk-reward calculation here
                    for agent in filtered_agents:
                        if random.random() < agent.valuable:    # chance of 0.4 that the agent will attempt to steal
                            self.intent = True
                            self.target = agent
                            self.goal = agent.pos

        else:
            self.goal = self.target.pos


        #if self.target is not None:
        #    print(self.people_id, self.target.people_id, self.intent, self.attempted_stealing, self.stealing)

    def step(self):
        #print(self.people_id, self.valuable, self.stolen_goods)
        self.was_observed_stealing_by = []
        self.determine_goal()
        self.move()
        self.get_vision()




class Place(mesa.Agent):
    def __init__(self, unique_id,store, model):
        """
        Create a new tree.
        Args:
            pos: The tree's coordinates on the grid.
            model: standard model reference for agent.
        """
        super().__init__(unique_id, model)

        self.vision_condition = "off"
        self.vision_owner_list = []
        self.store_location = store
        self.condition = self.vision_condition

    def set_condition(self, cond):
        self.vision_condition = cond
        self.condition = self.vision_condition

    def set_owner(self, r):
        self.vision_owner_list = []

    def set_owner_1(self, ranger): # who can observe this spot
        self.vision_owner_list.append(ranger)



class TreeCell(mesa.Agent):
    """
    A tree cell.

    Attributes:
        x, y: Grid coordinates
        condition: Can be "Fine", "On Fire", or "Burned Out"
        unique_id: (x,y) tuple.

    unique_id isn't strictly necessary here, but it's good
    practice to give one to each agent anyway.
    """

    def __init__(self, unique_id, model):
        """
        Create a new tree.
        Args:
            pos: The tree's coordinates on the grid.
            model: standard model reference for agent.
        """
        super().__init__(unique_id, model)

        self.condition = "Fine"

    def step(self):
        """
        If the tree is on fire, spread it to fine trees nearby.
        """
        if self.condition == "On Fire":
            for neighbor in self.model.grid.iter_neighbors(self.pos, True):
                if neighbor.condition == "Fine":
                    neighbor.condition = "On Fire"
            self.condition = "Burned Out"