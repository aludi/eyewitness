import mesa
from .agent import TreeCell, Ranger, Place
import random


'''
todo:
goal locations
agent jitter in walk
predator behaviour for stealing

'''
class ObservationStruc():
    def __init__(self, subj_agent, obj_agent, indirect_agent, predicate_sub, predicate_obj, obs_sense):
        self.predicate_sub = predicate_sub
        self.predicate_obj = predicate_obj

        self.subj_agent = subj_agent
        self.obj_agent = obj_agent
        self.indirect_agent = indirect_agent

        self.original_obj_agent_profile = obj_agent.profile
        self.observed_obj_agent_profile, self.vis_perturb = self.vision_perturbation(obs_sense)
        self.memorized_obj_agent_profile = False
        self.testified_obj_agent_profile = self.observed_obj_agent_profile
        self.memory_perturb = False
        self.veracity_perturb = False

        self.observation_time = self.subj_agent.model.schedule.time



        self.subj_agent_pos = subj_agent.pos
        if obj_agent is not None:
            self.obj_agent_pos = obj_agent.pos
        if indirect_agent is not None:
            self.indirect_agent_pos = indirect_agent.pos

    def __repr__(self):
        if self.predicate_obj is None:
            return (f'\n{self.subj_agent.people_id} {self.predicate_sub} {self.obj_agent.people_id}, '
                    f' at pos {self.obj_agent_pos} at time {self.observation_time}')
        else:

            return (f'\n{self.subj_agent.people_id} {self.predicate_sub} '
                f'<{self.testified_obj_agent_profile}> (real profile = <{self.original_obj_agent_profile}>), '
                f'  {self.predicate_obj} {self.indirect_agent.people_id} at {self.obj_agent_pos} at time {self.observation_time}')


    def vision_perturbation(self, obs_sense):
        vis_perturb = False
        if obs_sense == "good":
            return self.original_obj_agent_profile, vis_perturb
        else:
            (a, b, c) = self.original_obj_agent_profile
            if random.random() < 0.4:
                if random.random() < 0.9:
                    a = random.choice([0, 1, 2])
                    vis_perturb = True
                if random.random() < 0.3:
                    b = random.choice([0, 1, 2])
                    vis_perturb = True

                if random.random() < 0.3:
                    c = random.choice([0, 1, 2])
                    vis_perturb = True

        return (a,b,c), vis_perturb



    def memory_perturbation(self, d_t):
        (a, b, c) = self.observed_obj_agent_profile
        if self.memory_perturb is False:
            self.memory_perturb = False
            if d_t > 10:    # after some time the memory fades and risks bad reporting
                if random.random() < 0.4:
                    if random.random() < 0.3:
                        a = random.choice([0, 1, 2])
                        self.memory_perturb = True
                    if random.random() < 0.9:
                        b = random.choice([0, 1, 2])
                        self.memory_perturb = True

                    if random.random() < 0.3:
                        c = random.choice([0, 1, 2])
                        self.memory_perturb = True
        else: # if we recall again we are changing the memory, not the original observation
            (a, b, c) = self.memorized_obj_agent_profile
            if d_t > 10:  # after some time the memory fades and risks bad reporting
                if random.random() < 0.4:
                    if random.random() < 0.3:
                        a = random.choice([0, 1, 2])
                        self.memory_perturb = True
                    if random.random() < 0.9:
                        b = random.choice([0, 1, 2])
                        self.memory_perturb = True

                    if random.random() < 0.3:
                        c = random.choice([0, 1, 2])
                        self.memory_perturb = True

        self.memorized_obj_agent_profile = (a, b, c)
        return (a,b,c), self.memory_perturb


    def veracity_perturbation(self):
        (a, b, c) = self.memorized_obj_agent_profile
        if self.veracity_perturb == False: # if in the first testimony the agent has decided to lie they will keep doing that
            self.veracity_perturb = False
            # agents have a tendency to lie randomly
            if random.random() < 0.4:
                if random.random() < 0.3:
                    a = random.choice([0, 1, 2])
                    self.veracity_perturb = True

                if random.random() < 0.3:
                    b = random.choice([0, 1, 2])
                    self.veracity_perturb = True

                if random.random() < 0.9:
                    c = random.choice([0, 1, 2])
                    self.veracity_perturb = True

        self.testified_obj_agent_profile = (a, b, c)
        return (a,b,c), self.veracity_perturb

    def get_data_testimony(self):
        if self.indirect_agent is None:
            indirect_agent_id = "None"
        else:
            indirect_agent_id = self.indirect_agent.people_id
        if self.predicate_obj is None:
            action_2 = "None"
        else:
            action_2 = self.predicate_obj

        self.testimony_vector = {"subj_agent": self.subj_agent.people_id,
                                 "subj_agent_prof": self.subj_agent.profile,
                                 "object_agent": self.obj_agent.people_id,
                                 "obj_agent_prof": self.obj_agent.profile,
                                 "indirect_agent": indirect_agent_id,
                                 "action1":self.predicate_sub,
                                 "action2":action_2,
                                 "time_testimony": self.subj_agent.model.schedule.time,
                                 "time_observation":self.observation_time,
                                 "place_observer": self.subj_agent.pos,
                                 "place_observed_action": self.obj_agent_pos,
                                 "vision_perturb": self.vis_perturb,
                                 "observed_agent": self.observed_obj_agent_profile,
                                 "memory_perturb": self.memory_perturb,
                                 "memorized_agent": self.memorized_obj_agent_profile,
                                 "veracity_perturb": self.veracity_perturb,
                                 "testified_agent": self.testified_obj_agent_profile
                                 }



class ForestFire(mesa.Model):
    """
    Simple Forest Fire model.
    """

    def __init__(self, width=50, height=50, density=0.65):
        """
        Create a new forest fire model.

        Args:
            width, height: The size of the grid to model
            density: What fraction of grid cells have a tree in them.
        """
        # Set up model objects
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.MultiGrid(width, height, torus=False)

        self.place_agents = []
        self.people_agents = []
        self.data_collection_time = []
        self.collected_data_model_dict = {}

        self.relevant_data = []

        self.datacollector = mesa.DataCollector(
            {
                "Fine": lambda m: self.count_type(m, "Fine"),
                "On Fire": lambda m: self.count_type(m, "On Fire"),
                "Burned Out": lambda m: self.count_type(m, "Burned Out"),
            }
        )
        unique_id = 0
        self.ranger_num = 3

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
            if self.random.random() < density:
                # Create a tree
                new_tree = TreeCell(unique_id=unique_id, model=self)
                unique_id += 1
                # Set all trees in the first column on fire.
                if x == 0:
                    new_tree.condition = "On Fire"
                self.grid.place_agent(new_tree, (x, y))
                self.schedule.add(new_tree)
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
        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
        Advance the model by one step.
        """
        condition = "off"
        for agent in self.place_agents:
            agent.set_condition(condition)
            agent.set_owner = []

        self.schedule.step()
        #print("time", self.schedule.time)
        list_obs=self.who_saw_who()
        #print(list_obs)
        self.data_collection_time.append(list_obs)
        # collect data
        self.datacollector.collect(self)
        #self.collect_data()
        #print(self.data_collection_time)


        if self.schedule.time == 40: #%10 == 0:
            # ask for testimony
            self.get_testimony()

        # Halt if no more fire
        valuables_left = False
        for agent in self.people_agents:
            if agent.valuable > 0:
                valuables_left = True

        if not valuables_left:
            self.running = False
        #if self.count_type(self, "On Fire") == 0:
        #    self.running = False



    def get_testimony(self):
        for time_index in range(0, len(self.data_collection_time)):
            d_t = self.schedule.time - time_index
            for observation in self.data_collection_time[time_index]:
                if observation.predicate_obj is not None:
                    #print(observation)
                    #if observation.predicate_obj == "stealing from":
                    observation.memory_perturbation(d_t)
                    observation.veracity_perturbation()
                    #print(time_index,observation)
                observation.get_data_testimony()
                #print(observation.testimony_vector)


                self.relevant_data.append(observation.testimony_vector)
        #for agents in self.people_agents:


    # calculate who has seen who
    def who_saw_who(self):
        list_obs = []
        for people in self.people_agents:
            ppl_id = people.people_id
            agents_seen_only_bad, agents_seen_good = people.people_seen()

            # observations about self
            #if people.intent:
            #    n_o = ObservationStruc(people, people.target, None, "targeting", None, "good")
            #    list_obs.append(n_o)

            #if people.attempted_stealing:
            #    n_o = ObservationStruc(people, people.target, None, "attempting to steal from", None, "good")
            #    list_obs.append(n_o)
            if people.stealing:
                n_o = ObservationStruc(people, people.target, None, "stealing from", None, "good")
                list_obs.append(n_o)
            else:
                if people.target is None:
                    for people_not_targeting in self.people_agents:
                        n_o = ObservationStruc(people, people_not_targeting, None, "not stealing from", None, "good")
                        list_obs.append(n_o)

            for ag in agents_seen_good:

                if self.people_agents[ag].target is not None:

                    # observations about what is happening
                    observed_ag = self.people_agents[ag]
                    '''if observed_ag.intent:
                        n_o = ObservationStruc(people, observed_ag, observed_ag.target, "observed", "targeting", "good")
                        list_obs.append(n_o)
                    if observed_ag.attempted_stealing:
                        n_o = ObservationStruc(people, observed_ag, observed_ag.target, "observed", "attempting to steal from", "good")
                        list_obs.append(n_o)'''
                    if observed_ag.stealing:
                        observed_ag.was_observed_stealing_by.append((people, observed_ag.target, "good"))
                        #list_obs.append(n_o)
                        n_o = ObservationStruc(people, observed_ag, observed_ag.target, "observed", "stealing from", "good")
                        observed_ag.was_observed_stealing_by.append((people, observed_ag.target, "good"))
                        list_obs.append(n_o)
                    else:
                        n_o = ObservationStruc(people, observed_ag, observed_ag.target, "observed",
                                               "not stealing from",
                                               "good")
                        list_obs.append(n_o)

                n_o = ObservationStruc(people, self.people_agents[ag], None, "observed", None, "good")
                list_obs.append(n_o)

        for ag in agents_seen_only_bad:
            if self.people_agents[ag].target is not None:
                # observations about what is happening
                observed_ag = self.people_agents[ag]
                '''if observed_ag.intent:
                    n_o = ObservationStruc(people, observed_ag, observed_ag.target, "observed", "targeting", "bad")
                    list_obs.append(n_o)
                if observed_ag.attempted_stealing:
                    n_o = ObservationStruc(people, observed_ag, observed_ag.target, "observed",
                                           "attempting to steal from", "bad")
                    list_obs.append(n_o)'''

                if observed_ag.stealing:

                    n_o = ObservationStruc(people, observed_ag, observed_ag.target, "observed", "stealing from", "bad")
                    list_obs.append(n_o)
                    observed_ag.was_observed_stealing_by.append((people, observed_ag.target, "bad"))
                else:
                    n_o = ObservationStruc(people, observed_ag, observed_ag.target, "observed",
                                           "not stealing from",
                                           "bad")
                    list_obs.append(n_o)

            n_o = ObservationStruc(people, self.people_agents[ag], None, "observed", None, "bad")
            list_obs.append(n_o)

        for observed_ag in self.people_agents:
            if observed_ag not in agents_seen_good or agents_seen_only_bad:
                #print(observed_ag, observed_ag.target)

                if observed_ag.stealing == True and observed_ag.target is not None:
                    n_o = ObservationStruc(people, observed_ag, observed_ag.target, "not capable of observing", "stealing from", "bad")
                    list_obs.append(n_o)
                    n_o = ObservationStruc(people, observed_ag, observed_ag.target, "not observed", "stealing from", "bad")
                    list_obs.append(n_o)
                else:
                    n_o = ObservationStruc(people, observed_ag, None, "not capable of observing",
                                           None, "bad")
                    list_obs.append(n_o)
                    n_o = ObservationStruc(people, observed_ag, None, "not observed", None,
                                           "bad")
                    list_obs.append(n_o)


        return list_obs

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