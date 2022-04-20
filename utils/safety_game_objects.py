from enum import Enum
import random

"""
The following classes are the types of objects that we are currently supporting 
"""

class Entity:
    def __init__(self,i,j): #row and column
        self.i = i
        self.j = j

    def change_position(self,i,j):
        self.i = i
        self.j = j
        
    def idem_position(self,i,j):
        return self.i==i and self.j==j

    def interact(self, agent):
        return True

class Agent(Entity):
    def __init__(self,i,j,actions):
        super().__init__(i,j)
        self.num_keys = 0
        self.reward  = 0
        self.actions = actions
        self.observation = []

    def get_actions(self):
        return self.actions

    def interact(self, agent):
        return False

    def update_reward(self, r):	
        self.reward += r

    def __str__(self):
        return "A"

class Obstacle(Entity):
    def __init__(self,i,j):
        super().__init__(i,j)

    def interact(self, agent):
        return False

    def __str__(self):
        return "X"

class Empty(Entity):
    def __init__(self,i,j,label=" "):
        super().__init__(i,j)
        self.label = label

    def __str__(self):
        return self.label


"""
Enum with the actions that the agent can execute
"""
class Actions(Enum):
    # cautious movement
    S_up    = 0 # Slow move up
    S_right = 1 # Slow move right
    S_down  = 2 # Slow move down
    S_left  = 3 # Slow move left

    # # Normal movement
    # N_up    = 4 # Normal move up
    # N_right = 5 # Normal move right
    # N_down  = 6 # Normal move down
    # N_left  = 7 # Normal move left

    # F_up    = 8 # Slow move up
    # F_right = 9 # Slow move right
    # F_down  = 10 # Slow move down
    # F_left  = 11 # Slow move left
    # wait = 4
