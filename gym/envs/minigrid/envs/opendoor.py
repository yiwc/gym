from gym.envs.minigrid.minigrid import *
from gym.envs.minigrid.register import register
import random
import numpy as np
class OpenOneDoor(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(
        self,
        size=6,
        numObjs=3,
        manual_set_door_color=None,
    ):
        self.numObjs = numObjs

        self.manual_set_door_color=manual_set_door_color
        super().__init__(
            grid_size=size,
            max_steps=3*(size-2),
            see_through_walls=True
        )


    def set_target_door(self,color):
        assert color in COLOR_NAMES, COLOR_NAMES
        self.manual_set_door_color=color

    def _gen_grid(self, width, height):
        assert self.manual_set_door_color is not None, 'please use make("env_name",manual_set_door_color="red")'

        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Types and colors of objects we can generate
        types = ['door']

        objs = []
        objPos = []

        colors = []
        while self.manual_set_door_color not in colors:
            colors = self._rand_subset(COLOR_NAMES[:],self.numObjs)
        
        # objIdx=-1
        # self.target_obj
        while len(objs) < self.numObjs:
            objType = self._rand_elem(types)
            objColor = colors.pop()

            # If this object already exists, try again
            if (objType, objColor) in objs:
                continue

            elif objType == 'door':
                obj = Door(objColor)

            pos = self.place_obj(obj)
            objs.append((objType, objColor))
            objPos.append(pos)
            if objColor == self.manual_set_door_color:
                # targetOjbIdx=(objType, objColor)
                self.targetType = objType
                self.target_color = objColor
                self.target_pos = pos

        # Randomize the agent start position and orientation
        self.place_agent()

        descStr = '%s %s' % (self.target_color, self.targetType)
        self.mission = 'go to the %s' % descStr


    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        is_target_door_open=self._obj_attr_is_door_open(self.target_pos)
        if is_target_door_open:
            reward = 1
            done = True

        return obs, reward, done, info

class OpenOneDoor5x5Env(OpenOneDoor):
    def __init__(self,**kwargs):
        super().__init__(size=7,**kwargs)

class OpenOneDoor7x7Env(OpenOneDoor):
    def __init__(self,**kwargs):
        super().__init__(size=9,**kwargs)

# class OpenOneDoor9x9Env(OpenOneDoor):
#     def __init__(self):
#         super().__init__(size=9)
