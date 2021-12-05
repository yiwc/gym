from gym.envs.minigrid.minigrid import *
from gym.envs.minigrid.register import register
import random
import numpy as np

class OpenDoors(MiniGridEnv):
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
            max_steps=(len(manual_set_door_color)+1)*(size-2),
            see_through_walls=True
        )


    def get_manual_set_door_color(self):
        return self.manual_set_door_color

    def set_target_door(self,color):
        assert color in COLOR_NAMES, COLOR_NAMES
        self.manual_set_door_color=color

    def _gen_grid(self, width, height):
        assert self.manual_set_door_color is not None, 'please use make("env_name",manual_set_door_color="red")'
        assert isinstance(self.manual_set_door_color,list), "only accept list . eg.['red','yellow','blue']"

        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Types and colors of objects we can generate
        types = ['door']

        objs = []
        objPos = []

        colors = self.manual_set_door_color[:]
        for color in colors:
            assert color in COLOR_NAMES, "do not support color: %s" % color
        other_colors=list(set(COLOR_NAMES[:])-set(colors))
        assert self.numObjs>=len(colors)


        objType='door'
        
        self.target_objs_pos={}
        while len(objs) < self.numObjs:
            if len(colors):
                objColor = colors.pop()    
            else:
                objColor = self._rand_elem(other_colors)

            obj = Door(objColor)
            pos = self.place_obj(obj)
            objs.append((objType, objColor))
            objPos.append(pos)
            if objColor in self.manual_set_door_color:
                self.target_objs_pos[objColor]=pos

        # Randomize the agent start position and orientation
        self.place_agent()

        descStr = '%s %s' % (" ".join(self.manual_set_door_color), objType)
        self.mission = 'go to the %s' % descStr

        self.rewarded=0

    def get_target_doors_state(self):
        target_doors_state=[self._obj_attr_is_door_open(self.target_objs_pos[color]) for color in self.manual_set_door_color]
        return target_doors_state

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        target_doors_state=self.get_target_doors_state()
        if sum(target_doors_state)>self.rewarded:
            if target_doors_state[self.rewarded]==True:
                reward = 1
                self.rewarded+=1
            else:
                reward = -1
                done=True
        if self.rewarded >= len(target_doors_state):
            done=True
            
        return obs, reward, done, info

# class OpenDoors5x5Env(OpenOneDoor):
#     def __init__(self,**kwargs):
#         super().__init__(size=7,**kwargs)

class OpenDoors7x7Env(OpenDoors):
    def __init__(self,**kwargs):
        super().__init__(size=9,**kwargs)
