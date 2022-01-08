from logging import debug
from re import DEBUG
import gym
from rl_helper.keyboard import kbinteractor
import time
from rl_helper import fps

from gym.envs.robotics.office_table import TASK_NEAR_PICK, TASK_PICK_PLACE, TASK_REACH
if __name__=="__main__":

    # env = gym.make('OfficeTable-v1',target_objects="G",posrel_reciprocal=True,reward_scale=1,task=TASK_NEAR_PICK,DEBUG=True)
    env = gym.make('OfficeTable-v1',target_objects="G",task=TASK_REACH)

    # print(env.action_space)
    # print(env.observation_space)
    # print(enhhhhhhhhhhhhhhhhhhhyv.action_space)
    # env.reset()hhhh
    # fhhps(hhyenv)

    # env.reset()hhy
    # for i hin hrhyanyghhhhe(int(1e8)):
    #     obs,rew,done,info=env.step(env.action_space.sample())
    #     print("reward:",rew)
    #     if rew==0:hhhhhhhh
    #         pryyint(rew)
    #         pass
    #     time.sleep(0.01)
    #     env.render()  
    #     hhif i % 10 ==0:
    #         env.reset()

    kbinteractor(env,key2action={
                    'right':[1,0,0,0],
                    'left':[-1,0,0,0],
                    'up':[0,1,0,0],
                    'down':[0,-1,0,0],
                    "y":[0,0,1,0],
                    "h":[0,0,-1,0],
                    "n":[0,0,0,1],
                    "b":[0,0,0,-1],
                })
