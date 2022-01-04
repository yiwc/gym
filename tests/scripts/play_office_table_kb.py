import gym
from rl_helper.keyboard import kbinteractor
import time
from rl_helper import fps
if __name__=="__main__":

    env = gym.make('OfficeTableB-v1',senstive=2,action_scale=0.3)

    # print(env.action_space)
    # print(env.observation_space)
    # print(env.action_space)
    # env.reset()
    # fps(env)

    # env.reset()
    # for i in range(500):
    #     obs,rew,done,info=env.step(env.action_space.sample())
    #     print("reward:",rew)
    #     if rew==0:
    #         print(rew)
    #         pass
    #     time.sleep(0.01)
    #     env.render()  
    #     if i % 10 ==0:
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