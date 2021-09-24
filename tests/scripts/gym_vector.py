import rl_helper
# from rl_helper import fps
from rl_helper import fps
import gym
if __name__=="__main__":
    env = gym.vector.make('CartPole-v1', 10,asynchronous=True)  # Creates an Asynchronous env
    env.reset()
    fps(env)