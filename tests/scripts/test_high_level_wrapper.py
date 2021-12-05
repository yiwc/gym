import gym
from gym.envs.minigrid.wrappers import FullyObsWrapper,RGBImgPartialObsWrapper

if __name__=="__main__":

    env =gym.make("MiniGrid-OpenOneDoor-Yellow7x7-v0")
    # env = FullyObsWrapper(env)
    env = RGBImgPartialObsWrapper(env)
    env.reset()
    print(env)