import gym
from gym.envs.minigrid.wrappers import FullyObsWrapper,RGBImgPartialObsWrapper, ImgObsWrapper, FIRL_Mix2Levels_ObsWrapper, ImgObsFloatNormalizeWrapper
if __name__=="__main__":
    env =gym.make('MiniGrid-OpenDoors-7x7-v0',manual_set_door_color=["grey","blue",'yellow'])
    # env = FullyObsWrapper(env)
    env = RGBImgPartialObsWrapper(env, tile_size=6)
    env = ImgObsWrapper(env)
    env = ImgObsFloatNormalizeWrapper(env)
    env = FIRL_Mix2Levels_ObsWrapper(env)

    print(env.reset())
    print(env)