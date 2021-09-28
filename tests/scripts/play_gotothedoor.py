import gym
import rl_helper
from rl_helper import fps
from gym.envs.minigrid.envs.keycorridor import KeyCorridorS3R2
if __name__=="__main__":

    env = gym.make('MiniGrid-GoToDoor-5x5-v0')
    env.reset()
    print(env.metadata)
    fps(env)
    frames=[]
    for episode in range(2): 
        obs = env.reset()
        for step in range(15):
            # print(step)
            action = env.action_space.sample()  # or given a custom model, action = policy(observation)
            nobs, reward, done, info = env.step(action)
            # if done.any():
            #     break
            if done:
                break
            print(nobs['image'].mean())
            print(nobs['image'].shape)
            # env.render()
    # env.close()