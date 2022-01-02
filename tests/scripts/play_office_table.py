import gym
ENABLE_OFFSCREEN_HELPER=True
if ENABLE_OFFSCREEN_HELPER:
    import rl_helper
    from rl_helper import envhelper
# from gym.envs.minigrid.envs.keycorridor import KeyCorridorS3R2

from rl_helper import fps
if __name__=="__main__":

    env = gym.make('FetchPickAndPlaceDense-v1')
    if ENABLE_OFFSCREEN_HELPER:
        recorder=envhelper()
    env.reset()
    for episode in range(10): 
        obs = env.reset()
        for step in range(15):
            action = env.action_space.sample()  # or given a custom model, action = policy(observation)
            # print()
            # action=[-1,0,0,0]
            nobs, reward, done, info = env.step(action)
            if ENABLE_OFFSCREEN_HELPER:
                recorder.recording(env,env.render('rgb_array'))
            else:
                env.render()
            if done:
                break

    if ENABLE_OFFSCREEN_HELPER:
        recorder.save_gif()
