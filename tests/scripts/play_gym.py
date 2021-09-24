import gym
if __name__=="__main__":

    env = gym.make('CartPole-v1')
    print(env.metadata)

    frames=[]
    for episode in range(2): 
        obs = env.reset()
        for step in range(15):
            # print(step)
            action = env.action_space.sample()  # or given a custom model, action = policy(observation)
            nobs, reward, done, info = env.step(action)
            if done:
                break
            env.render()
    env.close()