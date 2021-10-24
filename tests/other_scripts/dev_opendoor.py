# 2021/10/23
# Developing the opendoor env

import gym
# from gym.envs.minigrid.envs.keycorridor import KeyCorridorS3R2
# import keyboard  # using module keyboard
import time
def play_sequence(env,actions,render=False,start_with_reset=False,seed=0,assert_reward=None):
    if start_with_reset:
        env.seed(seed)
        env.reset()
    for str_action in actions:
        action = getattr(env.actions,str_action)
        nobs, reward, done, info = env.step(action)
        if assert_reward:
            assert round(reward,3)==round(assert_reward,3),"{} {}".format(reward,assert_reward)
        if render:
            env.render()
            time.sleep(0.1)
        if done:
            break
    return nobs,reward,done,info
if __name__=="__main__":
    render=False


    
    # # Test max steps = 21
    env = gym.make('MiniGrid-OpenDoors-7x7-v0',manual_set_door_color=["grey","blue",'yellow'])
    nobs,reward,done,info=play_sequence(env,actions=['right' for i in range(4*7)],render=False,start_with_reset=True,seed=0)
    assert done==1
    nobs,reward,done,info=play_sequence(env,actions=['right' for i in range(4*7-1)],render=False,start_with_reset=True,seed=0)
    assert done==0
    env.close()

    env = gym.make('MiniGrid-OpenDoors-7x7-v0',manual_set_door_color=["grey","blue",'yellow'])
    nobs,reward,done,info=play_sequence(env,actions=['right','right','forward','forward','forward','forward','right'],render=render,start_with_reset=True,seed=0,assert_reward=0)
    assert done==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=None)
    assert done==0
    assert reward ==1 
    nobs,reward,done,info=play_sequence(env,actions=['left','forward','right','forward','forward','forward'],render=render,start_with_reset=False,seed=0,assert_reward=0)
    assert done==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=None)
    assert done==0
    assert reward ==1 
    nobs,reward,done,info=play_sequence(env,actions=['left','left','forward','forward','right'],render=render,start_with_reset=False,seed=0,assert_reward=0)
    assert done==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=None)
    assert done==1
    assert reward ==1 
    env.close()

    env = gym.make('MiniGrid-OpenDoors-7x7-v0',manual_set_door_color=["red","blue"])
    nobs,reward,done,info=play_sequence(env,actions=['right','right','forward','forward','forward','forward','forward','right','forward','forward','forward','forward','forward'],render=render,start_with_reset=True,seed=0,assert_reward=0)
    assert done==0
    assert reward==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=1)
    assert done==0
    assert reward ==1 
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=0)
    assert done==0
    assert reward ==0
    nobs,reward,done,info=play_sequence(env,actions=['right','right','forward','forward','right'],render=render,start_with_reset=False,seed=0,assert_reward=0)
    assert done==0
    assert reward ==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=None)
    assert done==1
    assert reward ==0
    env.close()


    env = gym.make('MiniGrid-OpenDoors-7x7-v0',manual_set_door_color=["red","blue"])
    nobs,reward,done,info=play_sequence(env,actions=['right','right','forward','forward','forward','forward','forward','right','forward','forward','forward','forward','forward'],render=render,start_with_reset=True,seed=0,assert_reward=0)
    assert done==0
    assert reward==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=1)
    assert done==0
    assert reward ==1 
    nobs,reward,done,info=play_sequence(env,actions=['right','right','forward','forward','right'],render=render,start_with_reset=False,seed=0,assert_reward=0)
    assert done==0
    assert reward ==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=None)
    assert done==1
    assert reward ==1
    env.close()

    env = gym.make('MiniGrid-OpenDoors-7x7-v0',manual_set_door_color=["yellow","red"])
    nobs,reward,done,info=play_sequence(env,actions=['right','right','forward','forward','forward','forward','forward','right','forward','forward','forward','forward','forward'],render=render,start_with_reset=True,seed=0,assert_reward=0)
    assert done==0
    assert reward==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=1)
    assert done==0
    assert reward ==1 
    nobs,reward,done,info=play_sequence(env,actions=['right','right','forward','forward','right'],render=render,start_with_reset=False,seed=0,assert_reward=0)
    assert done==0
    assert reward ==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=None)
    assert done==1
    assert reward ==1
    env.close()


    env = gym.make('MiniGrid-OpenOneDoor-5x5-v0',manual_set_door_color="yellow")
    
    nobs,reward,done,info=play_sequence(env,actions=['forward','forward','forward'],render=render,start_with_reset=True,seed=0,assert_reward=0)
    assert done==0
    assert reward==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=1)
    assert done==1
    assert reward ==1 
    env.close()


    env = gym.make('MiniGrid-OpenOneDoor-5x5-v0',manual_set_door_color="red")
    
    nobs,reward,done,info=play_sequence(env,actions=['forward','forward','forward','left','forward','forward','forward'],render=render,start_with_reset=True,seed=0,assert_reward=0)
    assert done==0
    assert reward==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=1)
    assert done==1
    assert reward ==1 


    # Test max steps = 3*5 = 15
    nobs,reward,done,info=play_sequence(env,actions=['right' for i in range(3*5)],render=False,start_with_reset=True,seed=0)
    assert done==1
    nobs,reward,done,info=play_sequence(env,actions=['right' for i in range(3*5-1)],render=False,start_with_reset=True,seed=0)
    assert done==0

    env.close()


    env = gym.make('MiniGrid-OpenOneDoor-7x7-v0',manual_set_door_color="yellow")
    nobs,reward,done,info=play_sequence(env,actions=['right','right','forward','forward','left','forward','forward','forward','forward'],render=render,start_with_reset=True,seed=0,assert_reward=0)
    assert done==0
    assert reward==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=1)
    assert done==1
    assert reward ==1 
    env.close()


    env = gym.make('MiniGrid-OpenOneDoor-7x7-v0',manual_set_door_color="red")
    nobs,reward,done,info=play_sequence(env,actions=['right','right','forward','forward','forward','forward',
    'forward',"right",'forward','forward','forward','forward','forward'],render=render,start_with_reset=True,seed=0,assert_reward=0)
    assert done==0
    assert reward==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=1)
    assert done==1
    assert reward ==1 

    nobs,reward,done,info=play_sequence(env,actions=['forward','forward','forward','forward'],render=render,start_with_reset=True,seed=1,assert_reward=0)
    assert done==0
    assert reward==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=1,assert_reward=1)
    assert done==1
    assert reward ==1 
    env.close()




    env = gym.make('MiniGrid-OpenOneDoor-7x7-v0',manual_set_door_color="blue")
    nobs,reward,done,info=play_sequence(env,actions=['right','right','forward','forward','forward','forward',
    'forward',"right",'forward','forward','forward','forward','forward'],render=render,start_with_reset=True,seed=0,assert_reward=0)
    assert done==0
    assert reward==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=0)
    assert done==0
    assert reward ==0 
    nobs,reward,done,info=play_sequence(env,actions=['right','right','forward','forward','forward','forward', "right"],render=render,start_with_reset=True,seed=0,assert_reward=0)
    assert done==0
    assert reward==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=0,assert_reward=1)
    assert done== 1
    assert reward ==1
    nobs,reward,done,info=play_sequence(env,actions=['forward','forward','forward'],render=render,start_with_reset=True,seed=1,assert_reward=0)
    assert done==0
    assert reward==0
    nobs,reward,done,info=play_sequence(env,actions=['toggle'],render=render,start_with_reset=False,seed=1,assert_reward=1)
    assert done==1
    assert reward ==1
    
    # Test max steps = 21
    nobs,reward,done,info=play_sequence(env,actions=['right' for i in range(3*7)],render=False,start_with_reset=True,seed=0)
    assert done==1
    nobs,reward,done,info=play_sequence(env,actions=['right' for i in range(3*7-1)],render=False,start_with_reset=True,seed=0)
    assert done==0
    env.close()
