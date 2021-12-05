#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym.envs.minigrid
from gym.envs.minigrid.wrappers import *
from gym.envs.minigrid.window import Window

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)
        args.seed+=1

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f, done=%d' % (env.step_count, reward, done))
    # print(obs['image'][0,:])
    if done:
        # print('done!')
        reset()
    else:
        redraw(obs)

def key_handler(event):
    keys={
        'done':"enter",
        "toggle":" ",
        "pickup":"x",
        "drop":"c",
        "left":"left",
        "right":'right',
        'forward':'up',
        'done':'enter'
    }
    print("how to use it : \n",keys)
    # print('pressed', event.key)
    
    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == keys['left']:
        step(env.actions.left)
        return
    if event.key == keys['right']:
        step(env.actions.right)
        return
    if event.key == keys['forward']:
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == keys['toggle']:
        step(env.actions.toggle)
        return
    if event.key == keys['pickup']:
        step(env.actions.pickup)
        return
    if event.key == keys['drop']:
        step(env.actions.drop)
        return

    if event.key == keys['done']:
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--env",
#     help="gym environment to load",
#     default='MiniGrid-OpenDoors-7x7-v0'
# )
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=0
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

args = parser.parse_args()

# env=
# env = gym.make("MiniGrid-OpenOneDoor-7x7-v0",manual_set_door_color="red")
# env = gym.make("MiniGrid-OpenDoors-7x7-v0",manual_set_door_color=['blue',"red"])
# env = gym.make("MiniGrid-OpenDoors-7x7-v0",manual_set_door_color=['blue',"red",'yellow'])
env = gym.make("MiniGrid-OpenDoors-7x7-v0",manual_set_door_color=['red','blue','yellow'])

# if args.agent_view:
#     env = RGBImgPartialObsWrapper(env)
    # env = ImgObsWrapper(env)


env = RGBImgPartialObsWrapper(env, tile_size=6)
env = ImgObsWrapper(env)
env = ImgObsFloatNormalizeWrapper(env)
env = FIRL_Mix2Levels_ObsWrapper(env)


window = Window('gym_minigrid - ')
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
