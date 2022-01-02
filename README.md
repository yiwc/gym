# GYM
This is an extented GYM environment from openai GYM. 

There are more envs, but the api style strictly remained as the original official verison. It will be long-term maintained up-to-date with openai GYM, as well as our new features.
# Usage:

    git clone https://github.com/yiwc/gym.git
    cd gym
    make install


# Envs included
* See all envs: [docs/envs](docs/envs)

* 2021-09-25 Integrated with gym-minigrid (https://github.com/maximecb/gym-minigrid)

* 2021-10-23 single-stage task - [MiniGrid-OpenOneDoor-7x7-v0](./tests/other_scripts/dev_opendoor.py)

* 2021-10-24  multi-stage task - [MiniGrid-OpenDoors-7x7-v0](./tests/other_scripts/dev_opendoor.py)

* 2022-01-02 multi-stage task - [OfficeTableRGB-v1],
* 2022-01-02 single-stage task - [OfficeTableR-v1],[OfficeTableG-v1],[OfficeTableB-v1],


# Quick Start:

* play with it:

        source scripts/manual_play.sh

* see how it looks like:
    
        python scripts/play_keydoor.py

