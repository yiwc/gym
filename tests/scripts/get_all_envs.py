from gym.envs.registration import registry
if __name__=="__main__":
    envs=registry.env_specs

    # for e in envs:
    #     print(e)
    with open("docs/envs",'w') as f:
        f.writelines([str(e)+"\n" for e in envs])