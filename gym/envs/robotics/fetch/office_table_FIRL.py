import os
from gym import utils
from gym.envs.robotics import office_table


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "office_table.xml")



class OfficeTablePickAndPlaceFIRLEnv(office_table.OfficeTable, utils.EzPickle):
    def __init__(self, reward_type="sparse", target_objects="RGB",senstive=1):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.2, 0.5, 0.4, 1.0, 0.0, 0.0, 0.0],
            "object1:joint": [1.2, 0.7, 0.4, 1.0, 0.0, 0.0, 0.0],
            "object2:joint": [1.2, 0.9, 0.4, 1.0, 0.0, 0.0, 0.0]
        }
        assert len(target_objects),"please key in target_objects [R,G,B,RGB]"
        # for k in target_objects:
        assert target_objects in ['R',"G","B","RGB"]
        office_table.OfficeTable.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            target_objects=target_objects,
            all_objects="RGB",
            senstive=senstive
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)


class OfficeTableRPickAndPlaceFIRLEnv(OfficeTablePickAndPlaceFIRLEnv):
    def __init__(self, reward_type="sparse",senstive=1):
        target_objects="R"
        super().__init__(reward_type=reward_type, target_objects=target_objects,senstive=senstive)

class OfficeTableGPickAndPlaceFIRLEnv(OfficeTablePickAndPlaceFIRLEnv):
    def __init__(self, reward_type="sparse",senstive=1):
        target_objects="G"
        super().__init__(reward_type=reward_type, target_objects=target_objects,senstive=senstive)

class OfficeTableBPickAndPlaceFIRLEnv(OfficeTablePickAndPlaceFIRLEnv):
    def __init__(self, reward_type="sparse",senstive=1):
        target_objects="B"
        super().__init__(reward_type=reward_type, target_objects=target_objects,senstive=senstive)

class OfficeTableRGBPickAndPlaceFIRLEnv(OfficeTablePickAndPlaceFIRLEnv):
    def __init__(self, reward_type="sparse",senstive=1):
        target_objects="RGB"
        super().__init__(reward_type=reward_type, target_objects=target_objects,senstive=senstive)