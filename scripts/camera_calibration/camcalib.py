import os
import mujoco as mp
from mujoco import MjData, MjModel
import mujoco_viewer
from time import sleep
import numpy as np
import collections
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control import mujoco

model_dir = '../../third_party/mujoco-2.3.2/model/abb/irb_1600'
mjcf = 'irb1600_6_12_camcalib.xml'

dt = 1
manip_joints = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
calib_target_ori_range = np.array([[np.pi, 0, -np.pi/4], [np.pi/6, np.pi/6, np.pi/6]])
model = MjModel.from_xml_path(os.path.join(model_dir, mjcf))
data = MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)
phy = mujoco.Physics.from_xml_path(os.path.join(model_dir, mjcf))


def get_quaternion_from_euler(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qw, qx, qy, qz]

def getCalibTarget(m, d):
    id = mp.mj_name2id(m, mp.mjtObj.mjOBJ_GEOM, 'calib_target')
    calib_target = [m.geom_pos[id], m.geom_size[id]]
    return calib_target

def sampleFromCalibTarget(calib_target):
    seed = np.random.random_sample((3,))
    offset = 2*calib_target[1]*seed - calib_target[1]
    sample_xpos = calib_target[0] + offset

    ori_seed = np.random.random_sample((3,))
    ori_offset = 2*calib_target_ori_range[1]*seed - calib_target_ori_range[1]
    sample_rpy = calib_target_ori_range[0] + ori_offset
    sample_xquat = get_quaternion_from_euler(sample_rpy[0], sample_rpy[1], sample_rpy[2])
    # sample_xquat = [0, 0.3826, -0.9238, 0]
    return [sample_xpos, sample_xquat]

def sampleCalibrationPose(m, d, calib_target):
    s = sampleFromCalibTarget(calib_target)
    ik_result = qpos_from_site_pose(phy, 'calibtool_center', s[0], s[1], manip_joints, inplace=False)
    for i in range(100):
        if ik_result.success:
            break
        s = sampleFromCalibTarget(calib_target)
        ik_result = qpos_from_site_pose(phy, 'calibtool_center', s[0], s[1], manip_joints, inplace=False)
    return ik_result.qpos



if __name__ == "__main__":
    calib_target = getCalibTarget(model, data)
    count = 0
    while count < 100:
        s = sampleFromCalibTarget(calib_target)
        ik_result = qpos_from_site_pose(phy, 'calibtool_center', s[0], s[1], manip_joints, inplace=False)

        if viewer.is_alive and ik_result.success:
            count = count + 1
            data.qpos[:] = ik_result.qpos[:]
            mp.mj_step(model, data)
            viewer.render()
            sleep(dt)


