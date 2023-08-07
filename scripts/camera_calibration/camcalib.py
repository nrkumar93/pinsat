import os
import mujoco as mp
from mujoco import MjData, MjModel
import mujoco_viewer
from time import sleep
import numpy as np
import collections
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control import mujoco
import pickle

# zed
import cv2
import pyzed.sl as sl

# ROS
import rospy
import actionlib
from pydrake.all import BsplineTrajectory, KinematicTrajectoryOptimization, Solve
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState

sim = True

# model
model_dir = '/home/shield/code/parallel_search/third_party/mujoco-2.3.2/model/abb/irb_1600/'
mjcf = 'irb1600_6_12_camcalib.xml'

# viewer params
viz_dt = 1

# dm_control
manip_joints = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
calib_target_ori_range = np.array([[np.pi, 0, -np.pi/4], [np.pi/6, np.pi/6, np.pi/6]])
model = MjModel.from_xml_path(os.path.join(model_dir, mjcf))
data = MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)
phy = mujoco.Physics.from_xml_path(os.path.join(model_dir, mjcf))

# camera settings
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD2K
init_params.camera_fps = 30
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(-1)

# output save dir
save_dir = 'calib_data'

# planner params
nq = 6
qmin = np.array([-3.14159, -1.0995, -4.1015, -3.4906, -2.0071, -6.9813])
qmax  = np.array([3.14159, 1.9198, 0.9599, 3.4906, 2.0071, 6.9813])
dqmin = np.array([-2.618, -2.7925, -2.967, -5.585, -6.9813, -7.854])
dqmax = np.array([2.618, 2.7925, 2.967, 5.585, 6.9813, 7.854])
ddqmin = -2*np.ones(nq)
ddqmax = 2*np.ones(nq)
dddqmin = -10*np.ones(nq)
dddqmax = 10*np.ones(nq)
dt = 4e-3


# ROS subscriber
curr_state = JointTrajectoryControllerState()
def robotStateCallback(data):
    curr_state = data

# execution
with open('/home/shield/code/shield_ws/src/abb_robot_driver/abb_robot_bringup_examples/scripts/calib_start.pkl', 'rb') as file:
    start_tpva = pickle.load(file)

with open('/home/shield/code/shield_ws/src/abb_robot_driver/abb_robot_bringup_examples/scripts/calib_end.pkl', 'rb') as file:
    end_tpva = pickle.load(file)
calib_end_start = np.array([0, 1.57, -1.57, 0, 0, 0])

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

def generateTrajectory(q0, qF):
    wp = np.array([])
    trajopt = KinematicTrajectoryOptimization(nq, 10, 4)
    prog = trajopt.get_mutable_prog()
    trajopt.AddDurationCost(10.0)
    trajopt.AddPathLengthCost(1.0)
    trajopt.AddPositionBounds(qmin, qmax)
    trajopt.AddVelocityBounds(dqmin, dqmax)
    trajopt.AddAccelerationBounds(ddqmin, ddqmax)
    trajopt.AddJerkBounds(ddqmin, ddqmax)
    trajopt.AddDurationConstraint(0.5, 25)
    trajopt.AddPathPositionConstraint(q0, q0, 0)
    trajopt.AddPathPositionConstraint(qF, qF, 1)
    prog.AddQuadraticErrorCost(np.eye(nq), q0, trajopt.control_points()[:, -1])
    trajopt.AddPathVelocityConstraint(np.zeros((nq, 1)), np.zeros((nq, 1)), 0) # start and end with zero velocity
    trajopt.AddPathVelocityConstraint(np.zeros((nq, 1)), np.zeros((nq, 1)), 1) # start and end with zero velocity

    ds = 1.0/(np.shape(wp)[0]+1)
    for i, r in zip(range(np.shape(wp)[0]), wp):
        trajopt.AddPathPositionConstraint(r, r, (i+1)*ds)

    # Solve once without the collisions and set that as the initial guess for
    # the version with collisions.
    result = Solve(prog)
    if not result.is_success():
        print("Trajectory optimization failed, even without collisions!")
    print("trajopt succeeded!")
    op_traj = trajopt.ReconstructTrajectory(result)
    print('traj duration: ', op_traj.end_time())

    tpva = np.empty((int(np.ceil(op_traj.end_time()/dt)), 1+3*nq))
    test_goal = FollowJointTrajectoryGoal()
    goal_traj = JointTrajectory()
    goal_traj.header.frame_id = 'odom_combined'
    goal_traj.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

    j = 0
    for t in np.arange(op_traj.start_time(), op_traj.end_time(), dt):
        p = JointTrajectoryPoint()
        p.time_from_start.secs = int(t)
        p.time_from_start.nsecs = int(rospy.Time.from_sec(t).to_nsec() % 1e9)
        p.positions = [0,0,0,0,0,0]
        p.velocities = [0,0,0,0,0,0]
        p.accelerations = [0,0,0,0,0,0]

        tpva[j, 0] = t

        for i in range(nq):
            p.positions[i] = op_traj.value(t)[i][0]
            # p.velocities[i] = op_traj.EvalDerivative(t, 1)[i][0]
            # p.accelerations[i] = op_traj.EvalDerivative(t, 2)[i][0]

            tpva[j, 1+i] = op_traj.value(t)[i][0]
            tpva[j, 1+nq+i] = op_traj.EvalDerivative(t, 1)[i][0]
            tpva[j, 1+2*nq+i] = op_traj.EvalDerivative(t, 2)[i][0]

        j += 1

    return tpva


def isValidTrajectory(traj):
    for wp in traj:
        for i in range(model.nq):
            data.qpos[i] = wp[1+i]

    mp.mj_forward(model, data)

    if data.ncon > 0:
        return False
    else:
        return True

def executeTraj(tpva):
    duration = tpva[-1][0]

    test_goal = FollowJointTrajectoryGoal()
    goal_traj = JointTrajectory()
    goal_traj.header.frame_id = 'odom_combined'
    goal_traj.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

    for r in tpva:
        p = JointTrajectoryPoint()
        p.time_from_start.secs = int(r[0])
        p.time_from_start.nsecs = int(rospy.Time.from_sec(r[0]).to_nsec() % 1e9)
        p.positions = [0,0,0,0,0,0]
        p.velocities = [0,0,0,0,0,0]
        p.accelerations = [0,0,0,0,0,0]

        for i in range(nq):
            p.positions[i] = r[1+i]
            # p.velocities[i] = r[1+nq+i]
            # p.accelerations[i] = r[1+2*nq+i]
        goal_traj.points.append(p)

    test_goal.trajectory = goal_traj
    client.send_goal_and_wait(test_goal)
    client.wait_for_result(rospy.Duration.from_sec(duration+2))
    rospy.spin()

def visualize(tpva):
    viz_dt = dt
    if viewer.is_alive:
        for wp in tpva:
            data.qpos[:] = wp[1:model.nq+1]
            mp.mj_step(model, data)
            viewer.render()
            sleep(viz_dt)
    curr_state.actual.positions = tpva[-1][1:model.nq+1]
    sleep(0.5)

def clickAndSavePic(id):
    limg = sl.Mat()
    rimg = sl.Mat()
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # A new image is available if grab() returns SUCCESS
        lret = zed.retrieve_image(limg, sl.VIEW.LEFT) # Retrieve the left image
        rret = zed.retrieve_image(rimg, sl.VIEW.RIGHT) # Retrieve the left image
        if lret == sl.ERROR_CODE.SUCCESS and rret == sl.ERROR_CODE.SUCCESS:
          f = os.path.join(save_dir, 'left', str(id)+'.png')
          limg.write(f)
          f = os.path.join(save_dir, 'right', str(id)+'.png')
          rimg.write(f)

if __name__ == "__main__":

    rospy.init_node('camcalib_node')
    rospy.Subscriber("/egm/joint_velocity_trajectory_controller/state", JointTrajectoryControllerState, robotStateCallback)
    client = actionlib.SimpleActionClient('egm/joint_velocity_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    # client = actionlib.SimpleActionClient('egm/joint_position_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    if not client.wait_for_server(rospy.Duration.from_sec(1.0)):
        print("joint_trajectory_action server not available")
    print("Connected to follow_joint_trajectory server")

    calib_target = getCalibTarget(model, data)

    # go to start calib pose
    if isValidTrajectory(start_tpva):
        if sim:
            visualize(start_tpva)
        else:
            executeTraj(start_tpva)
    else:
        exit(0)

    count = 0
    while count < 100:
        s = sampleFromCalibTarget(calib_target)
        ik_result = qpos_from_site_pose(phy, 'calibtool_center', s[0], s[1], manip_joints, inplace=False)
        if ik_result.success:
            print(curr_state.actual.positions)
            tpva = generateTrajectory(curr_state.actual.positions, ik_result.qpos)
            if isValidTrajectory(tpva):
                count = count + 1
                if sim:
                    visualize(tpva)
                else:
                    executeTraj(tpva)
                    clickAndSavePic(count)
            else:
                print('Sampling new pose to avoid collision')

    tpva = generateTrajectory(curr_state.actual.positions, calib_end_start)
    if isValidTrajectory(tpva):
        if sim:
            visualize(tpva)
        else:
            executeTraj(tpva)

    if isValidTrajectory(end_tpva):
        if sim:
            visualize(end_tpva)
        else:
            executeTraj(end_tpva)





